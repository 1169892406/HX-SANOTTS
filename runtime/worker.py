#!/usr/bin/env python3
"""HX-SANOTTS Worker —— 本地播报（sanoTTS）插件的主进程。

# 与宿主的关系

宿主以 `python3.12 -u <entry>` 拉起本文件，cwd = 插件物化目录，
PYTHONPATH = 同一目录，通过 **stdin/stdout 的 JSON lines** 收发（与 HXCS 同一协议）：

    请求  {"id":3,"op":"speak","text":"打开客厅灯"}
    应答  {"ok":true,"result":{…}}   /   {"ok":false,"error":{"code":…,"message":…}}

`op` 就是 manifest 里声明的工具名（prepare/health/drain 是宿主保留的控制面）。
一轮调用**串行**（宿主侧同一实例加锁），所以这里不需要给状态加锁。

# 为什么音频要分块

宿主不是唯一调用方：插件自带的配置页也调它（`plugin.tool.call`），而两者对单次
应答都有量级限制。一句中文 22050 Hz/16-bit 大约 44 KB/秒，十秒就 440 KB —— 一次
塞回去会让"能听"和"传不出去"互相矛盾。所以：

  · `speak` 先合成，把 WAV 整段留在内存（按 utterance_id 索引，保留最近 4 条），
    回执里只带**第一块** + 总块数/总字节/时长；
  · 调用方随后用 `audio_chunk {utterance_id, index}` 逐块取（默认 48 KiB/块，
    base64 后约 64 KiB）。

# 线程与内存

不建线程、不开 socket、不写临时文件：一份 .so（C 推理）+ 一段 PCM 就在本进程内存里。
`max_memory_mb=256` 对 1.55M 参数的模型 + 单条 utterance 足够（实测峰值 < 40 MB）。
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sanotts_zh as zh  # noqa: E402  （本地模块，见同目录 sanotts_zh.py）

VERSION = "1.0.0"
CHUNK_BYTES = 48 * 1024
KEEP_UTTERANCES = 4

# 模型一次能吃的音素 id 上限（语音包里 duration config 的 max_tokens=432）。
#
# ## 为什么字数上限必须由它推出来（2026-11-08 实测）
#
# 中文每个音节占 4 个 id（声母 + 韵母 + 声调 + 一个 PAD），所以 432 个 id 只够
# **约 108 个汉字**。原先 manifest 把 max_chars 默认写成 200、上限 400，于是
# 300 字的输入顺利通过字数校验，然后在 C 引擎里报
# 「音素 id 过多：1202 > max_tokens 432」—— 用户看到的是一条内部错误，
# 而不是"这段话太长，请拆成两句"。
#
# 正确的分工：模型能力是硬上限（不可配），配置只能在它下面收窄。
MAX_TOKENS = 432
IDS_PER_HAN = 4
ENGINE_CHAR_CAP = MAX_TOKENS // IDS_PER_HAN   # 108


def _log(msg: str) -> None:
    """日志走 stderr：stdout 是协议通道，写一个字节都会破坏 JSON lines。"""
    print(f"[sanotts] {msg}", file=sys.stderr, flush=True)


class Engine:
    """一份加载好的语音包 + C 推理库。"""

    def __init__(self) -> None:
        self.lib = None
        self.table: zh.PhonemeTable | None = None
        self.voice = "zh"
        self.sample_rate = 22050
        self.model_dir = HERE / "models" / "zh"
        self.loaded_at = 0.0
        self.load_error = ""
        self.stats = {"synth_calls": 0, "last_ms": 0.0, "last_frames": 0}

    # ---- 装载 ----

    def load(self) -> None:
        if self.lib is not None:
            return
        started = time.time()
        lib_path = HERE / "libsanotts.so"
        if not lib_path.is_file():
            raise RuntimeError(
                f"缺少推理库 {lib_path.name}：本插件是源码 + 预编译库的包，"
                "若平台架构不是 linux/amd64，请按 BUILD.md 重编"
            )
        lib = ctypes.CDLL(str(lib_path))
        lib.snt_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.snt_load.restype = ctypes.c_int
        lib.snt_synth.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        lib.snt_synth.restype = ctypes.c_int
        lib.snt_free.argtypes = [ctypes.c_void_p]
        lib.snt_free.restype = None
        lib.snt_error.restype = ctypes.c_char_p
        lib.snt_sample_rate.restype = ctypes.c_int

        layout = self.model_dir / "layout.bin"
        weights = self.model_dir / "weights.fp16.bin"
        if lib.snt_load(str(layout).encode(), str(weights).encode()) != 0:
            raise RuntimeError(f"权重装载失败：{lib.snt_error().decode('utf-8', 'replace')}")
        self.lib = lib
        self.sample_rate = int(lib.snt_sample_rate())
        self.table = zh.load_phoneme_table(self.model_dir / "piper-phoneme-config.json")
        if not zh.voice_supports(self.table):
            raise RuntimeError(
                f"语音包 phoneme_type={self.table.phoneme_type!r}：本插件只实现中文 pinyin 前端"
            )
        # 预热：首次调用会把整份音素词典解压（xz → 41923 条字表 + 38355 条词组），
        # 约 0.3 s。放在装载里做，避免用户点第一次"播报"才等。
        zh.shared_lexicon()
        self.loaded_at = time.time()
        _log(f"engine ready in {self.loaded_at - started:.2f}s sr={self.sample_rate}")

    # ---- 合成 ----

    def synthesize(self, text: str, speed: float) -> tuple[bytes, dict]:
        if self.lib is None:
            self.load()
        assert self.lib is not None and self.table is not None
        ids, unmapped = zh.text_to_phoneme_ids(text, self.table)
        arr = (ctypes.c_int32 * len(ids))(*ids)
        pcm = ctypes.POINTER(ctypes.c_float)()
        n = ctypes.c_int()
        frames = ctypes.c_int()
        started = time.time()
        rc = self.lib.snt_synth(arr, len(ids), ctypes.c_float(speed), ctypes.byref(pcm),
                                ctypes.byref(n), ctypes.byref(frames))
        elapsed = time.time() - started
        if rc != 0:
            raise RuntimeError(f"合成失败：{self.lib.snt_error().decode('utf-8', 'replace')}")
        try:
            samples = bytes(bytearray(n.value * 4))
            ctypes.memmove(samples, pcm, n.value * 4)
        finally:
            self.lib.snt_free(pcm)
        floats = struct.unpack(f"<{n.value}f", samples)
        clipped = [(-1.0 if v < -1.0 else (1.0 if v > 1.0 else v)) for v in floats]
        pcm16 = struct.pack(f"<{n.value}h", *(int(round(v * 32767.0)) for v in clipped))
        wav = _wav_bytes(pcm16, self.sample_rate)
        duration_ms = int(round(n.value * 1000.0 / self.sample_rate))
        self.stats["synth_calls"] += 1
        self.stats["last_ms"] = round(elapsed * 1000.0, 1)
        self.stats["last_frames"] = int(frames.value)
        meta = {
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "format": "wav/pcm_s16le/mono",
            "bytes": len(wav),
            "samples": n.value,
            "frames": int(frames.value),
            "phonemes": len(ids),
            "unmapped": unmapped,
            "duration_ms": duration_ms,
            "synth_ms": round(elapsed * 1000.0, 1),
            "rtf": round(elapsed / max(duration_ms / 1000.0, 1e-6), 3),
            "sha256": hashlib.sha256(wav).hexdigest(),
        }
        return wav, meta


def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


class Worker:
    def __init__(self) -> None:
        self.engine = Engine()
        self.config: dict = {}
        self.namespace = ""
        self.utterances: dict[str, dict] = {}   # id → {"wav": bytes, "meta": dict}
        self.order: list[str] = []

    # ---- 控制面 ----

    def prepare(self, req: dict) -> dict:
        self.namespace = str(req.get("namespace") or "")
        cfg = req.get("config") or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        self.config = dict(cfg) if isinstance(cfg, dict) else {}
        try:
            self.engine.load()
        except Exception as exc:  # 装载失败不抛给宿主：health 要说清原因
            self.engine.load_error = str(exc)
            _log(f"prepare 失败：{exc}")
        return {"ok": not self.engine.load_error, "voice": self.engine.voice,
                "sample_rate": self.engine.sample_rate, "error": self.engine.load_error}

    def health(self, _req: dict) -> dict:
        ok = self.engine.lib is not None and not self.engine.load_error
        return {"ok": ok, "message": "" if ok else self.engine.load_error,
                "engine": "sanotts", "voice": self.engine.voice,
                "sample_rate": self.engine.sample_rate}

    def drain(self, _req: dict) -> dict:
        self.utterances.clear()
        self.order.clear()
        return {"ok": True}

    # ---- 工具 ----

    def tool_voices(self, _req: dict) -> dict:
        return {
            "voices": [{
                "id": "zh",
                "name": "标准普通话（小雅）",
                "language": "zh_CN",
                "engine": "sanoTTS piperlite",
                "parameters": 1554284,
                "sample_rate": 22050,
                "weights_sha256": _sha256_file(self.engine.model_dir / "weights.fp16.bin"),
                "license": "权重 GPL-3.0（上游 ampixa/sanoTTS），推理代码 MIT",
            }],
            "bundled": True,
        }

    def tool_status(self, _req: dict) -> dict:
        return {
            "plugin": "hx.sanotts",
            "version": VERSION,
            "engine_loaded": self.engine.lib is not None,
            "load_error": self.engine.load_error,
            "voice": self.engine.voice,
            "sample_rate": self.engine.sample_rate,
            "config": self.config,
            "cache": {"utterances": len(self.utterances), "keep": KEEP_UTTERANCES},
            "stats": dict(self.engine.stats),
            "lexicon": zh.lexicon_stats(),
            # 生效的字数上限 = min(配置, 模型能力)：两者都要报出来，
            # 否则用户改了配置却看不到"其实被模型压着"。
            "max_chars_effective": self._char_cap(),
            "max_chars_config": self.config.get("max_chars"),
            "max_chars_engine": ENGINE_CHAR_CAP,
        }

    def _char_cap(self) -> int:
        """生效的单次播报字数上限。模型能力是硬上限，配置只能在它下面收窄。"""
        try:
            configured = int(self.config.get("max_chars") or ENGINE_CHAR_CAP)
        except (TypeError, ValueError):
            configured = ENGINE_CHAR_CAP
        if configured <= 0:
            configured = ENGINE_CHAR_CAP
        return min(configured, ENGINE_CHAR_CAP)

    def tool_speak(self, req: dict) -> dict:
        text = str(req.get("text") or "").strip()
        if not text:
            raise ToolError("PLUGIN_INVALID_ARGS", "text 不能为空")
        max_chars = self._char_cap()
        if len(text) > max_chars:
            # 文案里要同时给出**这一次**的上限与"为什么是这个数"：用户看到
            # 「上限 108 字」时该知道这不是哪个人随手定的，而是模型一次能吃的
            # 音素预算（432 个 id ÷ 每字 4 个）。
            hint = ""
            if max_chars < int(self.config.get("max_chars") or max_chars):
                hint = f"（配置里写的是 {self.config.get('max_chars')} 字，但模型单次最多约 {ENGINE_CHAR_CAP} 字，以小的为准）"
            raise ToolError(
                "PLUGIN_INVALID_ARGS",
                f"文本 {len(text)} 字超过上限 {max_chars} 字{hint}：播报请按句拆分后再调用",
            )
        speed = float(req.get("speed") or self.config.get("speed") or 1.0)
        speed = max(0.5, min(2.0, speed))
        wav, meta = self.engine.synthesize(text, speed)
        uid = hashlib.sha256((text + "|" + str(speed)).encode()).hexdigest()[:16]
        self.utterances[uid] = {"wav": wav, "meta": meta}
        if uid in self.order:
            self.order.remove(uid)
        self.order.append(uid)
        while len(self.order) > KEEP_UTTERANCES:
            self.utterances.pop(self.order.pop(0), None)
        chunk_count = max(1, (len(wav) + CHUNK_BYTES - 1) // CHUNK_BYTES)
        out = dict(meta)
        out.update({
            "ok": True,
            "utterance_id": uid,
            "chunk_bytes": CHUNK_BYTES,
            "chunk_count": chunk_count,
            "text_chars": len(text),
            "speed": speed,
            "audio_b64": base64.b64encode(wav[:CHUNK_BYTES]).decode(),
        })
        return out

    def tool_audio_chunk(self, req: dict) -> dict:
        uid = str(req.get("utterance_id") or "")
        entry = self.utterances.get(uid)
        if entry is None:
            raise ToolError("PLUGIN_NOT_FOUND",
                            f"没有这个播报片段（{uid or '空'}）：缓存只保留最近 {KEEP_UTTERANCES} 条")
        try:
            index = int(req.get("index") or 0)
        except (TypeError, ValueError):
            raise ToolError("PLUGIN_INVALID_ARGS", "index 必须是整数")
        wav = entry["wav"]
        total = max(1, (len(wav) + CHUNK_BYTES - 1) // CHUNK_BYTES)
        if index < 0 or index >= total:
            raise ToolError("PLUGIN_INVALID_ARGS", f"index 越界：0..{total - 1}")
        part = wav[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
        return {
            "utterance_id": uid,
            "index": index,
            "chunk_count": total,
            "bytes": len(part),
            "eof": index == total - 1,
            "audio_b64": base64.b64encode(part).decode(),
        }


class ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


TOOLS = {
    "voices": "tool_voices",
    "status": "tool_status",
    "speak": "tool_speak",
    "audio_chunk": "tool_audio_chunk",
}


def dispatch(worker: Worker, req: dict) -> dict:
    op = str(req.get("op") or "")
    if op == "prepare":
        return worker.prepare(req)
    if op == "health":
        return worker.health(req)
    if op == "drain":
        return worker.drain(req)
    if op == "snapshot":
        return {"ok": True}
    handler_name = TOOLS.get(op)
    if handler_name is None:
        raise ToolError("PLUGIN_TOOL_UNSUPPORTED", f"本插件没有工具 {op!r}")
    return getattr(worker, handler_name)(req)


def main() -> int:
    worker = Worker()
    _log(f"worker {VERSION} 启动（pid={os.getpid()}）")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": {
                "code": "PLUGIN_INVALID_ARGS", "message": f"请求不是合法 JSON：{exc}"}}), flush=True)
            continue
        if not isinstance(req, dict):
            print(json.dumps({"ok": False, "error": {
                "code": "PLUGIN_INVALID_ARGS", "message": "请求必须是 JSON 对象"}}), flush=True)
            continue
        try:
            result = dispatch(worker, req)
            print(json.dumps({"ok": True, "id": req.get("id"), "result": result},
                             ensure_ascii=False), flush=True)
        except ToolError as exc:
            print(json.dumps({"ok": False, "id": req.get("id"), "error": {
                "code": exc.code, "message": exc.message}}, ensure_ascii=False), flush=True)
        except Exception as exc:  # 任何未预期异常都要变成协议应答，否则宿主只能看到 EOF
            _log(f"未预期异常：{exc!r}")
            print(json.dumps({"ok": False, "id": req.get("id"), "error": {
                "code": "PLUGIN_CALL_FAILED", "message": str(exc)}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
