# HX-SANOTTS — 本地播报（sanoTTS）插件

HOMEX 平台的第三方插件：**把一句话在本机合成成语音，完全离线**。
引擎是 [sanoTTS](https://github.com/Ampixa/sanoTTS) 的中文语音包
`zh_CN-xiao_ya-medium`（1.55M 参数，piperlite 图，22050 Hz），
推理用本仓自带的 C 内核，**不依赖 numpy、不联网、不需要 espeak**。

| | |
|---|---|
| 插件 ID | `hx.sanotts` |
| 版本 | 1.0.0 |
| 运行时 | `python`（宿主 stdio 子进程，无独立端口） |
| 工具 | `speak` / `audio_chunk` / `voices` / `status` |
| 配置页 | 自带（`entry.ui`，iframe sandbox，可试听） |
| 依赖 | **零**（无 requirements，无第三方包） |

---

## 一、为什么不是"直接 pip sanotts"

上游 Python 包能跑，但在本平台上有三个硬约束：

| 约束 | 数字 | 后果 |
|---|---|---|
| `.hpkg` 单包上限 | 32 MiB | numpy wheel 就 16.5 MB |
| 包内文件数上限 | 256 个 | numpy 解包是 951 个文件 |
| 播报要"说完就响" | — | numpy 参考实现实测 **RTF ≈ 3**（4 秒的话算 12 秒） |

所以推理被移植成 C（`runtime/sanotts.c`，46 KB 的 .so），并且**逐样本对齐**了上游
numpy 实现：同一份权重、同一个音素 id 序列，输出相关系数 **1.000000**
（见「四、验证」）。

文本前端仍是上游的 `zh_g2p`（`runtime/sanotts_zh.py`，逐字移植），因为它包含
pypinyin 词表、多音字、`一/不` 变调、数字读法（CLDR zh 规则）与 piper 的
pinyin framing —— 重写只会引入无法归因的偏差。

---

## 二、目录

```
manifest.json                插件清单（runtime=python，entry.ui=ui/index.html）
requirements.lock            空锁（本插件没有任何第三方依赖，见文件内说明）
runtime/
  worker.py                  Worker 主进程（stdin/stdout JSON lines）
  sanotts_zh.py              中文文本 → 音素 id（上游 zh_g2p 移植）
  sanotts.c                  sanoTTS 推理内核（C99）
  libsanotts.so              预编译内核（linux/amd64，gcc -O3）
  models/zh/
    weights.fp16.bin         语音权重（3,108,568 B，上游原样）
    manifest.json            权重清单（张量表/形状/偏移，上游原样）
    layout.bin               张量表压缩版（C 侧零依赖解析，见 tools/gen_layout.py）
    piper-phoneme-config.json 音素表（85 符号/73 id，pinyin framing）
  g2p_tables/
    chars.xz                 41,923 条字表
    phrases.xz               8,756 条词组读音
    phrase_keys.xz           38,355 条词组键
ui/                          自带配置页（index.html / app.js / style.css）
tools/gen_layout.py          构建期：manifest.json → layout.bin
testdata/golden/             黄金数据（4 句中文的 id 序列 + numpy 参考 WAV）
```

---

## 三、接口

### 3.1 `speak` — 文本 → WAV（分块）

```jsonc
// 请求
{"text": "欢迎回家。", "speed": 1.0}
// 应答（第一块 + 元数据）
{
  "ok": true,
  "utterance_id": "34add6e0b3c19f27",
  "sample_rate": 22050, "format": "wav/pcm_s16le/mono",
  "bytes": 81964, "samples": 40960, "frames": 160,
  "phonemes": 24, "unmapped": "",
  "duration_ms": 1858, "synth_ms": 2226.3, "rtf": 1.198,
  "sha256": "2e1992…",
  "chunk_bytes": 49152, "chunk_count": 2,
  "audio_b64": "UklGR…"       // 第一块
}
```

**为什么分块**：宿主对单次应答有量级限制，而一句中文约 44 KB/秒（44.1 kHz 之前
的 22.05 kHz、16-bit 单声道）。十秒就是 440 KB —— 一次塞回去会让"能听"和
"传得出去"互相矛盾。取回方式：

```jsonc
{"tool": "audio_chunk", "args": {"utterance_id": "34add6e0b3c19f27", "index": 1}}
→ {"utterance_id": "…", "index": 1, "chunk_count": 2, "bytes": 32812, "eof": true, "audio_b64": "…"}
```

缓存只留**最近 4 条**（`KEEP_UTTERANCES`）：播报是"刚刚那句话"，历史没有必要
占内存。取不到时明确报 `PLUGIN_NOT_FOUND` 而不是给一段静音。

### 3.2 `voices` / `status`

`voices` 给语音包清单与**权重 SHA-256**（可核对包内字节）；`status` 给引擎就绪、
词表规模、上一次合成耗时与配置快照。两者都只读。

### 3.3 配置项

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `voice` | enum(`zh`) | `zh` | 随包只有一条中文语音包 |
| `speed` | number 0.5–2.0 | `1.0` | 语速倍率（上游 duration length_scale） |
| `max_chars` | int 1–400 | `200` | 单次播报字数上限；超限明文报错并提示拆分 |

---

## 四、验证

### 4.1 与上游 numpy 实现的一致性（这是本插件最重要的一条）

`testdata/golden/golden.json` 存 4 句中文的音素 id 序列与 numpy 参考实现的 WAV。
C 内核在同一份权重上重算，按样本比对：

```
0 打开客厅灯。                    corr 1.000000  frames 160
1 你好，欢迎回家。今天天气不错。    corr 1.000000  frames 320
2 客厅的灯已经打开了，室内温度二十六度。 corr 1.000000  frames 385
3 现在是晚上十点，记得关窗。        corr 1.000000  frames 288
```

相关系数到小数点后六位为 1.000000，说明**算子顺序、framing、时长整数化
（`exp → clamp(1) → ×scale → round-half-even → clamp(max)`）全部一致** ——
任何一处差半帧都会让这条数字掉下来。

### 4.2 性能（本机 i7-8550U，单线程，-O3）

| 句子 | 音频时长 | 合成耗时 | RTF |
|---|---|---|---|
| 打开客厅灯。 | 1.86 s | 1.8 s | 0.97 |
| 你好，欢迎回家。今天天气不错。 | 3.72 s | 3.7 s | 1.01 |

RTF ≈ 1 意味着"说完就响"尚差一点余量：这台机器上负载 9.7（8 线程），
空载会明显更快。上游用同一套权重的 numpy 实现是 RTF ≈ 3。

---

## 五、构建

```bash
# 1. C 内核（产物随包分发，平台侧不编译）
gcc -O3 -fPIC -shared -o runtime/libsanotts.so runtime/sanotts.c -lm

# 2. 张量表（语音包换了才需要重跑）
python3 tools/gen_layout.py runtime/models/zh runtime/models/zh/layout.bin

# 3. 打包 + 去签名（第三方插件走免签名通道）
cd <homex>
go run ./cmd/cli plugin package --dir <HX-SANOTTS> --out build/hx.sanotts-1.0.0.hpkg
go run ./cmd/cli plugin unsign build/hx.sanotts-1.0.0.hpkg
go run ./cmd/cli plugin verify build/hx.sanotts-1.0.0.hpkg   # 提示"未签名"是预期
```

> `plugin unsign` 只清空 `signature.value`，**保留** `algorithm`/`key_id` 骨架。
> 删掉整个 signature 段会让 manifest 校验失败。

---

## 六、许可证

- 本仓代码（`runtime/sanotts.c`、`runtime/worker.py`、`ui/`、`tools/`）：MIT
- `runtime/sanotts_zh.py`：MIT（上游 `pypkg/sanotts` 部分）
- `runtime/models/zh/`（语音权重）：**GPL-3.0**（上游 ampixa/sanoTTS）
- 词表数据：MIT（pypinyin 的 pinyin-data / phrase-pinyin-data）

整包按 GPL-3.0 分发（因为随包带了 GPL-3.0 权重）。插件不上传任何文本、
不发起任何网络请求，`permissions` 只有 `config.read`。