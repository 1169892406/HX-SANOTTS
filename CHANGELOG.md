# CHANGELOG

## 1.0.0

首个版本。把 sanoTTS 的中文语音包接进 HOMEX。

- 推理：`runtime/sanotts.c`（C99，46 KB .so，无第三方依赖），逐条对照上游
  `pypkg/sanotts/models.py`，与 numpy 参考实现**逐样本相关 1.000000**。
- 文本前端：`runtime/sanotts_zh.py`（上游 `zh_g2p.py` 逐字移植）：pypinyin 词表、
  多音字、`一/不` 变调、CLDR zh 数字读法、piper pinyin framing。
- 工具：`speak`（文本→WAV，48 KiB 分块）、`audio_chunk`、`voices`、`status`。
- 配置页：自带 iframe 页，WebAudio 直接播放（CSP 下 `<audio>` 的 data:/blob:
  会被拦，AudioBuffer 不经过任何 URL）。
- 语音包：随包分发 `zh_CN-xiao_ya-medium`（1.55M 参数，GPL-3.0）。
- 字数上限对齐模型能力：432 个音素 id ÷ 每字 4 个 = **108 字**，配置的 `max_chars`
  只能在它以下收窄；`status` 同时报 `max_chars_effective` 与 `max_chars_engine`。
  修掉此前"300 字通过字数校验、在引擎里报音素超限"的内部分裂。
- 配置页上报内容高度（`plugin.ui.resize`），滚动交回平台页，不再双滚动条。
