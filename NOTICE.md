# 第三方来源与许可证

本插件把两个上游作品接进 HOMEX。**代码与语音权重的许可证不同**，逐项列清：

| 路径 | 内容 | 来源 | 许可证 |
|---|---|---|---|
| `runtime/sanotts.c` | 推理内核（本仓移植） | 由 `pypkg/sanotts/models.py` 逐条移植 | MIT |
| `runtime/sanotts_zh.py` | 中文 pinyin 前端（逐字移植） | `pypkg/sanotts/zh_g2p.py` | MIT |
| `runtime/g2p_tables/*.xz` | 字表 41,923 条 / 词组 8,756 条 | pypinyin 的 `pinyin-data`、`phrase-pinyin-data` | MIT |
| `runtime/models/zh/weights.fp16.bin` | 语音权重 3,108,568 B | `huggingface.co/ampixa/sanoTTS` → `zh-xiaoya-1p55m` | **GPL-3.0** |
| `runtime/models/zh/manifest.json` | 权重清单（张量表） | 同上 | **GPL-3.0** |
| `runtime/models/zh/piper-phoneme-config.json` | 音素表（85 符号 / 73 id） | 同上（源自 Piper 的 zh_CN xiao_ya） | **GPL-3.0** |
| `ui/`、`runtime/worker.py`、`tools/` | 插件外壳 | 本仓原创 | MIT |

上游仓库：<https://github.com/Ampixa/sanoTTS>（仓库整体 GPL-3.0；其 `LICENSE.MIT`
明确列出推理运行时部分为 MIT，grapheme-to-phoneme 层因嵌入 espeak-ng 而整体 GPL —— 本插件
不使用 espeak 路径，中文走的是 MIT 的 pinyin 前端）。

因为随包分发了 GPL-3.0 的语音权重与清单，**整个 `.hpkg` 按 GPL-3.0 分发**；
本仓代码本身仍可按 MIT 使用。插件不上传任何文本、不发起任何网络请求。
