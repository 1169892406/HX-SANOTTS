# 构建说明

本插件是 **python 运行时 + 预编译 C 推理库**。宿主侧不编译、不装依赖，构建只发生在作者机器上。

## 1. C 推理内核

```bash
gcc -O3 -fPIC -shared -o runtime/libsanotts.so runtime/sanotts.c -lm
```

- 只需要 gcc 与 libm；**不依赖 numpy / BLAS / OpenMP**（`#ifdef _OPENMP` 的 pragma 只在
  编译器接受 `-fopenmp` 时生效；不传该参数时是一个普通单线程程序，链接结果不会多出 libgomp）。
- 产物按架构耦合：本仓的 `libsanotts.so` 是 **linux/amd64**（glibc）。别的架构需要在这台机器上重编，
  否则 Worker 的 `ctypes.CDLL` 会报「缺少推理库 / 不是有效的 ELF」。这是刻意的取舍：
  预编译换来的是"装完就能出声"，代价是包要按架构分发。

## 2. 语音包与 layout

```bash
# 上游语音包（HF 镜像；GitHub release 里没有中文包）
curl -L -o runtime/models/zh/manifest.json      https://hf-mirror.com/ampixa/sanoTTS/resolve/main/zh-xiaoya-1p55m/manifest.json
curl -L -o runtime/models/zh/piper-phoneme-config.json https://hf-mirror.com/ampixa/sanoTTS/resolve/main/zh-xiaoya-1p55m/piper-phoneme-config.json
curl -L -o runtime/models/zh/weights.fp16.bin   https://hf-mirror.com/ampixa/sanoTTS/resolve/main/zh-xiaoya-1p55m/weights.fp16.bin

# 张量表（manifest.json → C 侧可零依赖解析的 layout.bin）
python3 tools/gen_layout.py runtime/models/zh runtime/models/zh/layout.bin
```

权重摘要必须与下面这条一致（`voices` 工具会把它回报给调用方，可核对）：

```
ec4faabed7e443a3eab3517f8d1d353624babaadd8357213e83def82507f7497  weights.fp16.bin
```

## 3. 打包

```bash
cd <homex>
go run ./cmd/cli plugin package --dir <HX-SANOTTS> --out <HX-SANOTTS>/build/hx.sanotts-1.0.0.hpkg
go run ./cmd/cli plugin unsign <HX-SANOTTS>/build/hx.sanotts-1.0.0.hpkg
go run ./cmd/cli plugin verify <HX-SANOTTS>/build/hx.sanotts-1.0.0.hpkg   # 「未签名」是预期
```

`unsign` 只清空 `signature.value`，**保留** `algorithm` / `key_id` 骨架 —— 删掉整段会让
manifest 校验失败（`缺少 signature.algorithm`）。

## 4. 验证（改任何一行推理代码都要跑）

```bash
# ① 与 numpy 参考逐样本对齐（corr 必须 ≥0.99，实际是 1.000000）
PYTHONPATH=<homex>/internal/../..  python3 tools/check_golden.py

# ② 平台侧的端到端：物化 → Worker 起 → speak → audio_chunk → 拼回 WAV
cd <homex>
go test ./internal/app/pluginsvc/ -run 'TestSanottsPluginEndToEnd' -count=1 -v

# ③ 商店链路：catalog_list 能列出、catalog_fetch 取回的字节等于仓内制品
(cd <HXCS> && python3 -m http.server 18080 --bind 127.0.0.1 &)
HXCS_REGISTRY_URL=http://127.0.0.1:18080 go test ./internal/app/pluginsvc/ -run 'TestHXCSCatalogServesSanotts' -count=1 -v
```
