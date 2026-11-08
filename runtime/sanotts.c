/* sanotts.c — HOMEX 本地播报插件的 sanoTTS 推理内核（纯 C99，无第三方依赖）。
 *
 * SPDX-License-Identifier: MIT
 *
 * 来源与边界
 * ----------
 * 本文件是 sanoTTS（github.com/Ampixa/sanoTTS）Python 参考实现的 C 移植：
 *   · 图结构与算子语义逐条照 pypkg/sanotts/models.py（该文件 MIT）；
 *   · 权重格式 = 上游语音包原样（weights.fp16.bin + layout.bin，见 tools/gen_layout.py）；
 *   · 文本→音素在上游 piper-phoneme-config.json 的 pinyin 词表，本文件只吃音素 id 序列。
 *
 * 上游 mcu/ 那套 C 运行时用的是它自家 exporter 的 int8 blob 与 meta.bin，与本文件
 * 不通用 —— 这里刻意按「Python 参考实现的 fp32 语义」移植，验证方式最硬：
 * 同一份权重、同一个 id 序列，输出必须与 numpy 参考逐样本相关 ≥0.99。
 *
 * 为什么不用 numpy
 * ----------------
 * .hpkg 单包 32 MiB、单文件 16 MiB；numpy 的 wheel 就 16.5 MB、解包 33 MB、
 * 951 个文件。速度上更明显：numpy 参考实现 RTF ≈ 3（4 秒的话要算 12 秒），
 * 而这份朴素 C（-O3 单线程）同一段话只要几十毫秒。
 */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SNT_MAX_TENSORS 160
#define SNT_MAX_NAME 96
#define SNT_SCHWA_FALLBACK 59 /* 与 models.py 的 _SCHWA_FALLBACK_ID 一致 */

typedef struct {
    char name[SNT_MAX_NAME];
    int ndim;
    int dim[3];
    int n;
    const uint8_t *raw;
    float *f32;
} snt_tensor;

static snt_tensor g_t[SNT_MAX_TENSORS];
static int g_nt;
static uint8_t *g_weights;
static size_t g_weights_len;
static char g_err[512];
static int g_sample_rate = 22050;
static int g_hop = 256;
static int g_max_tokens = 432;
static int g_max_duration = 80;
static int g_pf_channels, g_pf_layers, g_pf_kernel;
static float g_pf_scale;

static void snt_errf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

const char *snt_error(void) { return g_err; }
int snt_sample_rate(void) { return g_sample_rate; }
int snt_hop_length(void) { return g_hop; }

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            mant &= 0x3ffu;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int load_weights(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snt_errf("打不开权重文件：%s", path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        snt_errf("权重文件不可 seek：%s", path);
        return -1;
    }
    long size = ftell(f);
    if (size <= 0) {
        fclose(f);
        snt_errf("权重文件为空：%s", path);
        return -1;
    }
    rewind(f);
    g_weights = (uint8_t *)malloc((size_t)size);
    if (!g_weights) {
        fclose(f);
        snt_errf("权重内存分配失败（%ld 字节）", size);
        return -1;
    }
    if (fread(g_weights, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        snt_errf("权重文件读取不完整：%s", path);
        return -1;
    }
    fclose(f);
    g_weights_len = (size_t)size;
    return 0;
}

static int load_layout(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snt_errf("打不开 layout：%s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    if (size <= 0) {
        fclose(f);
        snt_errf("layout 为空：%s", path);
        return -1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    if (!buf) {
        fclose(f);
        snt_errf("layout 内存分配失败");
        return -1;
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(buf);
        snt_errf("layout 读取不完整");
        return -1;
    }
    fclose(f);

    int rc = -1;
    size_t off = 0;
    if (size < 12 || memcmp(buf, "SNTL", 4) != 0) {
        snt_errf("layout magic 不符（期望 SNTL）");
        goto done;
    }
    off = 4;
    uint32_t version = rd_u32(buf + off);
    off += 4;
    if (version != 1) {
        snt_errf("layout 版本不支持：%u", version);
        goto done;
    }
    uint32_t nt = rd_u32(buf + off);
    off += 4;
    if (nt > SNT_MAX_TENSORS) {
        snt_errf("张量数超限：%u", nt);
        goto done;
    }
    g_nt = (int)nt;
    for (uint32_t i = 0; i < nt; i++) {
        if (off >= (size_t)size) {
            snt_errf("layout 被截断（表项 %u）", i);
            goto done;
        }
        uint8_t nlen = buf[off++];
        if (nlen == 0 || nlen >= SNT_MAX_NAME || off + nlen > (size_t)size) {
            snt_errf("layout 张量名非法（表项 %u）", i);
            goto done;
        }
        memcpy(g_t[i].name, buf + off, nlen);
        g_t[i].name[nlen] = 0;
        off += nlen;
        if (off + 1 + 12 + 8 > (size_t)size) {
            snt_errf("layout 被截断（表项 %u 头部）", i);
            goto done;
        }
        uint8_t ndim = buf[off++];
        g_t[i].ndim = ndim;
        for (int d = 0; d < 3; d++) {
            g_t[i].dim[d] = (int)rd_u32(buf + off);
            off += 4;
        }
        uint32_t toff = rd_u32(buf + off);
        off += 4;
        uint32_t nbytes = rd_u32(buf + off);
        off += 4;
        int n = 1;
        for (int d = 0; d < ndim; d++) n *= g_t[i].dim[d];
        g_t[i].n = n;
        if ((size_t)toff + nbytes > g_weights_len) {
            snt_errf("张量 %s 越界（offset %u + %u > %zu）", g_t[i].name, toff, nbytes,
                     g_weights_len);
            goto done;
        }
        if ((size_t)n * 2 != nbytes) {
            snt_errf("张量 %s 字节数不符：n=%d nbytes=%u（只支持 fp16）", g_t[i].name, n, nbytes);
            goto done;
        }
        g_t[i].raw = g_weights + toff;
        g_t[i].f32 = NULL;
    }
    if (off + 4 > (size_t)size) {
        snt_errf("layout 缺少 kv 段");
        goto done;
    }
    uint32_t nkv = rd_u32(buf + off);
    off += 4;
    for (uint32_t i = 0; i < nkv; i++) {
        if (off >= (size_t)size) {
            snt_errf("layout kv 被截断");
            goto done;
        }
        uint8_t klen = buf[off++];
        if (klen >= SNT_MAX_NAME || off + klen + 4 > (size_t)size) {
            snt_errf("layout kv 项非法");
            goto done;
        }
        char key[SNT_MAX_NAME];
        memcpy(key, buf + off, klen);
        key[klen] = 0;
        off += klen;
        float v;
        memcpy(&v, buf + off, 4);
        off += 4;
        if (!strcmp(key, "sample_rate")) g_sample_rate = (int)v;
        else if (!strcmp(key, "hop_length")) g_hop = (int)v;
        else if (!strcmp(key, "max_tokens")) g_max_tokens = (int)v;
        else if (!strcmp(key, "max_duration")) g_max_duration = (int)v;
        else if (!strcmp(key, "post_filter_channels")) g_pf_channels = (int)v;
        else if (!strcmp(key, "post_filter_layers")) g_pf_layers = (int)v;
        else if (!strcmp(key, "post_filter_kernel")) g_pf_kernel = (int)v;
        else if (!strcmp(key, "post_filter_scale")) g_pf_scale = v;
    }
    rc = 0;
done:
    free(buf);
    return rc;
}

static snt_tensor *tget(const char *name)
{
    for (int i = 0; i < g_nt; i++)
        if (!strcmp(g_t[i].name, name)) return &g_t[i];
    snt_errf("缺少张量：%s", name);
    return NULL;
}

static const float *tf32(const char *name);

static const float *bias_of(const char *weight_name)
{
    char buf[SNT_MAX_NAME];
    snprintf(buf, sizeof(buf), "%s", weight_name);
    size_t len = strlen(buf);
    const char *suffix = ".weight";
    if (len > strlen(suffix) && !strcmp(buf + len - strlen(suffix), suffix)) {
        buf[len - strlen(suffix)] = 0;
    }
    strncat(buf, ".bias", sizeof(buf) - strlen(buf) - 1);
    return tf32(buf);
}

static const float *tf32(const char *name)
{
    snt_tensor *t = tget(name);
    if (!t) return NULL;
    if (!t->f32) {
        t->f32 = (float *)malloc((size_t)t->n * sizeof(float));
        if (!t->f32) {
            snt_errf("张量 %s 展开失败", name);
            return NULL;
        }
        for (int i = 0; i < t->n; i++) {
            uint16_t h = (uint16_t)(t->raw[2 * i] | ((uint16_t)t->raw[2 * i + 1] << 8));
            t->f32[i] = half_to_float(h);
        }
    }
    return t->f32;
}

/* ---------------- 基础算子 ----------------
 * 激活一律 channel-major：x[ch][T]，行内连续，与 models.py 的 [ch, T] 一致。 */

static void conv1d_same(const float *restrict x, int in_ch, int T, const float *restrict w,
                        const float *restrict b, int out_ch, int K, int dilation,
                        float *restrict out)
{
    int pad = dilation * (K / 2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < out_ch; oc++)
        for (int t = 0; t < T; t++) out[(size_t)oc * T + t] = b[oc];
    for (int k = 0; k < K; k++) {
        int shift = k * dilation - pad;
        int t0 = shift < 0 ? -shift : 0;
        int t1 = T - shift;
        if (t1 > T) t1 = T;
        if (t1 <= t0) continue;
        int n = t1 - t0;
        for (int oc = 0; oc < out_ch; oc++) {
            float *o = out + (size_t)oc * T + t0;
            for (int ic = 0; ic < in_ch; ic++) {
                float wv = w[((size_t)oc * in_ch + ic) * K + k];
                if (wv == 0.0f) continue;
                const float *src = x + (size_t)ic * T + t0 + shift;
                for (int t = 0; t < n; t++) o[t] += wv * src[t];
            }
        }
    }
}

static void conv1d_1x1(const float *restrict x, int in_ch, int T, const float *restrict w,
                       const float *restrict b, int out_ch, float *restrict out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < out_ch; oc++) {
        float *o = out + (size_t)oc * T;
        for (int t = 0; t < T; t++) o[t] = b[oc];
        for (int ic = 0; ic < in_ch; ic++) {
            float wv = w[(size_t)oc * in_ch + ic];
            const float *src = x + (size_t)ic * T;
            for (int t = 0; t < T; t++) o[t] += wv * src[t];
        }
    }
}

static void conv_transpose1d(const float *restrict x, int in_ch, int T, const float *restrict w,
                             const float *restrict b, int out_ch, int K, int stride,
                             int padding, int L, float *restrict out)
{
    for (int oc = 0; oc < out_ch; oc++)
        for (int j = 0; j < L; j++) out[(size_t)oc * L + j] = b[oc];
    for (int k = 0; k < K; k++) {
        int shift = k - padding;
        int t0 = shift < 0 ? (-shift + stride - 1) / stride : 0;
        int t1 = (L - 1 - shift) / stride + 1;
        if (t1 > T) t1 = T;
        if (t1 <= t0) continue;
        int j0 = t0 * stride + shift;
        for (int oc = 0; oc < out_ch; oc++) {
            float *o = out + (size_t)oc * L;
            for (int ic = 0; ic < in_ch; ic++) {
                float wv = w[((size_t)ic * out_ch + oc) * K + k];
                if (wv == 0.0f) continue;
                const float *src = x + (size_t)ic * T;
                int j = j0;
                for (int t = t0; t < t1; t++, j += stride) o[j] += wv * src[t];
            }
        }
    }
}

static void leaky_to(const float *src, float *dst, int n, float slope)
{
    for (int i = 0; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : slope * src[i];
}

static void silu_inplace(float *p, int n)
{
    for (int i = 0; i < n; i++) p[i] = p[i] / (1.0f + expf(-p[i]));
}

/* ResidualConvBlock: x + scale * conv2(silu(conv1(x))) */
static int residual_block(const char *prefix, float *x, int ch, int T, float *scratch)
{
    char name[SNT_MAX_NAME], name2[SNT_MAX_NAME];
    snprintf(name, sizeof(name), "%s.net.0.weight", prefix);
    snprintf(name2, sizeof(name2), "%s.net.2.weight", prefix);
    snt_tensor *w0 = tget(name);
    if (!w0) return -1;
    snt_tensor *w2 = tget(name2);
    if (!w2) return -1;
    char sc[SNT_MAX_NAME];
    snprintf(sc, sizeof(sc), "%s.scale", prefix);
    const float *scale = tf32(sc);
    const float *b0 = bias_of(name);
    const float *b2 = bias_of(name2);
    if (!scale || !b0 || !b2) return -1;

    conv1d_same(x, ch, T, tf32(name), b0, ch, w0->dim[2], 1, scratch);
    silu_inplace(scratch, ch * T);
    conv1d_same(scratch, ch, T, tf32(name2), b2, ch, w2->dim[2], 1, scratch + (size_t)ch * T);
    const float *u = scratch + (size_t)ch * T;
    float s = scale[0];
    for (int i = 0; i < ch * T; i++) x[i] += s * u[i];
    return 0;
}

static void linspace01(int n, float *out)
{
    if (n <= 0) return;
    if (n == 1) {
        out[0] = 0.0f;
        return;
    }
    for (int i = 0; i < n; i++) out[i] = (float)((double)i / (double)(n - 1));
}

static int count_blocks(const char *prefix)
{
    size_t plen = strlen(prefix);
    int depth = 0;
    for (int i = 0; i < g_nt; i++) {
        const char *rest = g_t[i].name;
        if (strncmp(rest, prefix, plen) != 0) continue;
        rest += plen;
        if (rest[0] != '.' || rest[1] < '0' || rest[1] > '9') continue;
        if (!strstr(rest, ".net.0.weight")) continue;
        int idx = atoi(rest + 1);
        if (idx + 1 > depth) depth = idx + 1;
    }
    return depth;
}

static void clamp_ids(const int32_t *ids, int n, int vocab, int32_t *out)
{
    int fallback = SNT_SCHWA_FALLBACK < vocab ? SNT_SCHWA_FALLBACK : 0;
    for (int i = 0; i < n; i++) {
        int32_t v = ids[i];
        out[i] = (v < 0 || v >= vocab) ? fallback : v;
    }
}

/* ---------------- duration student ---------------- */

static int duration_forward(const int32_t *ids_in, int n, float length_scale, int32_t *dur_out)
{
    snt_tensor *emb = tget("duration/embedding.weight");
    snt_tensor *ip = tget("duration/input_proj.weight");
    snt_tensor *ow = tget("duration/output.weight");
    if (!emb || !ip || !ow) return -1;
    int vocab = emb->dim[0], hidden = emb->dim[1];
    const float *embed = tf32(emb->name);
    const float *ip_w = tf32(ip->name);
    const float *ip_b = bias_of("duration/input_proj.weight");
    const float *ow_w = tf32(ow->name);
    const float *ow_b = bias_of("duration/output.weight");
    if (!embed || !ip_w || !ip_b || !ow_w || !ow_b) return -1;

    int32_t *ids = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int in_ch = hidden + 3;
    float *x = (float *)malloc((size_t)in_ch * n * sizeof(float));
    float *proj = (float *)malloc((size_t)hidden * n * sizeof(float));
    float *scratch = (float *)malloc((size_t)2 * hidden * n * sizeof(float));
    float *logdur = (float *)malloc((size_t)n * sizeof(float));
    int rc = -1;
    if (!ids || !x || !proj || !scratch || !logdur) {
        snt_errf("duration 阶段内存不足（n=%d）", n);
        goto done;
    }
    clamp_ids(ids_in, n, vocab, ids);
    linspace01(n, x + (size_t)hidden * n);
    float length_hint = (float)(log1p((double)n) / log1p((double)g_max_tokens));
    for (int i = 0; i < n; i++) {
        x[(size_t)(hidden + 1) * n + i] = length_hint;
        x[(size_t)(hidden + 2) * n + i] = 1.0f;
    }
    for (int c = 0; c < hidden; c++) {
        float *dst = x + (size_t)c * n;
        for (int i = 0; i < n; i++) dst[i] = embed[(size_t)ids[i] * hidden + c];
    }
    conv1d_1x1(x, in_ch, n, ip_w, ip_b, hidden, proj);
    int depth = count_blocks("duration/blocks");
    for (int i = 0; i < depth; i++) {
        char prefix[SNT_MAX_NAME];
        snprintf(prefix, sizeof(prefix), "duration/blocks.%d", i);
        if (residual_block(prefix, proj, hidden, n, scratch) != 0) goto done;
    }
    conv1d_1x1(proj, hidden, n, ow_w, ow_b, 1, logdur);
    for (int i = 0; i < n; i++) {
        float d = expf(logdur[i]);
        if (d < 1.0f) d = 1.0f;
        /* np.round = 就近取偶；rintf 在默认舍入模式（FE_TONEAREST）下一致 */
        float scaled = rintf(d * length_scale);
        if (scaled < 1.0f) scaled = 1.0f;
        if (scaled > (float)g_max_duration) scaled = (float)g_max_duration;
        dur_out[i] = (int32_t)scaled;
    }
    rc = 0;
done:
    free(ids);
    free(x);
    free(proj);
    free(scratch);
    free(logdur);
    return rc;
}

/* ---------------- acoustic student (token_context) ---------------- */

static int acoustic_forward(const int32_t *ids_in, const int32_t *dur_in, int n, long frames,
                            float *latent_out)
{
    snt_tensor *emb = tget("acoustic/embedding.weight");
    snt_tensor *tip = tget("acoustic/token_input_proj.weight");
    snt_tensor *fip = tget("acoustic/frame_input_proj.weight");
    snt_tensor *ow = tget("acoustic/output.weight");
    if (!emb || !tip || !fip || !ow) return -1;
    int vocab = emb->dim[0], hidden = emb->dim[1], out_ch = ow->dim[0];
    const float *embed = tf32(emb->name);
    const float *tip_w = tf32(tip->name);
    const float *tip_b = bias_of("acoustic/token_input_proj.weight");
    const float *fip_w = tf32(fip->name);
    const float *fip_b = bias_of("acoustic/frame_input_proj.weight");
    const float *ow_w = tf32(ow->name);
    const float *ow_b = bias_of("acoustic/output.weight");
    if (!embed || !tip_w || !tip_b || !fip_w || !fip_b || !ow_w || !ow_b) return -1;

    int rc = -1;
    int32_t *ids = NULL, *dur = NULL;
    float *tx = NULL, *tok = NULL, *scratch = NULL, *expanded = NULL, *fx = NULL;
    float *lat = NULL, *scratch2 = NULL;

    ids = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    dur = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!ids || !dur) {
        snt_errf("acoustic: 内存不足（token 表）");
        goto done;
    }
    clamp_ids(ids_in, n, vocab, ids);
    int32_t max_dur = 1;
    for (int i = 0; i < n; i++) {
        if (dur_in[i] < 1) {
            snt_errf("acoustic: 非正时长 %d", (int)dur_in[i]);
            goto done;
        }
        dur[i] = dur_in[i];
        if (dur[i] > max_dur) max_dur = dur[i];
    }

    int tok_in = hidden + 2;
    int scratch_elems = (int)(2 * hidden * (long)(frames > n ? frames : n));
    tx = (float *)malloc((size_t)tok_in * n * sizeof(float));
    tok = (float *)malloc((size_t)hidden * n * sizeof(float));
    scratch = (float *)malloc((size_t)scratch_elems * sizeof(float));
    if (!tx || !tok || !scratch) {
        snt_errf("acoustic: 内存不足（token 阶段）");
        goto done;
    }
    linspace01(n, tx + (size_t)hidden * n);
    for (int i = 0; i < n; i++)
        tx[(size_t)(hidden + 1) * n + i] =
            (float)(log1p((double)dur[i]) / log1p((double)max_dur));
    for (int c = 0; c < hidden; c++) {
        float *dst = tx + (size_t)c * n;
        for (int i = 0; i < n; i++) dst[i] = embed[(size_t)ids[i] * hidden + c];
    }
    conv1d_1x1(tx, tok_in, n, tip_w, tip_b, hidden, tok);
    free(tx);
    tx = NULL;
    int token_depth = count_blocks("acoustic/token_blocks");
    for (int i = 0; i < token_depth; i++) {
        char prefix[SNT_MAX_NAME];
        snprintf(prefix, sizeof(prefix), "acoustic/token_blocks.%d", i);
        if (residual_block(prefix, tok, hidden, n, scratch) != 0) goto done;
    }

    expanded = (float *)malloc((size_t)hidden * (size_t)frames * sizeof(float));
    if (!expanded) {
        snt_errf("acoustic: 内存不足（展开 %ld 帧）", frames);
        goto done;
    }
    long pos = 0;
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dur[i]; d++, pos++) {
            for (int c = 0; c < hidden; c++)
                expanded[(size_t)c * frames + pos] = tok[(size_t)c * n + i];
        }
    }
    if (pos != frames) {
        snt_errf("acoustic: 帧数不一致（展开 %ld 期望 %ld）", pos, frames);
        goto done;
    }
    free(tok);
    tok = NULL;

    int frame_in = hidden + 3;
    fx = (float *)malloc((size_t)frame_in * (size_t)frames * sizeof(float));
    lat = (float *)malloc((size_t)hidden * (size_t)frames * sizeof(float));
    if (!fx || !lat) {
        snt_errf("acoustic: 内存不足（帧阶段 %ld 帧）", frames);
        goto done;
    }
    memcpy(fx, expanded, (size_t)hidden * (size_t)frames * sizeof(float));
    free(expanded);
    expanded = NULL;
    linspace01((int)frames, fx + (size_t)hidden * frames);
    float *token_pos = fx + (size_t)(hidden + 1) * frames;
    float *dur_pos = fx + (size_t)(hidden + 2) * frames;
    int token_count = n - 1 > 1 ? n - 1 : 1;
    pos = 0;
    for (int i = 0; i < n; i++) {
        int d = dur[i];
        float tp = (float)((double)i / (double)token_count);
        for (int k = 0; k < d; k++) token_pos[pos + k] = tp;
        if (d == 1) {
            dur_pos[pos] = 0.0f;
        } else {
            for (int k = 0; k < d; k++) dur_pos[pos + k] = (float)((double)k / (double)(d - 1));
        }
        pos += d;
    }
    conv1d_1x1(fx, frame_in, (int)frames, fip_w, fip_b, hidden, lat);
    free(fx);
    fx = NULL;
    int depth = count_blocks("acoustic/frame_blocks");
    scratch2 = (float *)malloc((size_t)2 * hidden * (size_t)frames * sizeof(float));
    if (!scratch2) {
        snt_errf("acoustic: 内存不足（帧阶段 scratch）");
        goto done;
    }
    for (int i = 0; i < depth; i++) {
        char prefix[SNT_MAX_NAME];
        snprintf(prefix, sizeof(prefix), "acoustic/frame_blocks.%d", i);
        if (residual_block(prefix, lat, hidden, (int)frames, scratch2) != 0) goto done;
    }
    conv1d_1x1(lat, hidden, (int)frames, ow_w, ow_b, out_ch, latent_out);
    rc = 0;
done:
    free(ids);
    free(dur);
    free(tx);
    free(tok);
    free(scratch);
    free(expanded);
    free(fx);
    free(lat);
    free(scratch2);
    return rc;
}

/* ---------------- piperlite decoder ---------------- */

static const int k_bank_kernels[3] = {3, 5, 7};
static const int k_bank_dil1[3] = {1, 2, 3};
static const int k_bank_dil2[3] = {2, 6, 12};

/* PiperResidualBank：三分支 y2 = conv2(lrelu(y1)) + y1，y1 = conv1(lrelu(x)) + x，
 * 输出 = 分支均值。x 在各分支间**不变**（models.py 里同理）。 */
static int residual_bank(const char *prefix, float *x, int ch, int T, float *acc, float *t,
                         float *y1, float *y2)
{
    memset(acc, 0, (size_t)ch * T * sizeof(float));
    for (int b = 0; b < 3; b++) {
        char w1[SNT_MAX_NAME], w2[SNT_MAX_NAME];
        snprintf(w1, sizeof(w1), "%s.blocks.%d.conv1.weight", prefix, b);
        snprintf(w2, sizeof(w2), "%s.blocks.%d.conv2.weight", prefix, b);
        snt_tensor *c1 = tget(w1);
        if (!c1) return -1;
        snt_tensor *c2 = tget(w2);
        if (!c2) return -1;
        const float *b1 = bias_of(w1);
        const float *b2 = bias_of(w2);
        if (!b1 || !b2) return -1;
        leaky_to(x, t, ch * T, 0.1f);
        conv1d_same(t, ch, T, tf32(w1), b1, ch, k_bank_kernels[b], k_bank_dil1[b], y1);
        for (int i = 0; i < ch * T; i++) y1[i] += x[i];
        leaky_to(y1, t, ch * T, 0.1f);
        conv1d_same(t, ch, T, tf32(w2), b2, ch, k_bank_kernels[b], k_bank_dil2[b], y2);
        for (int i = 0; i < ch * T; i++) acc[i] += y2[i] + y1[i];
    }
    float inv = 1.0f / 3.0f;
    for (int i = 0; i < ch * T; i++) x[i] = acc[i] * inv;
    return 0;
}

static int post_filter(float *audio, int samples)
{
    if (g_pf_channels <= 0) return 0;
    /* 与 models.py _apply_post_filter 一致：单声道 [1, T]。zh 语音包 pf_channels=0，
     * 这里只为别的 piperlite 语音包保留同语义实现。 */
    int K = g_pf_kernel > 0 ? g_pf_kernel : 9;
    int layers = g_pf_layers > 0 ? g_pf_layers : 0;
    float *r = (float *)malloc((size_t)g_pf_channels * samples * sizeof(float));
    float *t = (float *)malloc((size_t)g_pf_channels * samples * sizeof(float));
    float *u = (float *)malloc((size_t)g_pf_channels * samples * sizeof(float));
    float *out = (float *)malloc((size_t)samples * sizeof(float));
    if (!r || !t || !u || !out) {
        free(r); free(t); free(u); free(out);
        snt_errf("post filter 内存不足");
        return -1;
    }
    const float *inw = tf32("decoder/post_filter.in_conv.weight");
    const float *inb = bias_of("decoder/post_filter.in_conv.weight");
    const float *ow = tf32("decoder/post_filter.out_conv.weight");
    const float *ob = bias_of("decoder/post_filter.out_conv.weight");
    if (!inw || !inb || !ow || !ob) {
        free(r); free(t); free(u); free(out);
        return -1;
    }
    int in_k = tget("decoder/post_filter.in_conv.weight")->dim[2];
    conv1d_same(audio, 1, samples, inw, inb, g_pf_channels, in_k, 1, r);
    for (int layer = 0; layer < layers; layer++) {
        char base[SNT_MAX_NAME];
        snprintf(base, sizeof(base), "decoder/post_filter.units.%d", layer);
        char sc[SNT_MAX_NAME], w1[SNT_MAX_NAME], w2[SNT_MAX_NAME];
        snprintf(sc, sizeof(sc), "%s.scale", base);
        snprintf(w1, sizeof(w1), "%s.conv1.weight", base);
        snprintf(w2, sizeof(w2), "%s.conv2.weight", base);
        const float *scale = tf32(sc);
        const float *b1 = bias_of(w1);
        const float *b2 = bias_of(w2);
        snt_tensor *c1 = tget(w1);
        snt_tensor *c2 = tget(w2);
        if (!scale || !b1 || !b2 || !c1 || !c2) {
            free(r); free(t); free(u); free(out);
            return -1;
        }
        leaky_to(r, t, g_pf_channels * samples, 0.1f);
        conv1d_same(t, g_pf_channels, samples, tf32(w1), b1, g_pf_channels, c1->dim[2],
                    1 + layer, u);
        leaky_to(u, t, g_pf_channels * samples, 0.1f);
        conv1d_same(t, g_pf_channels, samples, tf32(w2), b2, g_pf_channels, c2->dim[2], 1, u);
        for (int i = 0; i < g_pf_channels * samples; i++) r[i] += scale[0] * u[i];
    }
    conv1d_same(r, g_pf_channels, samples, ow, ob, 1, K, 1, out);
    for (int i = 0; i < samples; i++) audio[i] = tanhf(audio[i] + g_pf_scale * out[i]);
    free(r); free(t); free(u); free(out);
    return 0;
}

static int decoder_forward(const float *latent, int frames, float *audio, int *samples_out)
{
    snt_tensor *pre = tget("decoder/pre.weight");
    snt_tensor *up0 = tget("decoder/up0.weight");
    snt_tensor *up1 = tget("decoder/up1.weight");
    snt_tensor *up2 = tget("decoder/up2.weight");
    snt_tensor *post = tget("decoder/post.weight");
    if (!pre || !up0 || !up1 || !up2 || !post) return -1;
    const float *pre_w = tf32(pre->name);
    const float *pre_b = bias_of("decoder/pre.weight");
    const float *post_w = tf32(post->name);
    const float *post_b = bias_of("decoder/post.weight");
    if (!pre_w || !pre_b || !post_w || !post_b) return -1;

    int in_ch = pre->dim[1], c0 = pre->dim[0];
    int c1 = up0->dim[1], c2 = up1->dim[1], c3 = up2->dim[1];
    int rc = -1;
    float *x = (float *)malloc((size_t)c0 * frames * sizeof(float));
    if (!x) return -1;
    conv1d_same(latent, in_ch, frames, pre_w, pre_b, c0, pre->dim[2], 1, x);

    snt_tensor *ups[3] = {up0, up1, up2};
    int chans[4] = {c0, c1, c2, c3};
    int strides[3] = {8, 8, 4};
    int pads[3] = {4, 4, 2};
    int len = frames;
    int cur = c0;
    for (int s = 0; s < 3; s++) {
        int K = ups[s]->dim[2];
        int L = (len - 1) * strides[s] - 2 * pads[s] + K;
        if (L <= 0) {
            snt_errf("decoder: 第 %d 级输出长度非法 %d", s, L);
            goto done;
        }
        leaky_to(x, x, cur * len, 0.1f);
        float *y = (float *)malloc((size_t)chans[s + 1] * L * sizeof(float));
        if (!y) {
            snt_errf("decoder: 内存不足（stage%d %d 帧）", s, L);
            goto done;
        }
        char bias_name[SNT_MAX_NAME];
        snprintf(bias_name, sizeof(bias_name), "decoder/up%d.weight", s);
        conv_transpose1d(x, cur, len, tf32(ups[s]->name), bias_of(bias_name), chans[s + 1], K,
                         strides[s], pads[s], L, y);
        free(x);
        x = y;
        len = L;
        cur = chans[s + 1];

        float *acc = (float *)malloc((size_t)cur * len * sizeof(float));
        float *t = (float *)malloc((size_t)cur * len * sizeof(float));
        float *y1 = (float *)malloc((size_t)cur * len * sizeof(float));
        float *y2 = (float *)malloc((size_t)cur * len * sizeof(float));
        if (!acc || !t || !y1 || !y2) {
            free(acc); free(t); free(y1); free(y2);
            snt_errf("decoder: 内存不足（resbank stage%d）", s);
            goto done;
        }
        char prefix[SNT_MAX_NAME];
        snprintf(prefix, sizeof(prefix), "decoder/res%d.0", s);
        int brc = residual_bank(prefix, x, cur, len, acc, t, y1, y2);
        free(acc); free(t); free(y1); free(y2);
        if (brc != 0) goto done;
    }

    leaky_to(x, x, cur * len, 0.01f);
    float *mono = (float *)malloc((size_t)len * sizeof(float));
    if (!mono) {
        snt_errf("decoder: 内存不足（输出）");
        goto done;
    }
    conv1d_same(x, cur, len, post_w, post_b, 1, post->dim[2], 1, mono);
    for (int i = 0; i < len; i++) mono[i] = tanhf(mono[i]);
    if (post_filter(mono, len) != 0) {
        free(mono);
        goto done;
    }
    memcpy(audio, mono, (size_t)len * sizeof(float));
    *samples_out = len;
    free(mono);
    rc = 0;
done:
    free(x);
    return rc;
}

/* ---------------- 对外 API ---------------- */

int snt_load(const char *layout_path, const char *weights_path)
{
    g_err[0] = 0;
    if (g_weights) return 0;
    if (load_weights(weights_path) != 0) return -1;
    if (load_layout(layout_path) != 0) return -1;
    return 0;
}

int snt_synth(const int32_t *ids, int n_ids, float length_scale, float **out_pcm,
              int *out_samples, int *out_frames)
{
    g_err[0] = 0;
    if (!g_weights) {
        snt_errf("未调用 snt_load");
        return -1;
    }
    if (!ids || n_ids <= 0) {
        snt_errf("音素 id 序列为空");
        return -1;
    }
    if (n_ids > g_max_tokens) {
        snt_errf("音素 id 过多：%d > max_tokens %d", n_ids, g_max_tokens);
        return -1;
    }
    if (!(length_scale > 0.0f)) length_scale = 1.0f;

    int32_t *durs = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
    if (!durs) return -1;
    if (duration_forward(ids, n_ids, length_scale, durs) != 0) {
        free(durs);
        return -1;
    }
    long frames = 0;
    for (int i = 0; i < n_ids; i++) frames += durs[i];

    snt_tensor *ow = tget("acoustic/output.weight");
    if (!ow) {
        free(durs);
        return -1;
    }
    int a_out = ow->dim[0];
    float *latent = (float *)malloc((size_t)a_out * (size_t)frames * sizeof(float));
    if (!latent) {
        free(durs);
        return -1;
    }
    if (acoustic_forward(ids, durs, n_ids, frames, latent) != 0) {
        free(durs);
        free(latent);
        return -1;
    }
    free(durs);

    float *audio = (float *)malloc((size_t)frames * (size_t)g_hop * sizeof(float));
    if (!audio) {
        free(latent);
        return -1;
    }
    int samples = 0;
    if (decoder_forward(latent, (int)frames, audio, &samples) != 0) {
        free(latent);
        free(audio);
        return -1;
    }
    free(latent);
    *out_pcm = audio;
    *out_samples = samples;
    if (out_frames) *out_frames = (int)frames;
    return 0;
}

void snt_free(void *p) { free(p); }
