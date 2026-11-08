/**
 * 本地播报（sanoTTS）插件自带配置页。
 *
 * 运行环境（平台侧契约，与 HXCS 的插件页同一套）：
 *   · `<iframe sandbox="allow-scripts">`，**没有** allow-same-origin ⇒ 本页的源是
 *     不透明 `null`，读不到平台 localStorage，也带不上 Authorization；
 *   · 路由下发的 CSP 里有 `connect-src 'none'` ⇒ 本页**一次网络请求都不能发**；
 *   · 因此所有数据交互只能经 postMessage 求宿主代办：
 *       握手（扁平 postMessage，宿主→本页）：{hx:1,type:'connect',plugin_id} + transfer 一个 MessagePort
 *       此后业务帧走**那个端口**：请求 {hx:1,id,op,payload}，应答 {hx:1,id,ok,result|error}
 *       可用 op：plugin.config.get / plugin.config.save / plugin.tool.call
 *       存活探测：{hx:1,type:'ping'} ⇒ 必须回 {hx:1,type:'pong',plugin_id}，
 *       不回 pong 的后果是宿主判定"文档已换"并**永久切断**通道（界面从此空白）。
 *
 * # 为什么用 WebAudio 而不是 <audio src=...>
 *
 * 平台的插件 UI CSP 只有 `img-src 'self' data:`，没有 media-src —— `<audio>` 用
 * `data:` 或 `blob:` 都会被浏览器拦掉。WebAudio 的 AudioBuffer **不经过任何 URL**，
 * 是唯一既符合 CSP 又不需要放开网络的播放方式：把 WAV 的 PCM 段解成 Int16，
 * 再写进 AudioBuffer，播放在本页内完成。
 */
(function () {
  'use strict';

  var HX = 1;
  var RPC_TIMEOUT_MS = 20000;

  var port = null;
  var pluginId = '';
  var pending = new Map();
  var seq = 0;
  var lastWav = null;
  var audioCtx = null;

  /**
   * 只读探针：给浏览器闸门断言"WebAudio 真的拿到了这一段音频"。
   *
   * 为什么需要它：`AudioBufferSourceNode.start()` 是**无声**的（声音进扬声器，
   * 不进 DOM），Playwright 读不到任何证据。没有探针时，"播出来了"只能靠 hint
   * 文案 —— 而那恰好是最容易"文案写了但根本没播"的地方。探针只记两个数：
   * AudioContext 的采样率与最后一次 AudioBuffer 的时长。
   */
  var probe = { sampleRate: 0, lastBufferDuration: 0 };
  window.__sanottsProbe = function () { return probe; };

  var $ = function (id) { return document.getElementById(id); };

  function setHint(el, text, kind) {
    el.textContent = text || '';
    el.dataset.kind = kind || '';
  }

  // ---------- 端口与 RPC ----------

  /**
   * 请求 id 生成器。
   *
   * # 为什么必须是**字符串**（照 HXCS 的口径）
   *
   * 宿主侧 `parsePluginUiMessage` 的判据是 `typeof frame.id !== 'string'` ⇒ 直接按
   * bad-id 丢弃整帧，**而且不回错误帧**（它认为那是坏帧，不是请求）。用数字 id 的
   * 后果是界面完全不工作：握手成功、配置页一片"—"、点按钮永远"合成中…"，
   * 而浏览器控制台一声不响 —— 因为丢掉是静默的。这与 SEC-M2 无关，纯粹是形状不对。
   *
   * 前缀 `ui-` + 时间基 + 单调序号：同一毫秒内多次调用也不会撞 id（宿主按 id 对账，
   * 重复 id 会被拒）。
   */
  function nextId() {
    seq += 1;
    return 'ui-' + Date.now().toString(36) + '-' + seq.toString(36);
  }

  function rpc(op, payload) {
    return new Promise(function (resolve, reject) {
      if (!port) { reject(new Error('与平台的通道尚未建立')); return; }
      var id = nextId();
      var timer = setTimeout(function () {
        pending.delete(id);
        reject(new Error('宿主没有在 ' + Math.round(RPC_TIMEOUT_MS / 1000) + ' 秒内应答'));
      }, RPC_TIMEOUT_MS);
      pending.set(id, { resolve: resolve, reject: reject, timer: timer });
      port.postMessage({ hx: HX, id: id, op: op, payload: payload || {} });
    });
  }

  function onPortMessage(ev) {
    var msg = ev.data;
    if (!msg || typeof msg !== 'object' || msg.hx !== HX) { return; }
    if (msg.type === 'ping') {
      port.postMessage({ hx: HX, type: 'pong', plugin_id: pluginId });
      return;
    }
    if (msg.type === 'host-ready') { return; }
    var entry = pending.get(msg.id);
    if (!entry) { return; }
    pending.delete(msg.id);
    clearTimeout(entry.timer);
    if (msg.ok) { entry.resolve(msg.result); return; }
    var err = msg.error || {};
    entry.reject(new Error((err.code ? err.code + '：' : '') + (err.message || '宿主返回失败')));
  }

  function attachPort(p) {
    port = p;
    port.onmessage = onPortMessage;
    // 主动读一次配置：宿主可能在本页脚本就绪前就给过 host-ready，
    // 两条路都能到，不会因为时序把界面卡在"加载中"。
    boot();
    startHeightReporting();
  }

  window.addEventListener('message', function (ev) {
    var data = ev.data;
    if (!data || typeof data !== 'object' || data.hx !== HX || data.type !== 'connect') { return; }
    // 握手帧的 event.origin 在不透明源下恒为 'null'，因此判据是
    // "端口必须随帧转移过来"（只有宿主能 transfer）。
    if (!ev.ports || !ev.ports.length) { return; }
    pluginId = data.plugin_id || '';
    attachPort(ev.ports[0]);
  });

  // ---------- 内容高度上报 ----------
  //
  // # 为什么必须由本页报（而不是平台自己量）
  //
  // iframe 是 opaque origin（sandbox 只给 allow-scripts），平台**读不到**它的
  // scrollHeight —— 跨源隔离的必然结果，不是实现没做。不报的话只能给固定高度，
  // 内容超出就在 iframe 内部滚，于是页面上出现"外层一条 + 内层一条"的双滚动条。
  //
  // 本页把实际内容高度经协议 op `plugin.ui.resize` 报给宿主，宿主把 iframe 高度
  // 设成它 —— 滚动交回平台页，与平台自己的页面一致。
  //
  // 去抖 150ms，且只在**整数像素**变化时发：子像素抖动会让宿主每帧重设高度。
  // 上报失败静默：这是体验优化，通道未就绪时刷错误只会误导用户。
  var lastHeight = 0;
  var heightTimer = null;

  function reportHeight() {
    var h = Math.ceil(document.documentElement.getBoundingClientRect().height);
    if (!(h > 0)) { return; }
    if (Math.abs(h - lastHeight) < 1) { return; }
    lastHeight = h;
    rpc('plugin.ui.resize', { height: h }).catch(function () {});
  }

  function scheduleHeight() {
    if (heightTimer) { clearTimeout(heightTimer); }
    heightTimer = setTimeout(reportHeight, 150);
  }

  function startHeightReporting() {
    reportHeight();
    if (typeof ResizeObserver !== 'function') {
      window.addEventListener('resize', scheduleHeight);
      return;
    }
    var ro = new ResizeObserver(scheduleHeight);
    ro.observe(document.documentElement);
    ro.observe(document.body);
    var main = document.querySelector('main');
    if (main) { ro.observe(main); }   // 试听/状态区是异步填充的，高度会变
  }

  // ---------- 页面逻辑 ----------

  function fmtMs(v) { return (v / 1000).toFixed(2) + ' 秒'; }

  function boot() {
    rpc('plugin.config.get').then(function (cfg) {
      if (cfg && typeof cfg === 'object') { applyConfig(cfg); }
    }).catch(function () { /* 读不到配置不致命：用默认值渲染 */ });
    refreshStatus();
  }

  function applyConfig(cfg) {
    if (cfg.voice) { $('voice').value = cfg.voice; }
    if (typeof cfg.speed === 'number') {
      $('speed').value = String(cfg.speed);
      $('speed-val').textContent = Number(cfg.speed).toFixed(2) + '×';
    }
    if (typeof cfg.max_chars === 'number') { $('max-chars').value = String(cfg.max_chars); }
  }

  function refreshStatus() {
    rpc('plugin.tool.call', { tool: 'status', args: {} }).then(function (st) {
      $('status').hidden = false;
      $('s-engine').textContent = st.engine_loaded ? 'sanoTTS C 内核已装载' : ('未装载：' + (st.load_error || '未知'));
      $('engine-pill').textContent = st.engine_loaded ? '就绪 · 离线' : '引擎异常';
      $('engine-pill').dataset.kind = st.engine_loaded ? 'ok' : 'bad';
      $('s-sr').textContent = (st.sample_rate || 0) + ' Hz · 单声道 16-bit';
      if (st.lexicon) {
        $('s-lexicon').textContent = st.lexicon.chars + ' 字 / ' + st.lexicon.phrases + ' 词组';
      }
      if (st.stats && st.stats.synth_calls) {
        $('s-last').textContent = st.stats.synth_calls + ' 次调用，最近 ' + st.stats.last_ms + ' ms';
      } else {
        $('s-last').textContent = '本代还没有合成过';
      }
      $('status-tip').textContent = '引擎与语音包都在本机：不联网、不上传文本。';
      return rpc('plugin.tool.call', { tool: 'voices', args: {} });
    }).then(function (vs) {
      var v = vs && vs.voices && vs.voices[0];
      if (v) {
        $('s-hash').textContent = (v.weights_sha256 || '').slice(0, 16) + '…';
        $('s-hash').title = v.weights_sha256 || '';
      }
      scheduleHeight();
    }).catch(function (err) {
      $('engine-pill').textContent = '状态不可用';
      $('engine-pill').dataset.kind = 'bad';
      $('status-tip').textContent = String(err.message || err);
    });
  }

  function callTool(tool, args) {
    return rpc('plugin.tool.call', { tool: tool, args: args || {} });
  }

  function base64ToBytes(b64) {
    var bin = atob(b64);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) { out[i] = bin.charCodeAt(i); }
    return out;
  }

  function concatBytes(parts) {
    var total = parts.reduce(function (n, p) { return n + p.length; }, 0);
    var out = new Uint8Array(total);
    var off = 0;
    parts.forEach(function (p) { out.set(p, off); off += p.length; });
    return out;
  }

  /** 逐块把整段 WAV 取回来。单次应答有大小限制，所以必须分块。 */
  function fetchWav(first) {
    var parts = [base64ToBytes(first.audio_b64)];
    var total = first.chunk_count || 1;
    function next(index) {
      if (index >= total) { return Promise.resolve(concatBytes(parts)); }
      return callTool('audio_chunk', { utterance_id: first.utterance_id, index: index })
        .then(function (chunk) {
          parts.push(base64ToBytes(chunk.audio_b64));
          return next(index + 1);
        });
    }
    return next(1);
  }

  /** WAV（PCM s16le mono）→ AudioBuffer，写进 AudioContext 播放。 */
  function playWav(bytes) {
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, false) !== 0x52494646) { throw new Error('返回的不是 WAV'); }
    var sampleRate = view.getUint32(24, true);
    var channels = view.getUint16(22, true);
    var bits = view.getUint16(34, true);
    var offset = 12;
    while (offset + 8 <= bytes.length) {
      var id = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
      var size = view.getUint32(offset + 4, true);
      if (id === 'data') { offset += 8; break; }
      offset += 8 + size + (size % 2);
    }
    if (bits !== 16 || channels !== 1) { throw new Error('只支持 16-bit 单声道 WAV'); }
    var pcm = new Int16Array(bytes.buffer, bytes.byteOffset + offset,
      Math.floor((bytes.length - offset) / 2));
    if (!audioCtx) { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    var buffers = audioCtx.sampleRate === sampleRate
      ? null
      : (audioCtx.createBuffer(1, pcm.length * audioCtx.sampleRate / sampleRate, audioCtx.sampleRate));
    var buffer = buffers || audioCtx.createBuffer(1, pcm.length, sampleRate);
    var data = buffer.getChannelData(0);
    if (buffers) {
      // 浏览器采样率与 22050 不同时按线性插值重采样：不重采样会以错误速度播放（音调变尖）。
      var ratio = sampleRate / audioCtx.sampleRate;
      for (var i = 0; i < data.length; i++) {
        var pos = i * ratio;
        var i0 = Math.floor(pos);
        var frac = pos - i0;
        var a = pcm[i0] || 0;
        var b = pcm[i0 + 1] || a;
        data[i] = (a + (b - a) * frac) / 32768;
      }
    } else {
      for (var j = 0; j < pcm.length; j++) { data[j] = pcm[j] / 32768; }
    }
    var src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);
    src.start();
    probe.lastBufferDuration = buffer.duration;
    probe.sampleRate = audioCtx.sampleRate;
    return buffer.duration;
  }

  function doSpeak() {
    var text = $('text').value.trim();
    var hint = $('speak-hint');
    if (!text) { setHint(hint, '先说一句话', 'bad'); return; }
    setHint(hint, '合成中…');
    $('play').disabled = true;
    var started = Date.now();
    callTool('speak', { text: text, speed: Number($('speed').value) })
      .then(function (first) {
        $('speak-metrics').hidden = false;
      scheduleHeight();
        $('m-duration').textContent = (first.duration_ms / 1000).toFixed(2) + ' 秒';
        $('m-synth').textContent = (first.synth_ms / 1000).toFixed(2) + ' 秒';
        $('m-rtf').textContent = String(first.rtf);
        $('m-phonemes').textContent = first.phonemes + ' / ' + first.frames;
        return fetchWav(first).then(function (wav) {
          lastWav = wav;
          $('download').disabled = false;
          var duration = playWav(wav);
          setHint(hint, '已播放 ' + duration.toFixed(2) + ' 秒（总耗时 ' + fmtMs(Date.now() - started) + '）', 'ok');
          scheduleHeight();
        });
      })
      .catch(function (err) {
        setHint(hint, String(err.message || err), 'bad');
        $('download').disabled = !lastWav;
      })
      .then(function () { $('play').disabled = false; });
  }

  function downloadWav() {
    if (!lastWav) { return; }
    // blob: 的 <a download> 是**导航**而不是取数，不受 connect-src 'none' 约束。
    var blob = new Blob([lastWav], { type: 'audio/wav' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'sanotts.wav';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  function saveConfig() {
    var cfg = {
      voice: $('voice').value,
      speed: Number($('speed').value),
      max_chars: Number($('max-chars').value)
    };
    setHint($('save-hint'), '保存中…');
    rpc('plugin.config.save', cfg).then(function () {
      setHint($('save-hint'), '已保存（下一次播报生效）', 'ok');
    }).catch(function (err) {
      setHint($('save-hint'), String(err.message || err), 'bad');
    });
  }

  $('play').addEventListener('click', doSpeak);
  $('download').addEventListener('click', downloadWav);
  $('save').addEventListener('click', saveConfig);
  $('speed').addEventListener('input', function () {
    $('speed-val').textContent = Number($('speed').value).toFixed(2) + '×';
  });

  // 宿主可能在本页脚本执行完之前就投递了 connect（端口已在队列里），
  // 也可能之后才投；两种时序都靠 attachPort 里的 boot() 收敛到同一状态。
  setHint($('speak-hint'), '点击「合成并播放」试听一段本地合成');
})();
