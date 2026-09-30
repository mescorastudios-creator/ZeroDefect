/* ZeroDefect in-browser backend for the no-install web version (built by backend/export_web.py).
   Answers the UI's /api/* requests without a server. The defect model is the same PatchCore model
   as the Python app (ResNet-18 features, memory bank of good patches, k-NN defect typing), run with
   TensorFlow.js. Production context, dashboard and factory map are placeholder data. */
(() => {
  const realFetch = window.fetch.bind(window);
  const SIZE = 224, GOOD_SHARE = 0.6, INTERVAL_MS = 1500, TOP_PATCHES = 8, K_NEIGHBOURS = 5;
  let D, COL, W = {}, BANKS = {}, BLOBS = [], URLS = [], MEAN, STD;
  const recentParts = new Map();

  const banner = document.createElement("div");
  banner.className = "note";
  banner.style.cssText = "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:10;max-width:calc(100vw - 32px)";
  document.body.appendChild(banner);
  const say = t => { banner.textContent = t; };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

  /* ---------- loading ---------- */
  const F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    F16[h] = s * (e === 0 ? f * 2 ** -24 : e === 31 ? (f ? NaN : Infinity) : 2 ** (e - 15) * (1 + f / 1024));
  }
  const f32 = (buf, off, n) => { const u = new Uint16Array(buf, off * 2, n), o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = F16[u[i]]; return o; };
  async function load(url, kind) {
    const r = await realFetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    if (kind === "json") return r.json();
    const bin = atob(await r.text()), u = new Uint8Array(bin.length);  // base64 text -> bytes
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u.buffer;
  }

  const dataReady = (async () => {  // enough for the dashboard, factory map and search
    D = await load("data/demo.json", "json");
    COL = Object.fromEntries(D.cols.map((c, i) => [c, i]));
  })();
  const ready = dataReady.then(async () => {  // the AI model and the demo part images
    say("Loading the AI model (about 30 MB, first time only)…");
    await tf.ready();
    MEAN = tf.tensor1d([0.485, 0.456, 0.406]); STD = tf.tensor1d([0.229, 0.224, 0.225]);
    const [wb, ib] = await Promise.all([load("model/backbone.txt"), load("data/images.txt")]);
    for (const [name, s] of Object.entries(D.backbone)) {
      const n = s.shape.reduce((a, b) => a * b, 1);
      W[name] = { k: tf.tensor4d(f32(wb, s.w, n), s.shape), b: tf.tensor1d(f32(wb, s.b, s.shape[3])) };
    }
    await Promise.all(Object.entries(D.parts).map(async ([pt, p]) => {
      const buf = await load(`model/${pt}.txt`), nb = p.bank * 384, nc = p.labels.length * 384;
      const bank = tf.tensor2d(f32(buf, 0, nb), [p.bank, 384]);
      BANKS[pt] = { ...p, bank, bankSq: bank.square().sum(1), cls: tf.tensor2d(f32(buf, nb, nc), [p.labels.length, 384]) };
    }));
    BLOBS = D.samples.map(s => s.views.map(v => new Blob([new Uint8Array(ib, v.off, v.len)], { type: "image/jpeg" })));
    URLS = BLOBS.map(vs => vs.map(b => URL.createObjectURL(b)));
    say("Starting the AI model…");
    await inspectViews(D.samples[0].part_type, await Promise.all(BLOBS[0].map(b => createImageBitmap(b))));
    const chip = document.querySelector(".chip.ai");
    if (chip) chip.textContent += " · runs in your browser";
    banner.remove();
  });
  ready.catch(e => say(`The AI model could not start: ${e.message}. Reload the page to try again.`));

  /* ---------- the model: same steps as inspection/patchcore.py ---------- */
  function conv(x, name, stride, pad, relu = true) {
    const w = W[name];
    if (pad) x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
    const y = tf.conv2d(x, w.k, stride, "valid").add(w.b);
    return relu ? y.relu() : y;
  }
  function block(x, name, stride) {
    const y = conv(conv(x, `${name}.conv1`, stride, 1), `${name}.conv2`, 1, 1, false);
    const skip = W[`${name}.downsample`] ? conv(x, `${name}.downsample`, stride, 0, false) : x;
    return y.add(skip).relu();
  }
  const features = bmps => tf.tidy(() => {  // -> [B, 28, 28, 384]
    let x = tf.stack(bmps.map(b => {  // same steps as inspection.patchcore.prepare
      let im = tf.browser.fromPixels(b).toFloat();
      const k = Math.floor(Math.min(b.width, b.height) / SIZE);
      if (k >= 2) im = tf.avgPool(im.slice([0, 0, 0], [Math.floor(b.height / k) * k, Math.floor(b.width / k) * k, 3]), k, k, "valid");
      return tf.image.resizeBilinear(im, [SIZE, SIZE], false, true).round().div(255).sub(MEAN).div(STD);
    }));
    x = tf.maxPool(tf.pad(conv(x, "conv1", 2, 3), [[0, 0], [1, 1], [1, 1], [0, 0]]), 3, 2, "valid");
    x = block(block(x, "layer1.0", 1), "layer1.1", 1);
    const f2 = block(block(x, "layer2.0", 2), "layer2.1", 1);
    const f3 = tf.image.resizeBilinear(block(block(f2, "layer3.0", 2), "layer3.1", 1), [28, 28], false, true);
    return tf.avgPool(tf.pad(tf.concat([f2, f3], 3), [[0, 0], [1, 1], [1, 1], [0, 0]]), 3, 1, "valid");
  });
  const nearest = (q, P) => tf.tidy(() =>  // distance of each patch to its nearest good patch
    q.square().sum(1, true).add(P.bankSq.reshape([1, -1])).sub(q.matMul(P.bank, false, true).mul(2)).min(1).relu().sqrt());
  function blur(m, sigma) {  // Gaussian blur with reflected borders, like cv2.GaussianBlur
    const k = Math.round(sigma * 8 + 1) | 1, r = (k - 1) / 2;
    const g = Array.from({ length: k }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
    const sum = g.reduce((a, b) => a + b), kern = tf.tensor1d(g.map(v => v / sum));
    const y = tf.conv2d(tf.mirrorPad(m, [[0, 0], [r, r], [0, 0], [0, 0]], "reflect"), kern.reshape([k, 1, 1, 1]), 1, "valid");
    return tf.conv2d(tf.mirrorPad(y, [[0, 0], [0, 0], [r, r], [0, 0]], "reflect"), kern.reshape([1, k, 1, 1]), 1, "valid");
  }
  async function classify(feat, low, P) {  // anomaly-weighted mean of the top patches -> k-NN vote
    const top = Array.from(low.keys()).sort((a, b) => low[b] - low[a]).slice(0, TOP_PATCHES);
    const simsT = tf.tidy(() => {
      const v = tf.gather(feat, top).mul(tf.tensor2d(top.map(i => low[i]), [TOP_PATCHES, 1])).sum(0);
      return P.cls.matMul(v.div(v.norm()).reshape([384, 1])).reshape([-1]);
    });
    const sims = await simsT.data();
    simsT.dispose();
    const best = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, K_NEIGHBOURS);
    const votes = {};
    for (const i of best) votes[P.labels[i]] = (votes[P.labels[i]] || 0) + Math.max(sims[i], 0);
    const label = Object.keys(votes).reduce((a, b) => (votes[b] > votes[a] ? b : a));
    const total = Object.values(votes).reduce((a, b) => a + b, 0);
    return [label, total ? votes[label] / total : 1];
  }
  function findBoxes(hm, w, h, thr, cls) {  // regions above 75% of the threshold that reach the threshold
    const fg = new Uint8Array(w * h);
    for (let i = 0; i < fg.length; i++) fg[i] = hm[i] > 0.75 * thr ? 1 : 0;
    const r = Math.max(5, Math.floor(w / 50)), row = new Uint8Array(w * h), dil = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let on = 0; for (let d = Math.max(0, x - r); d <= Math.min(w - 1, x + r) && !on; d++) on = fg[y * w + d];
      row[y * w + x] = on;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let on = 0; for (let d = Math.max(0, y - r); d <= Math.min(h - 1, y + r) && !on; d++) on = row[d * w + x];
      dil[y * w + x] = on;
    }
    const lab = new Int32Array(w * h), out = [];
    for (let start = 0, n = 0; start < dil.length; start++) {
      if (!dil[start] || lab[start]) continue;
      n++;
      let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
      const stack = [start]; lab[start] = n;
      while (stack.length) {
        const i = stack.pop(), x = i % w, y = (i - x) / w;
        if (fg[i]) { area++; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy, j = ny * w + nx;
          if (nx >= 0 && ny >= 0 && nx < w && ny < h && dil[j] && !lab[j]) { lab[j] = n; stack.push(j); }
        }
      }
      if (area < 4) continue;
      let peak = -Infinity;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) peak = Math.max(peak, hm[y * w + x]);
      if (peak > thr) out.push({ x: 100 * x0 / w, y: 100 * y0 / h, w: 100 * (x1 - x0 + 1) / w, h: 100 * (y1 - y0 + 1) / h, cls });
    }
    return out;
  }
  function heatURL(hm, w, h, thr) {  // transparent jet overlay, clear where the part looks normal
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const ctx = c.getContext("2d"), im = ctx.createImageData(w, h), cl = v => Math.min(1, Math.max(0, v));
    for (let i = 0; i < hm.length; i++) {
      const lv = cl((hm[i] - 0.8 * thr) / (0.8 * thr));
      im.data[4 * i] = 255 * cl(1.5 - Math.abs(4 * lv - 3)); im.data[4 * i + 1] = 255 * cl(1.5 - Math.abs(4 * lv - 2));
      im.data[4 * i + 2] = 255 * cl(1.5 - Math.abs(4 * lv - 1)); im.data[4 * i + 3] = 190 * cl(lv * 3);
    }
    ctx.putImageData(im, 0, 0);
    return c.toDataURL("image/png");
  }
  async function analyse(P, feats, bmps) {
    const flat = feats.reshape([-1, 384]);
    const low = nearest(flat, P);
    const views = [];
    for (let i = 0; i < bmps.length; i++) {
      const w = bmps[i].width, h = bmps[i].height;
      const lowView = low.slice([i * 784], [784]);
      const hmT = tf.tidy(() => blur(tf.image.resizeBilinear(lowView.reshape([1, 28, 28, 1]), [h, w], false, true), 4 * w / 256));
      const [hm, lowArr] = [await hmT.data(), await lowView.data()];
      hmT.dispose(); lowView.dispose();
      let score = -Infinity; for (let j = 0; j < hm.length; j++) if (hm[j] > score) score = hm[j];
      const v = { score, defect: score > P.threshold, cls: null, confidence: null, boxes: [] };
      if (v.defect) {
        const feat = flat.slice([i * 784, 0], [784, 384]);
        [v.cls, v.confidence] = await classify(feat, lowArr, P);
        feat.dispose();
        v.boxes = findBoxes(hm, w, h, P.threshold, v.cls);
      }
      v.heatmap = heatURL(hm, w, h, P.threshold);
      views.push(v);
    }
    flat.dispose(); low.dispose();
    return views;
  }
  async function inspectViews(pt, bmps, feats = null) {
    const t0 = performance.now(), P = BANKS[pt];
    const own = !feats;
    feats = feats || features(bmps);
    const views = await analyse(P, feats, bmps);
    if (own) feats.dispose();
    const flagged = views.filter(v => v.defect), defect = flagged.length >= Math.min(D.min_views, views.length);
    let cls = null, confidence = null;
    if (defect) {
      const votes = {};
      for (const v of flagged) votes[v.cls] = (votes[v.cls] || 0) + v.confidence;
      cls = Object.keys(votes).reduce((a, b) => (votes[b] > votes[a] ? b : a));
      confidence = votes[cls] / Object.values(votes).reduce((a, b) => a + b, 0);
    }
    return { defect, cls, confidence, score: Math.max(...views.map(v => v.score)), threshold: P.threshold,
             flagged_views: flagged.length, ms: Math.round(performance.now() - t0), views };
  }

  /* ---------- parts ---------- */
  function partJSON(rec, s, label) {
    const c = n => rec[COL[n]];
    const params = Object.fromEntries(D.cols.slice(COL.melt_temp, COL.sample).map(p => [p, c(p)]));
    return {
      id: c("part_id"), ts: c("timestamp"), part_type: c("part_type"), label,
      context: { shift_date: c("shift_date"), shift: c("shift"), line: c("line_id"), machine: c("machine_id"),
                 mould: c("mould_id"), cavity: c("cavity"), material: c("material"), operator: c("operator_id"),
                 supplier: c("supplier_id"), lot: c("resin_lot"), params },
      views: D.samples[s].views.map((v, i) => ({ view: v.view, url: URLS[s][i] })),
    };
  }
  async function inspectRecord(rec, s) {
    const id = rec[COL.part_id];
    if (recentParts.has(id)) return recentParts.get(id);
    const smp = D.samples[s];
    const res = await inspectViews(smp.part_type, await Promise.all(BLOBS[s].map(b => createImageBitmap(b))));
    const out = partJSON(rec, s, smp.label);
    out.result = { defect: res.defect, cls: res.cls, confidence: res.confidence, score: res.score,
                   threshold: res.threshold, flagged_views: res.flagged_views, ms: res.ms };
    out.views.forEach((v, i) => Object.assign(v, res.views[i]));
    recentParts.set(id, out);
    if (recentParts.size > 200) recentParts.delete(recentParts.keys().next().value);
    return out;
  }
  const byType = () => {
    const m = {};
    D.samples.forEach((s, i) => { (m[s.part_type] ??= { good: [], defect: [] })[s.label ? "defect" : "good"].push(i); });
    return m;
  };

  /* ---------- the /api routes ---------- */
  async function route(url, init = {}) {
    const u = new URL(url, location.href), path = u.pathname.replace(/^.*\/api\//, "/api/"), q = u.searchParams;
    if (path === "/api/config") return json(D.config);
    if (path === "/api/model") return json(D.model);
    if (path === "/api/summary") return json(D.summary);
    if (path === "/api/machines") return json(D.machines);
    let m;
    if ((m = path.match(/^\/api\/machines\/([^/]+)\/history$/))) return D.history[m[1]] ? json(D.history[m[1]]) : json({ detail: "unknown machine" }, 404);
    if (path === "/api/search") {
      const qq = (q.get("q") || "").trim().toUpperCase(), machine = q.get("machine") || "", result = q.get("result") || "";
      const limit = Math.min(+(q.get("limit") || 100), 500), rows = [];
      let total = 0;
      for (let i = D.records.length - 1; i >= 0; i--) {
        const r = D.records[i];
        if (qq && !(r[COL.part_id].includes(qq) || r[COL.resin_lot].includes(qq) || r[COL.operator_id] === qq)) continue;
        if (machine && r[COL.machine_id] !== machine) continue;
        if (result === "good" && r[COL.true_class]) continue;
        if (result === "defect" && !r[COL.true_class]) continue;
        total++;
        if (rows.length < limit) rows.push({ part_id: r[COL.part_id], timestamp: r[COL.timestamp], machine_id: r[COL.machine_id],
          cavity: r[COL.cavity], operator_id: r[COL.operator_id], resin_lot: r[COL.resin_lot], true_class: r[COL.true_class] });
      }
      return json({ total, rows });
    }
    if ((m = path.match(/^\/api\/parts\/(.+)$/))) {
      const id = decodeURIComponent(m[1]), rec = D.records.find(r => r[COL.part_id] === id);
      if (recentParts.has(id)) return json(recentParts.get(id));
      return rec ? json(await inspectRecord(rec, rec[COL.sample])) : json({ detail: "unknown part" }, 404);
    }
    if (path === "/api/test-image") {
      const want = q.get("defect") === "true", pool = D.samples.map((s, i) => i).filter(i => !!D.samples[i].label === want);
      const s = pool[Math.floor(Math.random() * pool.length)], v = Math.floor(Math.random() * BLOBS[s].length);
      return new Response(BLOBS[s][v], { headers: { "Content-Type": "image/jpeg", "X-Label": D.samples[s].label || "good", "X-Part-Type": D.samples[s].part_type } });
    }
    if (path === "/api/inspect") {
      const body = init.body;
      if (!(body instanceof Blob)) return json({ detail: "No image received" }, 400);
      if (body.size > 20 * 1024 * 1024) return json({ detail: "Image is larger than 20 MB" }, 413);
      let bmp;
      try { bmp = await createImageBitmap(body); } catch { return json({ detail: "Not an image file (use JPEG, PNG or BMP)" }, 400); }
      const scale = 1024 / Math.max(bmp.width, bmp.height);
      if (scale < 1) bmp = await createImageBitmap(bmp, { resizeWidth: Math.round(bmp.width * scale), resizeHeight: Math.round(bmp.height * scale), resizeQuality: "medium" });
      let pt = q.get("part_type") || "auto";
      if (pt !== "auto" && !BANKS[pt]) return json({ detail: `Unknown part type '${pt}'` }, 400);
      const feats = features([bmp]);
      if (pt === "auto") {  // the part type whose model finds the image least unusual
        let best = Infinity;
        for (const [k, P] of Object.entries(BANKS)) {
          const t = tf.tidy(() => nearest(feats.reshape([-1, 384]), P).max()), s = t.dataSync()[0] / P.threshold;
          t.dispose();
          if (s < best) { best = s; pt = k; }
        }
      }
      const res = await inspectViews(pt, [bmp], feats);
      feats.dispose();
      const v = res.views[0];
      return json({ part_type: pt, defect: res.defect, cls: res.cls, confidence: res.confidence, score: res.score,
                    threshold: res.threshold, flagged_views: res.flagged_views, ms: res.ms, boxes: v.boxes, heatmap: v.heatmap });
    }
    return json({ detail: "not found" }, 404);
  }

  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (!/^\/api\//.test(url)) return realFetch(input, init);
    await (/^\/api\/(parts|inspect|test-image)/.test(url) ? ready : dataReady);
    return route(url, init);
  };

  /* ---------- the live line ---------- */
  window.EventSource = class {
    constructor() {
      this.onmessage = null; this.onerror = null;
      ready.then(() => this.run()).catch(e => this.onerror?.(e));
    }
    async run() {
      const pools = byType();
      for (let i = 0; ; i++) {
        const t0 = performance.now(), rec = D.live[i % D.live.length], pt = rec[COL.part_type];
        const pool = pools[pt][Math.random() < GOOD_SHARE ? "good" : "defect"];
        const ev = await inspectRecord([...rec.slice(0, -1), null], pool[Math.floor(Math.random() * pool.length)]);
        recentParts.delete(ev.id);  // the live line reuses record ids; always inspect afresh
        this.onmessage?.({ data: JSON.stringify(ev) });
        await sleep(Math.max(0, INTERVAL_MS - (performance.now() - t0)));
      }
    }
    close() {}
  };

  /* Check the browser against the Python model: inspect every shipped test part. */
  window.__zdSelfTest = async (n = Infinity, step = 1, start = 0) => {
    await ready;
    const out = [];
    for (let s = start; s < D.samples.length && out.length < n; s += step) {
      const res = await inspectViews(D.samples[s].part_type, await Promise.all(BLOBS[s].map(b => createImageBitmap(b))));
      out.push({ id: D.samples[s].id, defect: res.defect, cls: res.cls, score: res.score, view_scores: res.views.map(v => v.score), ms: res.ms });
    }
    return out;
  };
})();
