/* ============================================================
   deck_figs.js — CheXVision DECK figures
   Same figlib visual system as the report (isometric tensor
   slabs, slate-blue pathway, coral = primary multi-label task),
   re-proportioned to fill each 16:9 slide region exactly.
   Region aspect ratios (measured from the deck):
     dual-task  1728×619  → 2.79
     se-resnet  1274×664  → 1.92
     densenet   1274×608  → 2.10
     finetune   1728×514  → 3.36
     pipeline    430×936  → 0.46  (tall column)
   Composition system for the model slides (4 / 6 / 7):
     a backbone reads left→right on top, funnels through one
     centred "shared 512-d" vector, and splits symmetrically to
     two head cards — primary multi-label (coral) and auxiliary
     binary (blue). Same grammar, re-proportioned per region.
   Depends on figlib.js (window.FL).
   ============================================================ */
(function (global) {
  'use strict';
  const f = FL, C = FL.C, RAMP = FL.RAMP;

  // a labelled connector chip centred on a point (sits on a rail)
  function chipLabel(s, cx, cy, txt, w) {
    let r = `<rect x="${cx - w / 2}" y="${cy - 13}" width="${w}" height="26" rx="13" fill="#ffffff" stroke="${RAMP[1].s}" stroke-width="1.1"/>`;
    r += f.T(cx, cy + 4.5, txt, { size: 12.5, mono: true, weight: 600, fill: C.blueDk });
    return s + r;
  }

  // compact side-by-side task-head card. kind = 'ml' | 'bin'. fixed height.
  function headCard(x, y, w, kind) {
    const isML = kind === 'ml';
    const h = 156;
    const accent = isML ? C.coral : C.blue;
    const bg = isML ? '#fffcfa' : '#fafcff';
    const px = x + 30;
    let r = f.box(x, y, w, h, { stroke: C.hair, accent: accent, fill: bg, r: 13 });
    r += f.T(px, y + 41, isML ? 'Multi-label head' : 'Binary head', { anchor: 'start', size: 20, weight: 700, fill: C.ink });
    r += f.T(x + w - 26, y + 41, isML ? 'primary task' : 'auxiliary', { anchor: 'end', size: 12, mono: true, upper: true, ls: 0.5, fill: accent, weight: 600 });
    r += f.T(px, y + 75, isML ? 'Linear 512 → 14   ·   sigmoid' : 'Linear 512 → 1   ·   sigmoid', { anchor: 'start', size: 15, mono: true, fill: C.ink2 });
    r += f.T(px, y + 100, isML ? 'weighted BCE · per-class pos_weight' : 'BCE   ·   Normal vs. Abnormal', { anchor: 'start', size: 13.5, mono: true, fill: C.mut });
    if (isML) {
      const n = 14, sw = 18, inner = w - 60, step = (inner - sw) / (n - 1);
      for (let i = 0; i < n; i++) {
        const tx = px + i * step, on = i % 4 === 0;
        r += `<rect x="${tx}" y="${y + 119}" width="${sw}" height="24" rx="3" fill="${on ? C.coral : '#f2d9cd'}" stroke="${C.coral}" stroke-width="0.7" opacity="${on ? 0.92 : 0.55}"/>`;
      }
      r += f.T(px + inner + 2, y + 136, '14', { anchor: 'end', size: 11, fill: C.mut, opacity: 0 });
    } else {
      const bw = (w - 60 - 14) / 2;
      r += `<rect x="${px}" y="${y + 119}" width="${bw}" height="24" rx="5" fill="#eef2f8" stroke="${C.okline}" stroke-width="0.9"/>`;
      r += f.T(px + bw / 2, y + 135, 'Normal', { size: 12.5, mono: true, fill: C.blueDk, weight: 600 });
      r += `<rect x="${px + bw + 14}" y="${y + 119}" width="${bw}" height="24" rx="5" fill="${RAMP[2].f}" stroke="${C.okline}" stroke-width="0.9"/>`;
      r += f.T(px + bw + 14 + bw / 2, y + 135, 'Abnormal', { size: 12.5, mono: true, fill: C.blueDk, weight: 600 });
    }
    return { svg: r, h: h };
  }

  /* ---------------------------------------------------------
     D1 · Dual-task framing  (slide 4 · 2.79)
     Wide horizontal pass → split to two stacked heads →
     converge on the dark combined-loss terminal.
     --------------------------------------------------------- */
  global.drawDeckDualTask = function (id) {
    const W = 1860, H = 666;
    let s = '';
    const baseline = 470, aY = 446;

    // Front face (w×h) is the SPATIAL map → kept square; side ∝ resolution
    // (320,80,40,20,10), monotonically shrinking. depth ∝ channel count.
    const stack = [
      { x: 70,  w: 144, h: 144, depth: 8,  lvl: 0, img: true, shape: '3 × 320 × 320' },
      { x: 244, w: 108, h: 108, depth: 26, lvl: 1, shape: '80² · 64' },
      { x: 400, w: 84,  h: 84,  depth: 40, lvl: 2, shape: '40² · 128' },
      { x: 546, w: 66,  h: 66,  depth: 52, lvl: 3, shape: '20² · 256' },
      { x: 686, w: 50,  h: 50,  depth: 64, lvl: 4, shape: '10² · 512' },
    ];

    s += f.bracketT(stack[1].x, stack[4].x + stack[4].w, 112,
      'Shared backbone   ·   SE-ResNet or DenseNet-121   ·   one forward pass',
      { rise: 11, size: 18, weight: 600, labelFill: C.ink2, color: C.hair });

    stack.forEach((st, i) => {
      const y = baseline - st.h;
      s += f.slab(st.x, y, st.w, st.h, st.depth, st.lvl, { frontStripes: st.img, clip: 'cld1_' + i });
      s += f.T(st.x + st.w / 2, 158, st.shape, { size: 15, mono: true, fill: C.mut, weight: 500 });
    });
    s += f.T(stack[0].x + stack[0].w / 2, baseline + 40, 'Input', { size: 19, weight: 700, fill: C.ink });
    s += f.T((stack[1].x + stack[4].x + stack[4].w) / 2, baseline + 40, 'Convolutional feature extractor', { size: 16, fill: C.mut });

    for (let i = 0; i < stack.length - 1; i++) {
      const a = stack[i], b = stack[i + 1];
      s += f.arrow(a.x + a.w + 4, aY, b.x - 6, aY, { color: C.ink2, width: 2.1 });
    }

    // global avg pool
    const gx = 848, gy = 294, gw = 170, gh = 152;
    s += f.arrow(stack[4].x + stack[4].w + 4, aY, gx - 6, gy + gh / 2, { color: C.ink2, width: 2.1 });
    s += f.box(gx, gy, gw, gh, { stroke: C.hair, fill: '#fbfcfe', r: 14 });
    s += f.lines(gx + gw / 2, gy + gh / 2 - 8, [
      { t: 'Global', size: 20, weight: 700, fill: C.ink },
      { t: 'avg pool', size: 20, weight: 700, fill: C.ink },
    ], { lh: 27 });

    // 512-d vector
    const vx = 1066, vcy = 370;
    s += f.arrow(gx + gw + 4, vcy, vx - 11, vcy, { color: C.ink2, width: 2.1 });
    const vec = f.vector(vx, vcy, { label: '512-d', sub: 'shared', lvl: 2 });
    s += vec.svg;

    // two heads, stacked, fed by symmetric split
    const hx = 1180, hw = 448, splitX = 1150, mlY = 250, binY = 496;
    s += f.dualHeads({ vecRight: vec.right, vcy: vcy, splitX: splitX, hx: hx, hw: hw, mlY: mlY, binY: binY });

    // combined loss — dark terminal both heads converge on
    const lx = 1676, lw = 180, lcy = (mlY + binY) / 2, lh = 132, ly = lcy - lh / 2, ex = 1644;
    s += f.poly([[hx + hw, mlY], [ex, mlY], [ex, lcy - 12], [lx - 6, lcy - 12]], { color: C.coralLine, width: 2.1 });
    s += f.poly([[hx + hw, binY], [ex, binY], [ex, lcy + 12], [lx - 6, lcy + 12]], { color: C.blue, width: 2.1 });
    s += `<rect x="${lx}" y="${ly}" width="${lw}" height="${lh}" rx="15" fill="#20242c"/>`;
    s += f.T(lx + lw / 2, ly + 44, 'Combined', { size: 19, weight: 700, fill: '#ffffff' });
    s += f.T(lx + lw / 2, ly + 69, 'loss', { size: 19, weight: 700, fill: '#ffffff' });
    s += f.T(lx + lw / 2, ly + 98, '1.0 · L_ml', { size: 14.5, mono: true, fill: '#e9b9a4' });
    s += f.T(lx + lw / 2, ly + 118, '+ 0.5 · L_bin', { size: 14.5, mono: true, fill: '#9cc0f0' });

    f.mount(id, W, H, s, { bg: false });
  };

  /* ---------------------------------------------------------
     Shared funnel for the square-ish model slides (6 & 7).
     Backbone is drawn by the caller; this draws the centred
     pooled-vector rail and the symmetric split to two heads.
     opts: { srcCx, srcBottom, railY, vcy, splitY, cardY,
             cardW, mlCx, binCx, railLabel }
     --------------------------------------------------------- */
  function funnelToHeads(opts) {
    let s = '';
    const vcx = opts.cx;
    const vec = f.vector(vcx - 12, opts.vcy, { ncell: 8, w: 24, lvl: 2 });
    // rail: backbone source → down → across → into vector
    s += f.poly([[opts.srcCx, opts.srcBottom], [opts.srcCx, opts.railY], [vcx + 12, opts.railY]],
      { color: C.blue, width: 2, noHead: true });
    s = chipLabel(s, (vcx + 12 + opts.srcCx) / 2, opts.railY, opts.railLabel, opts.railLabelW || 198);
    s += vec.svg;
    // split: vector bottom → distribution rail → two cards
    const vb = opts.vcy + 48;
    s += `<line x1="${vcx}" y1="${vb}" x2="${vcx}" y2="${opts.splitY}" stroke="${C.ink2}" stroke-width="2"/>`;
    s += `<line x1="${opts.mlCx}" y1="${opts.splitY}" x2="${opts.binCx}" y2="${opts.splitY}" stroke="${C.ink2}" stroke-width="2"/>`;
    s += f.arrow(opts.mlCx, opts.splitY, opts.mlCx, opts.cardY, { color: C.coralLine, width: 2 });
    s += f.arrow(opts.binCx, opts.splitY, opts.binCx, opts.cardY, { color: C.blue, width: 2 });
    const ml = headCard(opts.mlCx - opts.cardW / 2, opts.cardY, opts.cardW, 'ml');
    const bn = headCard(opts.binCx - opts.cardW / 2, opts.cardY, opts.cardW, 'bin');
    s += ml.svg + bn.svg;
    return s;
  }

  /* ---------------------------------------------------------
     D2 · Custom SE-ResNet  (slide 6 · 1.92)
     Backbone row on top → centred funnel → two heads below.
     --------------------------------------------------------- */
  global.drawDeckSEResNet = function (id) {
    const W = 1300, H = 677;
    let s = '';
    const baseline = 286, aY = 262;

    // Square front faces: side ∝ spatial resolution (320→80→40→20→10),
    // Stem & Stage 1 share 80² so share a side. depth ∝ channels.
    const stages = [
      { x: 58,  w: 132, h: 132, depth: 8,  lvl: 0, img: true, shape: '320² · 3' },
      { x: 234, w: 96,  h: 96,  depth: 18, lvl: 1, shape: '80² · 64' },
      { x: 384, w: 96,  h: 96,  depth: 18, lvl: 1, front: '×3', shape: '80² · 64' },
      { x: 534, w: 76,  h: 76,  depth: 30, lvl: 2, front: '×4', ds: true, shape: '40² · 128' },
      { x: 676, w: 60,  h: 60,  depth: 44, lvl: 3, front: '×6', ds: true, shape: '20² · 256' },
      { x: 816, w: 46,  h: 46,  depth: 60, lvl: 4, front: '×3', ds: true, shape: '10² · 512' },
    ];

    s += f.bracketT(stages[2].x, stages[5].x + stages[5].w, 70,
      'Four residual stages   ·   depths [3, 4, 6, 3]   ·   widths 64 → 512   ·   ~23 M params, from scratch',
      { rise: 10, size: 14.5, weight: 600, labelFill: C.ink2, color: C.hair });

    stages.forEach(st => {
      const y = baseline - st.h;
      s += f.slab(st.x, y, st.w, st.h, st.depth, st.lvl, {
        front: st.front, frontSize: 19, frontFill: st.lvl >= 4 ? '#fff' : C.ink,
        frontStripes: st.img, clip: 'clD2_' + st.x,
      });
      s += f.T(st.x + st.w / 2, 118, st.shape, { size: 13.5, mono: true, fill: C.mut, weight: 500 });
    });
    s += f.T(stages[0].x + stages[0].w / 2, baseline + 30, 'Input', { size: 15, weight: 700, fill: C.ink });
    s += f.T(stages[1].x + stages[1].w / 2, baseline + 30, 'Stem', { size: 15, weight: 700, fill: C.ink });
    for (let i = 0; i < stages.length - 1; i++) {
      const a = stages[i], b = stages[i + 1];
      s += f.arrow(a.x + a.w + 4, aY, b.x - 6, aY, { color: C.ink2, width: 2 });
      if (b.ds) s += f.T((a.x + a.w + b.x) / 2, aY - 11, '↓2', { size: 14, mono: true, fill: C.coral, weight: 600 });
    }

    // global avg pool
    const gx = 940, gw = 160, gh = 104, gy = baseline - gh;
    s += f.arrow(stages[5].x + stages[5].w + 4, aY, gx - 6, gy + gh / 2, { color: C.ink2, width: 2 });
    s += f.box(gx, gy, gw, gh, { stroke: C.hair, fill: '#fbfcfe', r: 13 });
    s += f.lines(gx + gw / 2, gy + 44, [
      { t: 'Global avg pool', size: 16, weight: 700, fill: C.ink },
      { t: 'Dropout 0.5', size: 13.5, mono: true, fill: C.mut },
    ], { lh: 25 });

    s += funnelToHeads({
      cx: 650, vcy: 360, srcCx: gx + gw / 2, srcBottom: baseline, railY: 360,
      splitY: 442, cardY: 486, cardW: 464, mlCx: 352, binCx: 948,
      railLabel: 'pooled 512-d features', railLabelW: 200,
    });

    f.mount(id, W, H, s, { bg: false });
  };

  /* ---------------------------------------------------------
     D3 · DenseNet-121  (slide 7 · 2.10)
     Backbone + feature layer on top → centred funnel → heads.
     --------------------------------------------------------- */
  global.drawDeckDenseNet = function (id) {
    const W = 1300, H = 620;
    let s = '';
    const baseline = 268, aY = 250;

    // Square front faces: side ∝ spatial resolution. Each dense block and the
    // transition OUT of it share a resolution-stage side (80,40,20,10 from the
    // shape labels). Transitions are full squares (not thin plates) so the
    // spatial halving reads; their depth shrinks as they reduce channels.
    const stages = [
      { x: 52,  w: 88, h: 88, depth: 6,  lvl: 0, img: true, name: 'Input', shape: '320² · 3' },
      { x: 158, w: 66, h: 66, depth: 14, lvl: 1, name: 'Stem',  shape: '80² · 64' },
      { x: 250, w: 66, h: 66, depth: 30, lvl: 2, name: 'Dense 1', sub: '×6',  front: '×6',  shape: '80² · 256' },
      { x: 358, w: 52, h: 52, depth: 22, lvl: 2, trans: true, shape: '40² · 128' },
      { x: 444, w: 52, h: 52, depth: 40, lvl: 3, name: 'Dense 2', sub: '×12', front: '×12', shape: '40² · 512' },
      { x: 548, w: 42, h: 42, depth: 30, lvl: 3, trans: true, shape: '20² · 256' },
      { x: 632, w: 42, h: 42, depth: 50, lvl: 4, name: 'Dense 3', sub: '×24', front: '×24', shape: '20² · 1024' },
      { x: 736, w: 34, h: 34, depth: 40, lvl: 4, trans: true, shape: '10² · 512' },
      { x: 822, w: 34, h: 34, depth: 50, lvl: 5, name: 'Dense 4', sub: '×16', front: '×16', shape: '10² · 1024' },
    ];

    s += f.bracketT(stages[1].x, stages[8].x + stages[8].w, 70,
      'DenseNet-121   ·   ImageNet-pretrained   ·   dense connectivity   ·   ~7.9 M params',
      { rise: 9, size: 14, weight: 600, labelFill: C.ink2, color: C.hair });

    stages.forEach(st => {
      const y = baseline - st.h;
      s += f.slab(st.x, y, st.w, st.h, st.depth, st.lvl, {
        front: st.front, frontSize: 14, frontFill: st.lvl >= 4 ? '#fff' : C.ink,
        frontStripes: st.img, clip: 'clD3_' + st.x,
      });
      if (!st.trans) {
        s += f.T(st.x + st.w / 2, 116, st.shape, { size: 12, mono: true, fill: C.mut, weight: 500 });
        s += f.T(st.x + st.w / 2, baseline + 30, st.name, { size: 13.5, weight: 700, fill: C.ink });
      } else {
        s += f.T(st.x + st.w / 2 + 1, baseline + 30, '↓2', { size: 12, mono: true, fill: C.coral, weight: 600 });
      }
    });
    for (let i = 0; i < stages.length - 1; i++) {
      const a = stages[i], b = stages[i + 1];
      const gap = b.x - (a.x + a.w);
      if (gap > 26) s += f.arrow(a.x + a.w + 4, aY, b.x - 6, aY, { color: C.ink2, width: 1.9 });
      else s += `<line x1="${a.x + a.w}" y1="${aY}" x2="${b.x}" y2="${aY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    }

    // adaptive pool
    const px = 918, pw = 120, ph = 92, py = baseline - ph;
    s += f.arrow(stages[8].x + stages[8].w + 4, aY, px - 6, py + ph / 2, { color: C.ink2, width: 1.9 });
    s += f.box(px, py, pw, ph, { stroke: C.hair, fill: '#fbfcfe', r: 12 });
    s += f.lines(px + pw / 2, py + 36, [
      { t: 'Adaptive', size: 15, weight: 700, fill: C.ink },
      { t: 'avg pool', size: 15, weight: 700, fill: C.ink },
    ], { lh: 20 });
    s += f.T(px + pw / 2, py + ph - 12, '1024-d', { size: 12.5, mono: true, fill: C.mut, weight: 600 });

    // feature layer
    const flx = 1058, flw = 176, flh = 104, fly = baseline - flh;
    s += f.arrow(px + pw + 4, py + ph / 2, flx - 6, fly + flh / 2, { color: C.ink2, width: 1.9 });
    s += f.box(flx, fly, flw, flh, { stroke: C.hair, accent: C.blue, fill: '#fafcff', r: 12 });
    s += f.T(flx + flw / 2 + 4, fly + 34, 'Feature layer', { size: 16, weight: 700, fill: C.ink });
    s += f.lines(flx + flw / 2 + 4, fly + 60, [
      { t: 'Linear 1024 → 512', size: 13, mono: true, fill: C.ink2 },
      { t: 'ReLU · Dropout 0.3', size: 13, mono: true, fill: C.mut },
    ], { lh: 21 });

    s += funnelToHeads({
      cx: 650, vcy: 348, srcCx: flx + flw / 2, srcBottom: baseline, railY: 348,
      splitY: 426, cardY: 444, cardW: 464, mlCx: 352, binCx: 948,
      railLabel: 'shared 512-d features', railLabelW: 196,
    });

    f.mount(id, W, H, s, { bg: false });
  };

  /* ---------------------------------------------------------
     D4 · Two-phase fine-tuning timeline  (slide 8 · 3.36)
     --------------------------------------------------------- */
  global.drawDeckFinetune = function (id) {
    let s = '';
    const x0 = 250, x1 = 1620, EMAX = 60;
    const X = e => x0 + (e / EMAX) * (x1 - x0);
    const unfreezeX = X(6), bestX = X(18);
    const top = 70;

    s += `<rect x="${x0}" y="${top + 26}" width="${unfreezeX - x0}" height="350" fill="#fbf3ef"/>`;
    s += `<rect x="${unfreezeX}" y="${top + 26}" width="${x1 - unfreezeX}" height="350" fill="#f7fafe"/>`;

    s += f.bracketT(x0 + 2, unfreezeX - 2, top + 24, 'PHASE 1 — warm up the heads', { rise: 10, size: 14, weight: 700, upper: true, ls: 0.8, color: C.hair, labelFill: C.coralLine });
    s += f.bracketT(unfreezeX + 2, x1 - 2, top + 24, 'PHASE 2 — end-to-end fine-tuning', { rise: 10, size: 14, weight: 700, upper: true, ls: 0.8, color: C.hair, labelFill: C.blueDk });

    s += `<line x1="${unfreezeX}" y1="${top + 30}" x2="${unfreezeX}" y2="412" stroke="${C.coral}" stroke-width="1.7" stroke-dasharray="5 5"/>`;
    s += `<rect x="${unfreezeX - 124}" y="${top + 34}" width="248" height="44" rx="9" fill="#fff" stroke="${C.coral}" stroke-width="1.4"/>`;
    s += f.T(unfreezeX, top + 52, 'epoch 6', { size: 14, weight: 700, fill: C.coralLine, mono: true });
    s += f.T(unfreezeX, top + 70, 'unfreeze_backbone()', { size: 14, mono: true, fill: C.ink2 });

    s += `<line x1="${bestX}" y1="${top + 96}" x2="${bestX}" y2="412" stroke="${C.ink2}" stroke-width="1.3" stroke-dasharray="2 4"/>`;
    s += `<path d="M${bestX} ${top + 98} l 10 10 l -10 10 l -10 -10 z" fill="${C.coral}"/>`;
    s += f.T(bestX, top + 94, 'best val checkpoint · epoch 18', { size: 13, weight: 600, fill: C.ink });

    const laneH = 66;
    const bbY = top + 130, hdY = top + 214;

    const hatch = (x, y, w, h, col) => {
      let g = `<clipPath id="dfrz"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/></clipPath><g clip-path="url(#dfrz)">`;
      for (let i = -h; i < w; i += 9) g += `<line x1="${x + i}" y1="${y}" x2="${x + i + h}" y2="${y + h}" stroke="${col}" stroke-width="1" opacity="0.55"/>`;
      return g + '</g>';
    };

    s += f.lines(x0 - 24, bbY + 26, [
      { t: 'Backbone', size: 16, weight: 700, fill: C.ink, anchor: 'end' },
      { t: 'DenseNet-121', size: 12.5, mono: true, fill: C.mut, anchor: 'end' },
    ], { lh: 20, anchor: 'end' });
    s += f.lines(x0 - 24, hdY + 24, [
      { t: 'Feature layer', size: 16, weight: 700, fill: C.ink, anchor: 'end' },
      { t: '+ two heads', size: 12.5, mono: true, fill: C.mut, anchor: 'end' },
    ], { lh: 20, anchor: 'end' });

    const fw = unfreezeX - x0;
    s += `<rect x="${x0}" y="${bbY}" width="${fw}" height="${laneH}" rx="9" fill="#eef0f3" stroke="${C.mut2}" stroke-width="1.2"/>`;
    s += hatch(x0, bbY, fw, laneH, C.mut2);
    s += f.T(x0 + fw / 2, bbY + laneH / 2 + 5, 'FROZEN', { size: 13, weight: 700, upper: true, ls: 1, fill: C.mut });
    const tw = x1 - unfreezeX;
    s += `<rect x="${unfreezeX}" y="${bbY}" width="${tw}" height="${laneH}" rx="9" fill="${RAMP[2].f}" stroke="${RAMP[2].s}" stroke-width="1.2"/>`;
    s += f.T(unfreezeX + tw / 2, bbY + laneH / 2 + 5, 'trainable — backbone specializes to chest-X-ray features', { size: 14.5, weight: 600, fill: C.blueDk });

    const hw2 = x1 - x0;
    s += `<rect x="${x0}" y="${hdY}" width="${hw2}" height="${laneH}" rx="9" fill="#fdeee7" stroke="${C.coral}" stroke-width="1.2"/>`;
    s += `<line x1="${unfreezeX}" y1="${hdY}" x2="${unfreezeX}" y2="${hdY + laneH}" stroke="${C.coral}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>`;
    s += f.T(x0 + fw / 2, hdY + laneH / 2 + 5, 'trainable', { size: 13.5, weight: 600, fill: C.coralLine });
    s += f.T(unfreezeX + tw / 2, hdY + laneH / 2 + 5, 'trainable — newly-initialized weights adapt throughout', { size: 14.5, weight: 600, fill: C.coralLine });

    const lrTop = top + 300, lrZero = top + 372;
    const lrY = v => lrZero - (v / 1e-3) * (lrZero - lrTop);
    s += `<line x1="${x0}" y1="${lrZero}" x2="${x1}" y2="${lrZero}" stroke="${C.hair}" stroke-width="1"/>`;
    [['1e-3', 1e-3], ['1e-4', 1e-4]].forEach(([lab, v]) => {
      const y = lrY(v);
      s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${C.hairsoft}" stroke-width="1" stroke-dasharray="3 4"/>`;
      s += f.T(x0 - 14, y + 4, lab, { size: 12, mono: true, fill: C.mut, anchor: 'end' });
    });
    s += `<line x1="${x0}" y1="${lrY(0)}" x2="${x1}" y2="${lrY(0)}" stroke="${C.hairsoft}" stroke-width="1" stroke-dasharray="3 4"/>`;
    s += f.T(x0 - 14, lrTop - 16, 'learning rate', { size: 12.5, weight: 600, fill: C.ink2, anchor: 'end' });

    let d = `M${x0} ${lrY(1e-3)} L${unfreezeX} ${lrY(1e-3)} L${unfreezeX} ${lrY(1e-4)}`;
    const T2 = EMAX - 6;
    for (let e = 6; e <= EMAX; e += 1) {
      const prog = (e - 6) / T2;
      const lr = 0.5 * 1e-4 * (1 + Math.cos(Math.PI * prog));
      d += ` L${X(e)} ${lrY(lr)}`;
    }
    s += `<path d="${d}" fill="none" stroke="${C.blue}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>`;
    s += `<rect x="${(x0 + unfreezeX) / 2 - 54}" y="${lrY(1e-3) - 30}" width="108" height="23" rx="11" fill="#fff" stroke="${C.hair}" stroke-width="1"/>`;
    s += f.T((x0 + unfreezeX) / 2, lrY(1e-3) - 14, 'lr = 1e-3', { size: 13, mono: true, weight: 600, fill: C.blueDk });
    s += f.T(unfreezeX + 160, lrY(1e-4) - 13, 'lr = 1e-4, cosine anneal → 0', { size: 13, mono: true, weight: 600, fill: C.blueDk, anchor: 'start' });

    const axY = lrZero;
    [1, 5, 6, 10, 18, 20, 30, 40, 50, 60].forEach(e => {
      const x = X(e);
      s += `<line x1="${x}" y1="${axY}" x2="${x}" y2="${axY + 6}" stroke="${C.mut}" stroke-width="1.1"/>`;
      s += f.T(x, axY + 23, String(e), { size: 12, mono: true, fill: (e === 6 ? C.coralLine : e === 18 ? C.ink : C.mut), weight: (e === 6 || e === 18) ? 700 : 400 });
    });
    s += f.T((x0 + x1) / 2, axY + 45, 'epoch', { size: 12.5, fill: C.mut, weight: 600, upper: true, ls: 1 });

    f.mount(id, 1680, top + 430, s, { bg: false });
  };

  /* ---------------------------------------------------------
     D5 · Training pipeline (vertical)  (slide 9 · 0.46)
     --------------------------------------------------------- */
  global.drawDeckPipeline = function (id) {
    let s = '';
    const W = 442;
    const cx = W / 2;
    const bw = 322, bx = cx - bw / 2;

    function nodeV(y, h, o) {
      o = o || {};
      let r = f.box(bx, y, bw, h, { stroke: C.hair, fill: o.fill || '#ffffff', accent: o.accent, r: 11 });
      const tcx = cx + (o.accent ? 5 : 0);
      r += f.T(tcx, y + 28, o.title, { size: 15.5, weight: 700, fill: o.titleFill || C.ink });
      if (o.lines) r += f.lines(tcx, y + 50, o.lines.map(t => ({ t })), { lh: 19, size: 12.5, mono: true, fill: o.lineFill || C.ink2 });
      return r;
    }
    const down = (y1, y2, o) => f.arrow(cx, y1, cx, y2, Object.assign({ color: C.ink2, width: 1.8 }, o || {}));

    const cyY = 16, cyH = 96;
    s += f.cylinder(bx, cyY, bw, cyH, { fill: RAMP[1].f, stroke: RAMP[1].s, topFill: RAMP[1].t, ry: 13 });
    s += f.T(cx, cyY + 44, 'chest-xray-14-320', { size: 13.5, weight: 700, mono: true, fill: C.blueDk });
    s += f.lines(cx, cyY + 66, [
      { t: '112,120 images · 36 shards' },
      { t: '~7.97 GB · pinned revision' },
    ], { lh: 18, size: 11.5, mono: true, fill: C.ink2 });

    let y = cyY + cyH + 30;
    s += down(cyY + cyH, y);
    s += nodeV(y, 76, { title: 'Deterministic split', lines: ['train 78,468 · val 11,210', 'test 22,442'] });

    y += 76 + 30; let py = y - 30;
    s += down(py, y);
    s += nodeV(y, 92, { title: 'Augmentation pipeline', lines: ['CLAHE · h-flip · rotate ±15°', 'affine · jitter · blur · erasing'] });

    y += 92 + 30; py = y - 30;
    s += down(py, y);
    s += nodeV(y, 64, { title: 'Dual-head forward', accent: C.blue, lines: ['AMP fp16 · single pass'] });
    const fwdBottom = y + 64;

    const splitY = fwdBottom + 26;
    s += `<line x1="${cx}" y1="${fwdBottom}" x2="${cx}" y2="${splitY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    const lcx = bx + 80, rcx = bx + bw - 80;
    s += `<line x1="${lcx}" y1="${splitY}" x2="${rcx}" y2="${splitY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    const headY = splitY + 18, headH = 76, hbw = 146;
    s += f.arrow(lcx, splitY, lcx, headY, { color: C.ink2, width: 1.8 });
    s += f.arrow(rcx, splitY, rcx, headY, { color: C.ink2, width: 1.8 });
    s += f.box(lcx - hbw / 2, headY, hbw, headH, { stroke: C.coral, fill: '#fffcfa', r: 10 });
    s += f.T(lcx, headY + 24, 'multilabel', { size: 13, weight: 700, fill: C.ink });
    s += f.lines(lcx, headY + 44, [{ t: 'B × 14 logits' }, { t: 'weighted BCE' }], { lh: 17, size: 11, mono: true, fill: C.mut });
    s += f.box(rcx - hbw / 2, headY, hbw, headH, { stroke: C.blue, fill: '#fafcff', r: 10 });
    s += f.T(rcx, headY + 24, 'binary', { size: 13, weight: 700, fill: C.ink });
    s += f.lines(rcx, headY + 44, [{ t: 'B × 1 logit' }, { t: 'BCE · Norm/Abn' }], { lh: 17, size: 11, mono: true, fill: C.mut });

    const headBottom = headY + headH;
    const mergeY = headBottom + 24;
    s += `<line x1="${lcx}" y1="${headBottom}" x2="${lcx}" y2="${mergeY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    s += `<line x1="${rcx}" y1="${headBottom}" x2="${rcx}" y2="${mergeY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    s += `<line x1="${lcx}" y1="${mergeY}" x2="${rcx}" y2="${mergeY}" stroke="${C.ink2}" stroke-width="1.8"/>`;
    y = mergeY + 18;
    s += f.arrow(cx, mergeY, cx, y, { color: C.ink2, width: 1.8 });
    s += nodeV(y, 64, { title: 'Combined loss', lines: ['1.0 × L_ml  +  0.5 × L_bin'] });

    y += 64 + 28; py = y - 28;
    s += down(py, y);
    s += nodeV(y, 64, { title: 'Backward · grad clip 1.0', lines: ['grad accum ×4 · eff. batch 96'] });

    y += 64 + 28; py = y - 28;
    s += down(py, y);
    s += nodeV(y, 64, { title: 'AdamW · CosineAnnealingLR', lines: ['early stop · patience 15'] });

    y += 64 + 34; py = y - 34;
    s += down(py, y, { color: C.coral });
    s += f.T(cx + 8, py + 24, '↑ best val macro AUC-ROC', { size: 10.5, mono: true, fill: C.mut, anchor: 'start' });
    const tH = 66;
    s += `<rect x="${bx}" y="${y}" width="${bw}" height="${tH}" rx="11" fill="#20242c"/>`;
    s += f.T(cx, y + 28, 'Best checkpoint → HF Hub', { size: 15, weight: 700, fill: '#f0a07f' });
    s += f.T(cx, y + 50, 'model · config · history.json', { size: 12, mono: true, fill: '#cdd4e0' });

    const finalH = y + tH + 16;
    const VW = Math.round(finalH * 0.459);
    s = `<g transform="translate(${(VW - W) / 2},0)">${s}</g>`;
    f.mount(id, VW, finalH, s, { bg: false });
  };

})(window);
