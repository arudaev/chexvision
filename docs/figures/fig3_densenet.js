/* Figure 3 — Model 2: DenseNet-121 transfer learning */
(function () {
  const W = 1680, H = 772;

  window.drawFig3 = function (id) {
    const f = FL, C = f.C;
    let s = '';
    const baseline = 300, aY = 282;

    // input + stem + 4 dense blocks + 3 transitions.
    // Square front faces: side ∝ spatial resolution (320→80→40→20→10); each
    // dense block shares its resolution-stage side with the transition out of
    // it. Transitions are full output-size squares (not thin plates) so the
    // spatial halving reads; depth ∝ channels, so transitions get shallower.
    const stages = [
      { x: 70,  w: 86, h: 86, depth: 6,  lvl: 0, name: 'Input', sub: '', shape: '320 × 320 × 3', img: true },
      { x: 168, w: 64, h: 64, depth: 12, lvl: 1, name: 'Stem',  sub: '7×7 s2·pool', shape: '80² · 64' },
      { x: 250, w: 64, h: 64, depth: 28, lvl: 2, name: 'Dense block 1', sub: '×6',  shape: '80² · 256', front: '×6', dense: true },
      { x: 348, w: 50, h: 50, depth: 18, lvl: 2, name: 'Trans', sub: '↓2', shape: '40² · 128', trans: true },
      { x: 422, w: 50, h: 50, depth: 36, lvl: 3, name: 'Dense block 2', sub: '×12', shape: '40² · 512', front: '×12', dense: true },
      { x: 514, w: 40, h: 40, depth: 26, lvl: 3, name: 'Trans', sub: '↓2', shape: '20² · 256', trans: true },
      { x: 586, w: 40, h: 40, depth: 46, lvl: 4, name: 'Dense block 3', sub: '×24', shape: '20² · 1024', front: '×24', dense: true },
      { x: 678, w: 32, h: 32, depth: 34, lvl: 4, name: 'Trans', sub: '↓2', shape: '10² · 512', trans: true },
      { x: 750, w: 32, h: 32, depth: 44, lvl: 5, name: 'Dense block 4', sub: '×16', shape: '10² · 1024', front: '×16', dense: true },
    ];

    // backbone bracket
    s += f.bracketT(stages[1].x, stages[8].x + stages[8].w, 128,
      'DenseNet-121 backbone   ·   ImageNet-pretrained   ·   dense connectivity   ·   ~7.9 M params',
      { rise: 9, size: 13, weight: 600, labelFill: C.ink2, color: C.hair });

    stages.forEach(st => {
      const y = baseline - st.h;
      s += f.slab(st.x, y, st.w, st.h, st.depth, st.lvl, {
        front: st.front, frontSize: 15, frontFill: st.lvl >= 4 ? '#fff' : C.ink,
        frontStripes: st.img, clip: 'cl3_' + st.x,
      });
      // shape label only on input + the four dense blocks (transitions would collide)
      if (!st.trans) s += f.T(st.x + st.w / 2, 150, st.shape, { size: 12, mono: true, fill: C.mut, weight: 500 });
      if (!st.trans) {
        s += f.T(st.x + st.w / 2, baseline + 34, st.name, { size: 14.5, weight: 700, fill: C.ink });
        if (st.sub) s += f.T(st.x + st.w / 2, baseline + 53, st.sub, { size: 12.5, mono: true, fill: C.mut });
      } else {
        s += f.T(st.x + st.w / 2 + 2, baseline + 34, 'T', { size: 13, weight: 700, fill: C.mut });
        s += f.T(st.x + st.w / 2 + 2, baseline + 51, '↓2', { size: 11.5, mono: true, fill: C.coral, weight: 600 });
      }
    });

    // arrows between slabs (short)
    for (let i = 0; i < stages.length - 1; i++) {
      const a = stages[i], b = stages[i + 1];
      const gap = b.x - (a.x + a.w);
      if (gap > 26) s += f.arrow(a.x + a.w + 4, aY, b.x - 6, aY, { color: C.ink2, width: 1.8 });
      else s += `<line x1="${a.x + a.w}" y1="${aY}" x2="${b.x}" y2="${aY}" stroke="${C.ink2}" stroke-width="1.7"/>`;
    }

    // ---- Adaptive avg pool ----
    const last = stages[8];
    const px = 836, py = 214, pw = 126, ph = 92;
    s += f.arrow(last.x + last.w + 4, aY, px - 6, 260, { color: C.ink2, width: 1.8 });
    s += f.box(px, py, pw, ph, { stroke: C.hair, fill: '#fbfcfe' });
    s += f.lines(px + pw / 2, py + 34, [
      { t: 'Adaptive', size: 14.5, weight: 700, fill: C.ink },
      { t: 'avg pool', size: 14.5, weight: 700, fill: C.ink },
    ], { lh: 19 });
    s += f.T(px + pw / 2, py + ph - 12, '1024-d', { size: 12, mono: true, fill: C.mut, weight: 600 });

    // ---- Feature layer (shared) ----
    const fx = 1010, fy = 206, fw = 176, fh = 108;
    s += f.arrow(px + pw + 4, 260, fx - 6, 260, { color: C.ink2, width: 1.8 });
    s += f.box(fx, fy, fw, fh, { stroke: C.hair, accent: C.blue, fill: '#fafcff' });
    s += f.T(fx + fw / 2 + 4, fy + 30, 'Feature layer', { size: 15.5, weight: 700, fill: C.ink });
    s += f.lines(fx + fw / 2 + 4, fy + 56, [
      { t: 'Linear 1024 → 512', size: 12.5, mono: true, fill: C.ink2 },
      { t: 'ReLU · Dropout 0.3', size: 12.5, mono: true, fill: C.mut },
    ], { lh: 20 });

    // ---- 512-d vector + heads ----
    const vx = 1212, vcy = 260;
    s += f.arrow(fx + fw + 4, 260, vx - 8, 260, { color: C.ink2, width: 1.8 });
    const vec = f.vector(vx, vcy, { label: '512-d', sub: 'shared', lvl: 2 });
    s += vec.svg;
    // heads sit to the right but width is tight; place compactly
    // (re-use dualHeads with narrower head + closer split)
    s += denseHeads(f, C, vec.right, vcy);

    // ============ INSET: dense connectivity ============
    const iy = 432;
    s += `<line x1="70" y1="${iy}" x2="${W - 70}" y2="${iy}" stroke="${C.hairsoft}" stroke-width="1.4"/>`;
    s += f.T(72, iy + 26, 'DENSE CONNECTIVITY', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.6, fill: C.ink });
    s += f.T(290, iy + 26, '— within a block, every layer receives the concatenated feature maps of all preceding layers (growth k = 32)', { anchor: 'start', size: 12.5, fill: C.mut });

    // layer nodes along a baseline; concat skip arcs above
    const ly = 690;                       // layer-node row
    const lxs = [120, 320, 520, 720, 920, 1120];
    const labels = ['x₀\ninput', 'H₁', 'H₂', 'H₃', 'H₄', 'BN·ReLU\npool'];
    const nodeR = 26;

    // concatenation arcs: each node i feeds all nodes j>i (DenseNet signature)
    for (let i = 0; i < lxs.length - 1; i++) {
      for (let j = i + 1; j < lxs.length; j++) {
        const x1 = lxs[i], x2 = lxs[j];
        const span = j - i;
        const lift = ly - 36 - span * 24;     // higher arc for longer reach
        const isLong = span > 1;
        s += `<path d="M${x1} ${ly - nodeR} C ${x1} ${lift}, ${x2} ${lift}, ${x2} ${ly - nodeR}" `
          + `fill="none" stroke="${isLong ? C.blue : C.ink2}" stroke-width="${isLong ? 1.2 : 1.5}" `
          + `opacity="${isLong ? 0.5 : 0.9}"/>`;
        // arrowhead at destination
        s += `<path d="M${x2} ${ly - nodeR} l -4 -7 l 8 0 z" fill="${isLong ? C.blue : C.ink2}" opacity="${isLong ? 0.6 : 0.95}"/>`;
      }
    }

    // nodes on top of arcs
    lxs.forEach((x, i) => {
      const isH = i > 0 && i < lxs.length - 1;
      const fill = i === 0 ? '#eef1f5' : (i === lxs.length - 1 ? '#fbfcfe' : FL.RAMP[2].f);
      const stroke = i === 0 ? C.hair : (i === lxs.length - 1 ? C.hair : FL.RAMP[2].s);
      if (i === 0 || i === lxs.length - 1) {
        s += `<rect x="${x - 46}" y="${ly - 24}" width="92" height="48" rx="9" fill="${fill}" stroke="${stroke}" stroke-width="1.3"/>`;
        const parts = labels[i].split('\n');
        s += f.lines(x, ly - (parts.length > 1 ? 3 : -5), parts.map(t => ({ t })), { lh: 16, size: 12, mono: true, fill: C.ink2, weight: 500 });
      } else {
        s += `<circle cx="${x}" cy="${ly}" r="${nodeR}" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>`;
        s += f.T(x, ly + 5.5, labels[i], { size: 16, weight: 700, fill: C.ink });
      }
    });
    // forward arrows along the row (between adjacent)
    for (let i = 0; i < lxs.length - 1; i++) {
      const x1 = lxs[i] + (i === 0 ? 46 : nodeR);
      const x2 = lxs[i + 1] - (i + 1 === lxs.length - 1 ? 46 : nodeR);
      s += f.arrow(x1, ly, x2, ly, { color: C.ink2, width: 1.6 });
    }
    // legend
    s += f.T(1232, ly - 18, 'H', { anchor: 'start', size: 13, weight: 700, fill: C.ink });
    s += f.T(1248, ly - 18, ' = BN → ReLU → 3×3 Conv', { anchor: 'start', size: 12, mono: true, fill: C.mut });
    s += `<path d="M1232 ${ly + 4} q 24 -16 48 0" fill="none" stroke="${C.blue}" stroke-width="1.2" opacity="0.6"/>`;
    s += f.T(1288, ly + 8, 'feature-map concatenation', { anchor: 'start', size: 12, fill: C.mut });

    f.mount(id, W, H, s);
  };

  // heads variant tuned to the tighter right margin of fig3
  function denseHeads(f, C, vecRight, vcy) {
    const hx = 1320, hw = 296, mlY = 188, binY = 388, splitX = 1284;
    let s = '';
    s += f.poly([[vecRight, vcy], [splitX, vcy], [splitX, mlY], [hx - 6, mlY]], { color: C.ink2, width: 1.8 });
    s += f.poly([[vecRight, vcy], [splitX, vcy], [splitX, binY], [hx - 6, binY]], { color: C.ink2, width: 1.8 });
    const mlH = 132;
    s += f.box(hx, mlY - mlH / 2, hw, mlH, { stroke: C.hair, accent: C.coral, fill: '#fffcfa', r: 11 });
    s += f.T(hx + 22, mlY - 40, 'Multi-label head', { anchor: 'start', size: 16.5, weight: 700, fill: C.ink });
    s += f.T(hx + 22, mlY - 16, 'Linear 512 → 14 · sigmoid', { anchor: 'start', size: 12.5, mono: true, fill: C.ink2 });
    s += f.T(hx + 22, mlY + 4, 'weighted BCE · pos_weight', { anchor: 'start', size: 11.5, mono: true, fill: C.mut });
    for (let i = 0; i < 14; i++) {
      const tx = hx + 22 + i * 18.4;
      s += `<rect x="${tx}" y="${mlY + 22}" width="11.5" height="20" rx="2.5" fill="${i % 4 === 0 ? C.coral : '#f2d9cd'}" stroke="${C.coral}" stroke-width="0.7" opacity="${i % 4 === 0 ? 0.92 : 0.6}"/>`;
    }
    s += f.T(hx + hw - 22, mlY - 40, 'primary', { anchor: 'end', size: 10.5, mono: true, upper: true, ls: 0.4, fill: C.coral, weight: 600 });
    const binH = 104;
    s += f.box(hx, binY - binH / 2, hw, binH, { stroke: C.hair, accent: C.blue, fill: '#fafcff', r: 11 });
    s += f.T(hx + 22, binY - 22, 'Binary head', { anchor: 'start', size: 16.5, weight: 700, fill: C.ink });
    s += f.T(hx + 22, binY + 2, 'Linear 512 → 1 · sigmoid', { anchor: 'start', size: 12.5, mono: true, fill: C.ink2 });
    s += f.T(hx + 22, binY + 24, 'Normal vs. Abnormal', { anchor: 'start', size: 11.5, mono: true, fill: C.mut });
    s += f.T(hx + hw - 22, binY - 22, 'aux', { anchor: 'end', size: 10.5, mono: true, upper: true, ls: 0.4, fill: C.blue, weight: 600 });
    return s;
  }
})();
