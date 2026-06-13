/* Figure 2 — Model 1: Custom SE-ResNet (from scratch) */
(function () {
  const W = 1680, H = 772;

  window.drawFig2 = function (id) {
    const f = FL, C = f.C;
    let s = '';
    const baseline = 300;          // slab bottoms align here
    const aY = 262;                // level arrow height (inside every front face)

    // ---- stage configs (square spatial faces, channels grow as depth) ----
    // Front face (w×h) is the SPATIAL map → square; side ∝ resolution
    // (320→80→40→20→10). Stem & Stage 1 are both 80² → identical squares.
    // depth ∝ channel count; the 3-channel input is a very thin slab.
    const stages = [
      { x: 70,  w: 120, h: 120, depth: 8,  lvl: 0, name: 'Input',   sub: '',        shape: '320 × 320 × 3', img: true },
      { x: 216, w: 88,  h: 88,  depth: 16, lvl: 1, name: 'Stem',    sub: '7×7 s2 · pool', shape: '80² · 64', front: '' },
      { x: 338, w: 88,  h: 88,  depth: 16, lvl: 1, name: 'Stage 1', sub: '×3',       shape: '80² · 64',  front: '×3' },
      { x: 460, w: 70,  h: 70,  depth: 26, lvl: 2, name: 'Stage 2', sub: '×4',       shape: '40² · 128', front: '×4', ds: true },
      { x: 574, w: 54,  h: 54,  depth: 38, lvl: 3, name: 'Stage 3', sub: '×6',       shape: '20² · 256', front: '×6', ds: true },
      { x: 684, w: 42,  h: 42,  depth: 50, lvl: 4, name: 'Stage 4', sub: '×3',       shape: '10² · 512', front: '×3', ds: true },
    ];

    // top bracket over the four residual stages
    s += f.bracketT(stages[2].x, stages[5].x + stages[5].w, 132,
      'Four residual stages   ·   depths [3, 4, 6, 3]   ·   widths 64 → 512   ·   ~23 M params, no pretraining',
      { rise: 9, size: 13, weight: 600, labelFill: C.ink2, color: C.hair });

    stages.forEach(st => {
      const y = baseline - st.h;
      s += f.slab(st.x, y, st.w, st.h, st.depth, st.lvl, {
        front: st.front, frontSize: 16, frontFill: st.lvl >= 4 ? '#fff' : C.ink,
        frontStripes: st.img, clip: 'cl2_' + st.x,
      });
      // tensor shape label, fixed line above all oblique tops
      s += f.T(st.x + st.w / 2, 152, st.shape, { size: 12.5, mono: true, fill: C.mut, weight: 500 });
      // stage name + sub below baseline
      s += f.T(st.x + st.w / 2, baseline + 34, st.name, { size: 15.5, weight: 700, fill: C.ink });
      if (st.sub) s += f.T(st.x + st.w / 2, baseline + 54, st.sub, { size: 12.5, mono: true, fill: C.mut });
    });

    // arrows between slabs
    for (let i = 0; i < stages.length - 1; i++) {
      const a = stages[i], b = stages[i + 1];
      s += f.arrow(a.x + a.w + 4, aY, b.x - 6, aY, { color: C.ink2, width: 1.8 });
      if (b.ds) s += f.T((a.x + a.w + b.x) / 2, aY - 9, '↓2', { size: 12.5, mono: true, fill: C.coral, weight: 600 });
    }

    // ---- Global Avg Pool node ----
    const last = stages[5];
    const gx = 800, gy = 214, gw = 132, gh = 92;
    s += f.arrow(last.x + last.w + 4, aY, gx - 6, 260, { color: C.ink2, width: 1.8 });
    s += f.box(gx, gy, gw, gh, { stroke: C.hair, fill: '#fbfcfe' });
    s += f.lines(gx + gw / 2, gy + 36, [
      { t: 'Global avg pool', size: 14.5, weight: 700, fill: C.ink },
      { t: 'Dropout 0.5', size: 12.5, mono: true, fill: C.mut },
    ], { lh: 22 });

    // ---- 512-d feature vector glyph ----
    const vx = 974, vcy = 260;
    s += f.arrow(gx + gw + 4, 260, vx - 8, 260, { color: C.ink2, width: 1.8 });
    const vec = f.vector(vx, vcy, { label: '512-d', sub: 'shared', lvl: 2 });
    s += vec.svg;

    // ---- branch to two heads ----
    s += f.dualHeads({ vecRight: vec.right, vcy: vcy, splitX: 1070, hx: 1156, hw: 444, mlY: 188, binY: 388 });

    // ============ INSET: SE residual block ============
    const iy = 432;
    s += `<line x1="70" y1="${iy}" x2="${W - 70}" y2="${iy}" stroke="${C.hairsoft}" stroke-width="1.4"/>`;
    s += f.T(72, iy + 26, 'SE RESIDUAL BLOCK', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.6, fill: C.ink });
    s += f.T(290, iy + 26, '— the per-block unit; channel attention recalibrates features after every convolution', { anchor: 'start', size: 12.5, fill: C.mut });

    const lane = 660;           // main lane center y
    const bh = 56, by = lane - bh / 2;
    const localBox = (x, w, ttl, opts) => {
      opts = opts || {};
      let r = f.box(x, by, w, bh, { stroke: opts.stroke || C.hair, fill: opts.fill || '#fff', r: 9 });
      r += f.lines(x + w / 2, lane - (ttl.length > 1 ? 5 : -5), ttl.map(t => ({ t })), { lh: 17, size: 13, mono: true, fill: C.ink2, weight: 500 });
      return r;
    };

    // input node
    s += `<rect x="72" y="${by}" width="62" height="${bh}" rx="9" fill="#f4f6f9" stroke="${C.hair}" stroke-width="1.3"/>`;
    s += f.T(103, lane + 6, 'x', { size: 22, weight: 600, fill: C.ink, mono: false });
    s += f.T(103, by + bh + 18, 'block in', { size: 11, fill: C.mut });

    s += f.arrow(134, lane, 184, lane, { color: C.ink2, width: 1.7 });
    s += localBox(186, 156, ['3×3 Conv · BN · ReLU']);
    s += f.arrow(342, lane, 372, lane, { color: C.ink2, width: 1.7 });
    s += localBox(374, 140, ['3×3 Conv · BN']);
    // U signal along main lane
    s += `<line x1="514" y1="${lane}" x2="1196" y2="${lane}" stroke="${C.ink2}" stroke-width="1.7"/>`;
    s += f.T(560, lane - 9, 'U', { size: 14, weight: 700, fill: C.blueDk });
    s += f.T(560, lane + 18, 'feature maps', { size: 10.5, fill: C.mut });

    // SE branch
    const seY = 524, sbh = 44, sby = seY - sbh / 2;
    const seBox = (x, w, t) => f.box(x, sby, w, sbh, { stroke: C.hair, fill: '#fbfcfe', r: 8 })
      + f.T(x + w / 2, seY + 4.5, t, { size: 12, mono: true, fill: C.ink2, weight: 500 });
    // tap up from after conv2
    s += f.poly([[514, lane - 12], [514, seY], [536, seY]], { color: C.blue, width: 1.6 });
    s += seBox(538, 120, 'global avg pool');
    s += f.arrow(658, seY, 690, seY, { color: C.blue, width: 1.5 });
    s += seBox(692, 122, 'FC  C → C/16');
    s += f.arrow(814, seY, 842, seY, { color: C.blue, width: 1.5 });
    s += seBox(844, 64, 'ReLU');
    s += f.arrow(908, seY, 936, seY, { color: C.blue, width: 1.5 });
    s += seBox(938, 122, 'FC  C/16 → C');
    s += f.arrow(1060, seY, 1088, seY, { color: C.blue, width: 1.5 });
    s += seBox(1090, 86, 'sigmoid');
    // squeeze / excitation brackets
    s += f.bracketT(538, 658, sby - 4, 'squeeze', { rise: 7, size: 10.5, mono: true, upper: true, ls: 0.6, color: C.hair, labelFill: C.mut, weight: 600 });
    s += f.bracketT(692, 1176, sby - 4, 'excitation', { rise: 7, size: 10.5, mono: true, upper: true, ls: 0.6, color: C.hair, labelFill: C.mut, weight: 600 });
    // down into scale
    s += f.poly([[1176, seY], [1204, seY], [1204, lane - 12]], { color: C.blue, width: 1.6 });
    s += f.T(1232, seY - 2, 'channel', { anchor: 'start', size: 11, fill: C.blueDk, weight: 600 });
    s += f.T(1232, seY + 13, 'weights s', { anchor: 'start', size: 11, fill: C.blueDk, weight: 600 });

    // scale (multiply) and add
    const drawOp = (cx, sym, col) => `<circle cx="${cx}" cy="${lane}" r="18" fill="#fff" stroke="${col}" stroke-width="1.6"/>`
      + f.T(cx, lane + (sym === '+' ? 8 : 6.5), sym, { size: sym === '+' ? 26 : 20, weight: 600, fill: col });
    s += drawOp(1204, '×', C.blue);
    s += f.arrow(1222, lane, 1284, lane, { color: C.ink2, width: 1.7 });
    s += drawOp(1302, '+', C.coral);
    // identity skip
    s += f.poly([[103, by], [103, 478], [1302, 478], [1302, lane - 18]], { color: C.coral, width: 1.6 });
    s += `<rect x="560" y="469" width="180" height="17" fill="#fffcfa"/>`;
    s += f.T(650, 481, 'identity skip connection', { size: 11.5, fill: C.coral, weight: 600 });

    s += f.arrow(1320, lane, 1352, lane, { color: C.ink2, width: 1.7 });
    s += localBox(1354, 86, ['ReLU']);
    s += f.arrow(1440, lane, 1472, lane, { color: C.ink2, width: 1.7 });
    s += `<rect x="1474" y="${by}" width="78" height="${bh}" rx="9" fill="#f4f6f9" stroke="${C.hair}" stroke-width="1.3"/>`;
    s += f.T(1513, lane + 5, 'F_out', { size: 14, mono: true, weight: 600, fill: C.ink });
    s += f.T(1513, by + bh + 18, 'next block', { size: 11, fill: C.mut });

    f.mount(id, W, H, s);
  };
})();
