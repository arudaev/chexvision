/* Figure 1 — End-to-end pipeline (two swimlanes, U-flow) */
(function () {
  const W = 1680, H = 648;

  window.drawFig1 = function (id) {
    const f = FL, C = f.C;
    let s = '';

    // ---------- lane bands ----------
    const topBand = { y: 96, h: 168 }, botBand = { y: 386, h: 168 };
    s += `<rect x="56" y="${topBand.y}" width="${W - 112}" height="${topBand.h}" rx="14" fill="#f7fafd" stroke="${C.hairsoft}" stroke-width="1.2"/>`;
    s += `<rect x="56" y="${botBand.y}" width="${W - 112}" height="${botBand.h}" rx="14" fill="#f8fbfd" stroke="${C.hairsoft}" stroke-width="1.2"/>`;
    // lane captions
    s += f.T(74, topBand.y - 12, 'DATA PREPARATION', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.4, fill: C.ink });
    s += f.T(252, topBand.y - 12, '— one-time build, then per run', { anchor: 'start', size: 12.5, fill: C.mut });
    s += f.T(74, botBand.y - 12, 'TRAINING LOOP', { anchor: 'start', size: 12.5, weight: 700, upper: true, ls: 1.4, fill: C.ink });
    s += f.T(228, botBand.y - 12, '— per epoch', { anchor: 'start', size: 12.5, fill: C.mut });

    // ---------- node helper ----------
    function node(x, y, w, h, o) {
      let r = f.box(x, y, w, h, { stroke: C.hair, fill: '#ffffff', accent: o.accent, r: 12 });
      const cx = x + (o.accent ? (w + 10) / 2 + 5 : w / 2);
      r += f.T(cx, y + 30, o.title, { size: 16.5, weight: 700, fill: C.ink });
      r += f.lines(cx, y + 56, o.lines.map(t => ({ t })), { lh: 22, size: 13, mono: true, fill: C.ink2 });
      return r;
    }

    // =============== TOP LANE (L → R) ===============
    const tcy = topBand.y + 84;
    // dataset cylinder
    const cx = 92, cy = 112, cw = 232, ch = 140;
    s += f.cylinder(cx, cy, cw, ch, { fill: FL.RAMP[1].f, stroke: FL.RAMP[1].s, topFill: FL.RAMP[1].t, ry: 14 });
    s += f.T(cx + cw / 2, cy + 56, 'chest-xray-14-320', { size: 13.5, weight: 700, mono: true, fill: C.blueDk });
    s += f.lines(cx + cw / 2, cy + 82, [
      { t: '320² RGB · 36 Parquet shards' },
      { t: '~7.97 GB · pinned revision' },
    ].map(o => o), { lh: 20, size: 12, mono: true, fill: C.ink2 });
    s += f.T(cx + cw / 2, cy + ch + 22, 'Hugging Face dataset', { size: 11.5, fill: C.mut });

    // splits
    const sx = 470, sw = 330;
    s += node(sx, topBand.y + 26, sw, 116, {
      title: 'Deterministic split',
      lines: ['NIH manifests + SHA-256 bucket', 'train 77,967 · val 8,557', 'test 25,596 (patient-disjoint)'],
    });
    // augmentation
    const ax = 938, aw = 600;
    s += node(ax, topBand.y + 26, aw, 116, {
      title: 'Augmentation pipeline',
      lines: ['CLAHE · h-flip · rotate ±15° · affine · color jitter', 'Gaussian blur · random erasing · ImageNet norm'],
    });

    // top-lane arrows
    s += f.arrow(cx + cw + 6, tcy, sx - 8, tcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(sx + sw + 6, tcy, ax - 8, tcy, { color: C.ink2, width: 1.9 });

    // =============== handoff (down-turn) ===============
    const fwCx = 1423;            // forward node centre x (defined below)
    const turnX = 1235;
    s += f.poly([[turnX, topBand.y + topBand.h], [turnX, 332], [fwCx, 332], [fwCx, botBand.y]], { color: C.blue, width: 1.9 });
    s += `<rect x="${(turnX + fwCx) / 2 - 96}" y="320" width="192" height="26" rx="13" fill="#fff" stroke="${FL.RAMP[1].s}" stroke-width="1.1"/>`;
    s += f.T((turnX + fwCx) / 2, 337, 'augmented mini-batch', { size: 12, mono: true, weight: 600, fill: C.blueDk });

    // =============== BOTTOM LANE (R → L) ===============
    const bcy = botBand.y + 84;
    // forward (right)
    const fx = 1226, fw = 394;
    s += node(fx, botBand.y + 26, fw, 116, {
      title: 'Dual-head forward', accent: C.blue,
      lines: ['AMP fp16 · single pass', '→ 14 logits   +   1 logit'],
    });
    // losses
    const lx = 812, lw = 372;
    s += node(lx, botBand.y + 26, lw, 116, {
      title: 'Combined loss',
      lines: ['weighted BCE×14 (pos_weight)', '+ BCE binary', 'L = 1.0·L_ml + 0.5·L_bin'],
    });
    // optimize
    const ox = 398, ow = 372;
    s += node(ox, botBand.y + 26, ow, 116, {
      title: 'Optimize',
      lines: ['AdamW + cosine · clip 1.0', 'grad accum ×4 · eff. batch 96'],
    });
    // publish (left, terminal)
    const px = 60, pw = 300;
    s += node(px, botBand.y + 26, pw, 116, {
      title: 'Select + publish', accent: C.coral,
      lines: ['best val macro AUC-ROC', 'checkpoint + history', '→ HF Hub · model card'],
    });

    // bottom-lane arrows (leftward)
    s += f.arrow(fx - 6, bcy, lx + lw + 8, bcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(lx - 6, bcy, ox + ow + 8, bcy, { color: C.ink2, width: 1.9 });
    s += f.arrow(ox - 6, bcy, px + pw + 8, bcy, { color: C.ink2, width: 1.9 });

    // =============== per-epoch loop (dashed return) ===============
    const loopY = 600;
    s += f.poly([[ox + ow / 2, botBand.y + botBand.h], [ox + ow / 2, loopY], [fwCx, loopY], [fwCx, botBand.y + botBand.h]],
      { color: C.mut, width: 1.5, dash: '5 5' });
    s += `<rect x="${(ox + ow / 2 + fwCx) / 2 - 188}" y="${loopY - 13}" width="376" height="26" rx="13" fill="#fff" stroke="${C.hair}" stroke-width="1.1"/>`;
    s += f.T((ox + ow / 2 + fwCx) / 2, loopY + 4.5, 'repeat every step  ·  best epoch is checkpointed', { size: 12, mono: true, fill: C.mut, weight: 500 });

    f.mount(id, W, H, s);
  };
})();
