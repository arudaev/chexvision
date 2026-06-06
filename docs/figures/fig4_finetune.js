/* Figure 4 — Two-phase fine-tuning schedule (DenseNet-121) */
(function () {
  const W = 1680, H = 640;

  window.drawFig4 = function (id) {
    const f = FL, C = f.C;
    let s = '';

    const x0 = 232, x1 = 1592, EMAX = 60;
    const X = e => x0 + (e / EMAX) * (x1 - x0);
    const unfreezeX = X(6), bestX = X(18);

    // ---- phase band shading ----
    s += `<rect x="${x0}" y="118" width="${unfreezeX - x0}" height="446" fill="#fbf3ef"/>`;
    s += `<rect x="${unfreezeX}" y="118" width="${x1 - unfreezeX}" height="446" fill="#f7fafe"/>`;

    // ---- phase brackets (top) ----
    s += f.bracketT(x0 + 2, unfreezeX - 2, 116, 'PHASE 1 — warm up the heads', { rise: 9, size: 12.5, weight: 700, upper: true, ls: 0.8, color: C.hair, labelFill: C.coralLine });
    s += f.bracketT(unfreezeX + 2, x1 - 2, 116, 'PHASE 2 — end-to-end fine-tuning', { rise: 9, size: 12.5, weight: 700, upper: true, ls: 0.8, color: C.hair, labelFill: C.blueDk });

    // ---- unfreeze divider ----
    s += `<line x1="${unfreezeX}" y1="124" x2="${unfreezeX}" y2="560" stroke="${C.coral}" stroke-width="1.6" stroke-dasharray="5 5"/>`;
    s += `<rect x="${unfreezeX - 116}" y="128" width="232" height="40" rx="8" fill="#fff" stroke="${C.coral}" stroke-width="1.3"/>`;
    s += f.T(unfreezeX, 144, 'epoch 6', { size: 12.5, weight: 700, fill: C.coralLine, mono: true });
    s += f.T(unfreezeX, 160, 'unfreeze_backbone()', { size: 12.5, mono: true, fill: C.ink2 });

    // ---- best checkpoint marker ----
    s += `<line x1="${bestX}" y1="186" x2="${bestX}" y2="560" stroke="${C.ink2}" stroke-width="1.3" stroke-dasharray="2 4"/>`;
    s += `<path d="M${bestX} 188 l 9 9 l -9 9 l -9 -9 z" fill="${C.coral}"/>`;
    s += f.T(bestX, 184, 'best val checkpoint · epoch 18', { size: 12, weight: 600, fill: C.ink });

    // ================= lanes (Gantt) =================
    const laneH = 64;
    const bbY = 214, hdY = 300;

    // hatch helper for frozen segment
    const hatch = (x, y, w, h, col) => {
      let g = `<clipPath id="frz"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/></clipPath><g clip-path="url(#frz)">`;
      for (let i = -h; i < w; i += 9) g += `<line x1="${x + i}" y1="${y}" x2="${x + i + h}" y2="${y + h}" stroke="${col}" stroke-width="1" opacity="0.55"/>`;
      return g + '</g>';
    };

    // lane labels
    s += f.lines(x0 - 22, bbY + 24, [
      { t: 'Backbone', size: 15, weight: 700, fill: C.ink, anchor: 'end' },
      { t: 'DenseNet-121', size: 12, mono: true, fill: C.mut, anchor: 'end' },
    ], { lh: 19, anchor: 'end' });
    s += f.lines(x0 - 22, hdY + 22, [
      { t: 'Feature layer', size: 15, weight: 700, fill: C.ink, anchor: 'end' },
      { t: '+ two heads', size: 12, mono: true, fill: C.mut, anchor: 'end' },
    ], { lh: 19, anchor: 'end' });

    // Backbone lane: frozen segment + trainable segment
    const fw = unfreezeX - x0;
    s += `<rect x="${x0}" y="${bbY}" width="${fw}" height="${laneH}" rx="9" fill="#eef0f3" stroke="${C.mut2}" stroke-width="1.2"/>`;
    s += hatch(x0, bbY, fw, laneH, C.mut2);
    s += f.T(x0 + fw / 2, bbY + laneH / 2 + 4.5, 'FROZEN', { size: 12.5, weight: 700, upper: true, ls: 1, fill: C.mut });
    const tw = x1 - unfreezeX;
    s += `<rect x="${unfreezeX}" y="${bbY}" width="${tw}" height="${laneH}" rx="9" fill="${FL.RAMP[2].f}" stroke="${FL.RAMP[2].s}" stroke-width="1.2"/>`;
    s += f.T(unfreezeX + tw / 2, bbY + laneH / 2 + 5, 'trainable — backbone specializes to chest-X-ray features', { size: 13.5, weight: 600, fill: C.blueDk });

    // Heads lane: trainable throughout (coral = contains the primary head)
    const hw2 = x1 - x0;
    s += `<rect x="${x0}" y="${hdY}" width="${hw2}" height="${laneH}" rx="9" fill="#fdeee7" stroke="${C.coral}" stroke-width="1.2"/>`;
    s += `<line x1="${unfreezeX}" y1="${hdY}" x2="${unfreezeX}" y2="${hdY + laneH}" stroke="${C.coral}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>`;
    s += f.T(x0 + fw / 2, hdY + laneH / 2 + 5, 'trainable', { size: 13, weight: 600, fill: C.coralLine });
    s += f.T(unfreezeX + tw / 2, hdY + laneH / 2 + 5, 'trainable — newly-initialized weights adapt throughout', { size: 13.5, weight: 600, fill: C.coralLine });

    // ================= LR track =================
    const lrTop = 418, lrZero = 524;
    const lrAxX = x0;
    // y for an lr value where 1e-3 -> lrTop, 0 -> lrZero (1e-4 sits 1/10 up)
    const lrY = v => lrZero - (v / 1e-3) * (lrZero - lrTop);
    // gridlines / labels
    s += `<line x1="${x0}" y1="${lrZero}" x2="${x1}" y2="${lrZero}" stroke="${C.hair}" stroke-width="1"/>`;
    [['1e-3', 1e-3], ['1e-4', 1e-4], ['0', 0]].forEach(([lab, v]) => {
      const y = lrY(v);
      s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${C.hairsoft}" stroke-width="1" stroke-dasharray="3 4"/>`;
      s += f.T(x0 - 12, y + 4, lab, { size: 11.5, mono: true, fill: C.mut, anchor: 'end' });
    });
    s += f.T(x0 - 12, lrTop - 18, 'learning rate', { size: 12, weight: 600, fill: C.ink2, anchor: 'end' });

    // curve: flat 1e-3 (phase1) -> step to 1e-4 -> cosine anneal to ~0
    let d = `M${x0} ${lrY(1e-3)} L${unfreezeX} ${lrY(1e-3)}`;
    d += ` L${unfreezeX} ${lrY(1e-4)}`;
    const T2 = EMAX - 6;
    for (let e = 6; e <= EMAX; e += 1) {
      const prog = (e - 6) / T2;
      const lr = 0.5 * 1e-4 * (1 + Math.cos(Math.PI * prog)); // cosine to 0
      d += ` L${X(e)} ${lrY(lr)}`;
    }
    s += `<path d="${d}" fill="none" stroke="${C.blue}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>`;
    // phase markers on curve
    s += `<rect x="${(x0 + unfreezeX) / 2 - 52}" y="${lrY(1e-3) - 30}" width="104" height="22" rx="11" fill="#fff" stroke="${C.hair}" stroke-width="1"/>`;
    s += f.T((x0 + unfreezeX) / 2, lrY(1e-3) - 15, 'lr = 1e-3', { size: 12, mono: true, weight: 600, fill: C.blueDk });
    s += f.T(unfreezeX + 150, lrY(1e-4) - 14, 'lr = 1e-4, cosine anneal → 0', { size: 12, mono: true, weight: 600, fill: C.blueDk, anchor: 'start' });

    // ================= epoch axis =================
    const axY = lrZero;
    [1, 5, 6, 10, 18, 20, 30, 40, 50, 60].forEach(e => {
      const x = X(e);
      s += `<line x1="${x}" y1="${axY}" x2="${x}" y2="${axY + 6}" stroke="${C.mut}" stroke-width="1.1"/>`;
      s += f.T(x, axY + 22, String(e), { size: 11.5, mono: true, fill: (e === 6 ? C.coralLine : e === 18 ? C.ink : C.mut), weight: (e === 6 || e === 18) ? 700 : 400 });
    });
    s += f.T((x0 + x1) / 2, axY + 44, 'epoch', { size: 12, fill: C.mut, weight: 600, upper: true, ls: 1 });

    f.mount(id, W, H, s);
  };
})();
