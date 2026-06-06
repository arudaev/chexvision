/* ============================================================
   figlib.js — CheXVision publication-figure drawing library
   Builds SVG-string fragments. Shared palette + geometry so all
   four plates read as one system. No emoji, hairline strokes,
   restrained slate-blue feature pathway, coral as a meaningful
   second signal (primary multi-label task + key events).
   ============================================================ */
(function (global) {
  'use strict';

  const C = {
    paper:   '#ffffff',
    ink:     '#20242c',
    ink2:    '#3b414d',
    mut:     '#737a88',
    mut2:    '#9aa0ad',
    hair:    '#c6ccd6',
    hairsoft:'#e2e6ec',
    blue:    '#2f6fd6',
    blueDk:  '#214f96',
    coral:   '#cf6037',
    coralDk: '#a8482744',
    coralLine:'#cf6037',
    okline:  '#6f93c8',
  };

  // Feature-map slab tint ramp (front, top, right, stroke), light -> deep.
  const RAMP = [
    // 0 neutral (input image)
    { f:'#e8eaee', t:'#f4f5f7', r:'#d6d9df', s:'#aeb4bf' },
    // 1
    { f:'#e1ecfb', t:'#eff5fe', r:'#cbdef5', s:'#9cbbe4' },
    // 2
    { f:'#cbdef6', t:'#deecfb', r:'#b1d0f0', s:'#86abdd' },
    // 3
    { f:'#abc9f0', t:'#c6dcf7', r:'#90b5e9', s:'#6f97d6' },
    // 4
    { f:'#88b0ec', t:'#a6c6f2', r:'#6f97e1', s:'#5a82cf' },
    // 5
    { f:'#6592e3', t:'#85aceb', r:'#5077cc', s:'#43699f' },
  ];

  // fixed oblique direction (into the page, up-right)
  const OBL = { kx: 1.0, ky: -0.66 };

  function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  // ---- text -------------------------------------------------
  function T(x, y, s, o) {
    o = o || {};
    const size = o.size || 15;
    const w = o.weight || 400;
    const fill = o.fill || C.ink;
    const anchor = o.anchor || 'middle';
    const fam = o.mono ? "'JetBrains Mono', monospace" : "'Inter', sans-serif";
    const ls = o.ls != null ? `letter-spacing:${o.ls}px;` : '';
    const op = o.opacity != null ? `opacity:${o.opacity};` : '';
    const tr = o.rotate != null ? ` transform="rotate(${o.rotate} ${x} ${y})"` : '';
    const up = o.upper ? 'text-transform:uppercase;' : '';
    return `<text x="${x}" y="${y}" text-anchor="${anchor}" `
      + `style="font:${w} ${size}px ${fam};fill:${fill};${ls}${op}${up}"${tr}>${esc(s)}</text>`;
  }
  // multi-line stacked text
  function lines(x, y, arr, o) {
    o = o || {};
    const lh = o.lh || 18;
    return arr.map((ln, i) => {
      const opt = Object.assign({}, o, ln);
      return T(x, y + i * lh, ln.t, opt);
    }).join('');
  }

  // ---- arrows ----------------------------------------------
  function arrow(x1, y1, x2, y2, o) {
    o = o || {};
    const col = o.color || C.mut;
    const w = o.width || 1.7;
    const hs = o.head || 9;
    const ang = Math.atan2(y2 - y1, x2 - x1);
    // shorten line so it meets the back of the head
    const lx = x2 - Math.cos(ang) * hs * 0.9;
    const ly = y2 - Math.sin(ang) * hs * 0.9;
    const dash = o.dash ? `stroke-dasharray="${o.dash}"` : '';
    const a1x = x2 - Math.cos(ang - 0.4) * hs, a1y = y2 - Math.sin(ang - 0.4) * hs;
    const a2x = x2 - Math.cos(ang + 0.4) * hs, a2y = y2 - Math.sin(ang + 0.4) * hs;
    return `<line x1="${x1}" y1="${y1}" x2="${lx}" y2="${ly}" stroke="${col}" stroke-width="${w}" ${dash} stroke-linecap="round"/>`
      + `<path d="M${x2} ${y2} L${a1x} ${a1y} L${a2x} ${a2y} Z" fill="${col}"/>`;
  }
  // elbow / orthogonal connector (h then v then h optional). pts = [[x,y],...]
  function poly(pts, o) {
    o = o || {};
    const col = o.color || C.mut;
    const w = o.width || 1.7;
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join(' ');
    const last = pts[pts.length - 1], prev = pts[pts.length - 2];
    const ang = Math.atan2(last[1] - prev[1], last[0] - prev[0]);
    const hs = o.head || 9;
    const a1x = last[0] - Math.cos(ang - 0.4) * hs, a1y = last[1] - Math.sin(ang - 0.4) * hs;
    const a2x = last[0] - Math.cos(ang + 0.4) * hs, a2y = last[1] - Math.sin(ang + 0.4) * hs;
    const headStr = o.noHead ? '' : `<path d="M${last[0]} ${last[1]} L${a1x} ${a1y} L${a2x} ${a2y} Z" fill="${col}"/>`;
    const dash = o.dash ? `stroke-dasharray="${o.dash}"` : '';
    return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" ${dash} stroke-linejoin="round" stroke-linecap="round"/>` + headStr;
  }

  // ---- isometric tensor slab -------------------------------
  // x,y = front-face top-left. w,h front size. depth = oblique magnitude (encodes channels).
  // level = ramp index. opts: { front (center text), frontSize, frontFill, label/sublabel below handled separately }
  function slab(x, y, w, h, depth, level, o) {
    o = o || {};
    const c = RAMP[Math.max(0, Math.min(RAMP.length - 1, level))];
    const ox = depth * OBL.kx, oy = depth * OBL.ky;
    const sw = o.strokeW || 1.1;
    const topPts = `${x},${y} ${x + w},${y} ${x + w + ox},${y + oy} ${x + ox},${y + oy}`;
    const rightPts = `${x + w},${y} ${x + w + ox},${y + oy} ${x + w + ox},${y + h + oy} ${x + w},${y + h}`;
    let s = '';
    s += `<polygon points="${rightPts}" fill="${c.r}" stroke="${c.s}" stroke-width="${sw}" stroke-linejoin="round"/>`;
    s += `<polygon points="${topPts}" fill="${c.t}" stroke="${c.s}" stroke-width="${sw}" stroke-linejoin="round"/>`;
    s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c.f}" stroke="${c.s}" stroke-width="${sw}"/>`;
    if (o.front) {
      const fs = o.frontSize || 17;
      s += T(x + w / 2, y + h / 2 + fs * 0.35, o.front, { size: fs, weight: 700, fill: o.frontFill || C.ink, mono: o.frontMono });
    }
    if (o.frontStripes) {
      // subtle diagonal hairlines to denote an image
      let st = `<clipPath id="${o.clip}"><rect x="${x}" y="${y}" width="${w}" height="${h}"/></clipPath><g clip-path="url(#${o.clip})">`;
      for (let i = -h; i < w; i += 8) st += `<line x1="${x + i}" y1="${y}" x2="${x + i + h}" y2="${y + h}" stroke="${C.mut2}" stroke-width="0.8" opacity="0.5"/>`;
      st += '</g>';
      s += st;
    }
    return s;
  }

  // ---- flat process box (rounded) --------------------------
  function box(x, y, w, h, o) {
    o = o || {};
    const fill = o.fill || '#ffffff';
    const stroke = o.stroke || C.hair;
    const sw = o.strokeW || 1.3;
    const r = o.r != null ? o.r : 9;
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
    if (o.accent) { // left accent bar
      s += `<path d="M${x + 1.5} ${y + r} a${r - 1.5} ${r - 1.5} 0 0 1 ${r - 1.5} ${-(r - 1.5)} M${x + 1.5} ${y + h - r} a${r - 1.5} ${r - 1.5} 0 0 0 ${r - 1.5} ${r - 1.5}" />`;
      s += `<rect x="${x}" y="${y}" width="4" height="${h}" rx="2" fill="${o.accent}"/>`;
      s += `<rect x="${x + 4}" y="${y}" width="6" height="${h}" fill="${fill}"/>`;
    }
    return s;
  }

  // ---- cylinder (datastore) --------------------------------
  function cylinder(x, y, w, h, o) {
    o = o || {};
    const fill = o.fill || RAMP[1].f;
    const stroke = o.stroke || RAMP[1].s;
    const ry = o.ry || 13;
    const sw = o.strokeW || 1.3;
    let s = '';
    s += `<path d="M${x} ${y + ry} V${y + h - ry} A${w / 2} ${ry} 0 0 0 ${x + w} ${y + h - ry} V${y + ry}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
    s += `<ellipse cx="${x + w / 2}" cy="${y + ry}" rx="${w / 2}" ry="${ry}" fill="${o.topFill || RAMP[1].t}" stroke="${stroke}" stroke-width="${sw}"/>`;
    return s;
  }

  // ---- small bracket with label ----------------------------
  // horizontal bracket under [x1..x2] at y, label centered below
  function bracketH(x1, x2, y, label, o) {
    o = o || {};
    const col = o.color || C.mut;
    const drop = o.drop || 8;
    let s = `<path d="M${x1} ${y} V${y + drop} H${x2} V${y}" fill="none" stroke="${col}" stroke-width="1.2"/>`;
    if (label) s += T((x1 + x2) / 2, y + drop + (o.size || 12) + 4, label, { size: o.size || 12, fill: o.labelFill || C.mut, mono: o.mono, weight: o.weight || 500, upper: o.upper, ls: o.ls });
    return s;
  }
  // top bracket over [x1..x2] at y, label above
  function bracketT(x1, x2, y, label, o) {
    o = o || {};
    const col = o.color || C.mut;
    const rise = o.rise || 8;
    let s = `<path d="M${x1} ${y} V${y - rise} H${x2} V${y}" fill="none" stroke="${col}" stroke-width="1.2"/>`;
    if (label) s += T((x1 + x2) / 2, y - rise - 6, label, { size: o.size || 12, fill: o.labelFill || C.mut, mono: o.mono, weight: o.weight || 600, upper: o.upper, ls: o.ls });
    return s;
  }

  function chip(x, y, txt, o) {
    o = o || {};
    const padX = o.padX || 9, h = o.h || 22;
    const w = o.w || (txt.length * (o.cw || 7.2) + padX * 2);
    const fill = o.fill || '#fff', stroke = o.stroke || C.hair;
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${h / 2}" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`;
    s += T(x + w / 2, y + h / 2 + (o.size || 12) * 0.35, txt, { size: o.size || 12, fill: o.fill2 || C.ink2, mono: o.mono, weight: o.weight || 500 });
    return { svg: s, w: w };
  }

  // ---- shared dual task-head block -------------------------
  // o: { vecRight, vcy, splitX, hx, hw, mlY, binY }
  function dualHeads(o) {
    const hw = o.hw || 444, hx = o.hx, mlY = o.mlY, binY = o.binY;
    let s = '';
    s += poly([[o.vecRight, o.vcy], [o.splitX, o.vcy], [o.splitX, mlY], [hx - 6, mlY]], { color: C.ink2, width: 1.8 });
    s += poly([[o.vecRight, o.vcy], [o.splitX, o.vcy], [o.splitX, binY], [hx - 6, binY]], { color: C.ink2, width: 1.8 });
    // multi-label (primary, coral)
    const mlH = 132;
    s += box(hx, mlY - mlH / 2, hw, mlH, { stroke: C.hair, accent: C.coral, fill: '#fffcfa', r: 11 });
    s += T(hx + 26, mlY - 38, 'Multi-label head', { anchor: 'start', size: 18, weight: 700, fill: C.ink });
    s += T(hx + hw - 22, mlY - 38, 'primary task', { anchor: 'end', size: 11, mono: true, upper: true, ls: 0.4, fill: C.coral, weight: 600 });
    s += T(hx + 26, mlY - 12, 'Linear 512 → 14   ·   sigmoid', { anchor: 'start', size: 14, mono: true, fill: C.ink2 });
    s += T(hx + 26, mlY + 10, 'weighted BCE · per-class pos_weight', { anchor: 'start', size: 12.5, mono: true, fill: C.mut });
    for (let i = 0; i < 14; i++) {
      const tx = hx + 26 + i * 21;
      s += `<rect x="${tx}" y="${mlY + 28}" width="13" height="22" rx="2.5" fill="${i % 4 === 0 ? C.coral : '#f2d9cd'}" stroke="${C.coral}" stroke-width="0.7" opacity="${i % 4 === 0 ? 0.92 : 0.6}"/>`;
    }
    s += T(hx + 26 + 14 * 21 + 12, mlY + 44, '14 pathologies', { anchor: 'start', size: 11.5, fill: C.mut });
    // binary (blue)
    const binH = 112;
    s += box(hx, binY - binH / 2, hw, binH, { stroke: C.hair, accent: C.blue, fill: '#fafcff', r: 11 });
    s += T(hx + 26, binY - 26, 'Binary head', { anchor: 'start', size: 18, weight: 700, fill: C.ink });
    s += T(hx + hw - 22, binY - 26, 'auxiliary', { anchor: 'end', size: 11, mono: true, upper: true, ls: 0.4, fill: C.blue, weight: 600 });
    s += T(hx + 26, binY + 2, 'Linear 512 → 1   ·   sigmoid', { anchor: 'start', size: 14, mono: true, fill: C.ink2 });
    s += T(hx + 26, binY + 26, 'BCE   ·   Normal vs. Abnormal', { anchor: 'start', size: 12.5, mono: true, fill: C.mut });
    return s;
  }

  // 512-d (or any) feature-vector glyph; returns {svg, right}
  function vector(vx, vcy, o) {
    o = o || {};
    const cell = 12, ncell = o.ncell || 11, vh = ncell * cell, w = o.w || 26, lv = o.lvl || 2;
    const vy0 = vcy - vh / 2;
    let s = '';
    for (let i = 0; i < ncell; i++)
      s += `<rect x="${vx}" y="${vy0 + i * cell}" width="${w}" height="${cell}" fill="${RAMP[lv].f}" stroke="${RAMP[lv].s}" stroke-width="0.8"/>`;
    s += `<rect x="${vx}" y="${vy0}" width="${w}" height="${vh}" fill="none" stroke="${C.mut}" stroke-width="1.1"/>`;
    if (o.label) s += T(vx + w / 2, vy0 + vh + 18, o.label, { size: 12.5, mono: true, fill: C.mut, weight: 600 });
    if (o.sub) s += T(vx + w / 2, vy0 + vh + 34, o.sub, { size: 11.5, fill: C.mut2 });
    return { svg: s, right: vx + w };
  }

  // mount an svg string into a container by id, with viewBox
  function mount(id, vbW, vbH, body, o) {
    o = o || {};
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = `<svg viewBox="0 0 ${vbW} ${vbH}" width="100%" `
      + `style="display:block" xmlns="http://www.w3.org/2000/svg">`
      + (o.bg !== false ? `<rect x="0" y="0" width="${vbW}" height="${vbH}" fill="${o.bg || C.paper}"/>` : '')
      + body + `</svg>`;
  }

  global.FL = { C, RAMP, OBL, T, lines, arrow, poly, slab, box, cylinder, bracketH, bracketT, chip, dualHeads, vector, mount, esc };
})(window);
