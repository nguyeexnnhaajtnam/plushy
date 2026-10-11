'use strict';
/* =========================================================================
   APPLICATION — UI generation from the active SpeciesDefinition, camera,
   pointer/tool routing, fixed-step main loop, resize and cleanup.
   ========================================================================= */
(() => {
const { v3, mat4, clamp, damp, lerp, hexToLinear } = Engine.math;
const STEP = 1 / 60, MAX_STEPS = 3;
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids) if (kid !== null && kid !== undefined) e.append(kid);
  return e;
};

/* Engine-global controls share the species control schema but bind to the app. */
const GLOBAL_CONTROLS = [
  { id: 'pause', type: 'button', label: (app) => (app.paused ? 'Resume' : 'Pause'), action: 'app.pause', group: 'Run', hint: 'Space' },
  { id: 'reset', type: 'button', label: 'Reset', action: 'app.reset', group: 'Run', hint: 'R' },
  { id: 'quarter', type: 'toggle', label: 'Quarter speed', bind: 'app.quarter', default: false, group: 'Options', hint: 'Run the simulation at 0.25×' },
  { id: 'mesh', type: 'toggle', label: 'Mesh', bind: 'app.debugMesh', default: false, group: 'Options', hint: 'Overlay the XPBD lattice' },
];

const LIGHTS = {
  key: v3.norm([-0.5, 0.86, 0.42]), keyColor: [1.0, 0.94, 0.86],
  fill: v3.norm([0.7, 0.25, -0.45]), fillColor: [0.2, 0.22, 0.27],
  sky: [0.4, 0.41, 0.43], ground: [0.2, 0.18, 0.16],
};

const ENGINE_NOTES = {
  implemented: [
    'XPBD soft body: every hex cell is shape-matched (Müller rotation extraction, warm-started, mass-weighted so momentum is conserved) and split into five tetrahedra with volume constraints. Stuffing scales compliance, inflation and mass.',
    'The lattice is built from the species SDF so that it covers every render vertex. A surface-nets mesh is embedded in it and skinned on the GPU by trilinear FFD, and normals use the cofactor of the local deformation gradient.',
    'Rigid parts attach to the lattice frame at an anchor (polar decomposition of the local deformation gradient) and can hang off an optional 1-DoF damped joint.',
    'Fur uses alpha-to-coverage shells (4× MSAA) with procedural tapered strands. The pile lags on a spring driven by particle acceleration and drag, and a per-vertex comb field and minky nap shading show where it was brushed.',
    'Behaviour runs as a prioritised state machine with autonomous, user, reactive and environment states, preemption, a resume stack and retrigger hooks. Reactions are data rules: event + region → state.',
    'Steps are deterministic: fixed 60 Hz steps with 8 substeps, a seeded RNG and a fixed constraint order. Variable frame rates run up to 3 steps per frame and drop any remaining backlog.',
  ],
  approximations: [
    'Locomotion is hybrid: muscles pose the lattice rest shape (physical), while steering, pitch and roll are PD torques applied as body-wide velocity changes.',
    'Collisions are only against the environment box. There is no self-collision and no creature–item contact; feeding uses a scripted suction radius.',
    'Picking ray-tests CPU-skinned render vertices, not triangles.',
    'Shadows come from one directional shadow map, and fur shells do not cast shadows.',
  ],
  limitations: [
    'The solver runs single-threaded in JavaScript on the main thread (~4–6 ms per step). The GPU only skins and renders.',
    'Without WebGPU nothing is rendered. There is no WebGL fallback.',
  ],
};

/* ---------- Orbit camera ------------------------------------------------- */
class OrbitCamera {
  constructor() { this.yaw = 0.6; this.pitch = 0.3; this.dist = 6; this.target = [0, 1, 0]; this.base = [0, 1, 0]; this.follow = 0.5; this.fov = (30 * Math.PI) / 180; }
  configure(cfg) {
    Object.assign(this, { yaw: cfg.yaw, pitch: cfg.pitch, dist: cfg.distance, follow: cfg.follow ?? 0.5 });
    this.base = cfg.target.slice(); this.target = cfg.target.slice(); this.home = { ...cfg };
  }
  update(dt, focus) {
    const want = v3.lerp(this.base, focus, this.follow);
    this.target = v3.lerp(this.target, want, damp(1.8, dt));
  }
  eye(aspect) {
    const d = this.dist * Math.pow(Math.max(1, 1.25 / aspect), 0.75), cp = Math.cos(this.pitch);
    return [this.target[0] + d * cp * Math.sin(this.yaw), this.target[1] + d * Math.sin(this.pitch), this.target[2] + d * cp * Math.cos(this.yaw)];
  }
  matrices(aspect) {
    const eye = this.eye(aspect);
    const vp = mat4.mul(mat4.perspective(this.fov, aspect, 0.05, 80), mat4.lookAt(eye, this.target, [0, 1, 0]));
    this.cache = { eye, viewProj: vp, inv: mat4.invert(vp), aspect };
    return this.cache;
  }
  ray(nx, ny) {
    const { inv, eye } = this.cache, a = mat4.xform(inv, [nx, ny, 0]), b = mat4.xform(inv, [nx, ny, 1]);
    return { o: a, d: v3.norm(v3.sub(b, a)), view: v3.norm(v3.sub(this.target, eye)) };
  }
}

/* ---------- Control renderer (shared by species + global controls) ------- */
function renderControls(container, controls, B) {
  container.textContent = '';
  const live = [], groups = [];
  for (const c of controls) {
    let g = groups.find((x) => x.name === (c.group || ''));
    if (!g) groups.push((g = { name: c.group || '', items: [] }));
    g.items.push(c);
  }
  for (const g of groups) {
    const wrap = el('div', { class: 'group' });
    if (g.name) wrap.append(el('div', { class: 'group-label', text: g.name }));
    const buttons = g.items.filter((c) => c.type === 'button');
    let grid = null;
    if (buttons.length) {
      grid = el('div', { class: 'btn-grid' });
      wrap.append(grid);
    }
    for (const c of g.items) {
      const label = () => (typeof c.label === 'function' ? c.label(B.labelCtx()) : c.label);
      if (c.type === 'button') {
        const b = el('button', { class: 'btn' + (c.primary ? ' primary' : ''), type: 'button', title: c.hint || null, 'data-id': c.id, text: label() });
        const i = buttons.indexOf(c), rem = buttons.length % 3;
        if (i === buttons.length - 1 && rem === 1) b.classList.add('wide');
        if (i === buttons.length - 1 && rem === 2) b.style.gridColumn = 'span 2';
        b.addEventListener('click', () => { B.run(c); b.classList.remove('flash'); void b.offsetWidth; b.classList.add('flash'); });
        grid.append(b);
        live.push(() => { b.textContent = label(); b.disabled = !B.enabled(c); });
      } else if (c.type === 'toggle') {
        const sw = el('button', { class: 'switch', type: 'button', role: 'switch', 'aria-checked': String(!!B.get(c.bind)), 'aria-label': label() });
        sw.addEventListener('click', () => { const v = !B.get(c.bind); B.set(c.bind, v); sw.setAttribute('aria-checked', String(v)); });
        wrap.append(el('div', { class: 'toggle-row' }, el('span', { class: 'lbl' }, label(), c.hint ? el('span', { class: 'sub', text: c.hint }) : null), sw));
        live.push(() => { sw.setAttribute('aria-checked', String(!!B.get(c.bind))); sw.disabled = !B.enabled(c); });
      } else if (c.type === 'slider') {
        const id = `ctl-${B.prefix}-${c.id}`, v0 = B.get(c.bind) ?? c.default;
        const out = el('output', { for: id });
        const input = el('input', { type: 'range', id, min: c.min, max: c.max, step: c.step || 0.01, value: v0 });
        const paint = (v) => { out.textContent = c.format ? c.format(+v) : (+v).toFixed(2); input.style.setProperty('--p', `${((v - c.min) / (c.max - c.min)) * 100}%`); };
        input.addEventListener('input', () => { B.set(c.bind, +input.value); paint(input.value); });
        paint(v0);
        wrap.append(el('div', { class: 'slider' }, el('div', { class: 'row' }, el('label', { for: id, text: label() }), out), input));
        live.push(() => { input.disabled = !B.enabled(c); });
      } else if (c.type === 'select') {
        const s = el('select', { class: 'sel', 'aria-label': label() }, ...c.options.map((o) => el('option', { value: o.value, text: o.label })));
        s.value = B.get(c.bind) ?? c.default;
        s.addEventListener('change', () => B.set(c.bind, s.value));
        wrap.append(s);
      } else if (c.type === 'preset') {
        const row = el('div', { class: 'swatches', role: 'radiogroup', 'aria-label': label() });
        const opts = B.presets(c), cur = () => B.get(c.bind) ?? c.default;
        const btns = opts.map((o) => {
          const b = el('button', { class: 'swatch', type: 'button', role: 'radio', 'aria-checked': String(cur() === o.id) },
            el('span', { class: 'chip', style: `background: linear-gradient(180deg, ${o.colors[0]} 0 56%, ${o.colors[1]} 56% 100%)` }), o.label);
          b.addEventListener('click', () => { B.set(c.bind, o.id); btns.forEach((x, i) => x.setAttribute('aria-checked', String(opts[i].id === o.id))); });
          return b;
        });
        row.append(...btns);
        wrap.append(row);
      }
    }
    container.append(wrap);
  }
  return () => live.forEach((f) => f());
}

/* ---------- Application -------------------------------------------------- */
class App {
  constructor() {
    this.canvas = $('gpu');
    this.paused = false; this.quarter = false; this.debugMesh = false;
    this.toolId = 'hand'; this.assets = new Map();
    this.creature = null; this.def = null; this.renderer = null;
    this.camera = new OrbitCamera();
    this.acc = 0; this.last = performance.now(); this.stepMs = 0; this.uiClock = 0;
    this.pointers = new Map(); this.mode = null; this.hover = null; this.toolHintUntil = 0;
    this.purity = this.scanEngineSource();
  }
  async start() {
    this.buildSpeciesList();
    this.refreshGlobal = renderControls($('globalControls'), GLOBAL_CONTROLS, this.globalBinding());
    try {
      this.renderer = await Engine.Renderer.create(this.canvas);
      this.renderer.device.lost.then((info) => { if (info.reason !== 'destroyed') this.fail(new Error(`The GPU device was lost (${info.message || info.reason}).`)); });
    } catch (err) { this.fail(err); }
    const first = Engine.registry.find((s) => s.status === 'complete' && s.validation.ok);
    await this.loadSpecies(first.id);
    this.bindInput();
    new ResizeObserver(() => (this.needsResize = true)).observe($('viewport'));
    this.needsResize = true;
    addEventListener('pagehide', () => this.renderer?.destroy());
    requestAnimationFrame((t) => this.frame(t));
  }
  fail(err) {
    console.error(err);
    const box = $('fallback'), webgpu = err instanceof Engine.GPUUnavailable;
    box.hidden = false;
    box.innerHTML = '';
    box.append(el('div', { class: 'box' },
      el('h3', { text: webgpu ? 'WebGPU isn’t available here' : 'The renderer stopped' }),
      el('p', { text: err.message }),
      el('p', { text: webgpu
        ? 'Plush Creatures needs WebGPU for GPU skinning, fur shells and shadows. Try a current Chrome or Edge, Safari 18 or later, or Firefox with WebGPU enabled, on a device with hardware acceleration turned on.'
        : 'Reload the page to try again. If it keeps happening, the browser may have blocked the GPU for this tab.' }),
      el('p', { text: 'The species definitions, controls and technical notes on this page are still generated from the engine, but the simulation will not run without a renderer.' })));
    this.renderer = null;
    $('stateLabel').textContent = 'Renderer unavailable';
  }
  /* ---- species lifecycle */
  buildSpeciesList() {
    const list = $('speciesList');
    list.textContent = '';
    for (const s of Engine.registry) {
      const playable = s.status !== 'planned' && s.validation.ok;
      const chip = { complete: 'Complete', prototype: 'Prototype', planned: 'Planned' }[s.status] || s.status;
      const b = el('button', { class: 'species-opt', type: 'button', role: 'radio', 'aria-checked': 'false', 'data-id': s.id, disabled: !playable,
        title: playable ? s.description : `${s.description} — not implemented yet` },
        el('span', { class: 'nm', text: s.name }), el('span', { class: 'ct', text: playable ? s.category : `${s.category} · not playable` }),
        el('span', { class: `chip ${s.status}`, text: chip }));
      if (playable) b.addEventListener('click', () => this.loadSpecies(s.id));
      list.append(b);
    }
  }
  async loadSpecies(id) {
    const def = Engine.registry.find((s) => s.id === id);
    if (!def || def.status === 'planned' || (this.def && this.def.id === id)) return;
    const ov = $('overlay');
    ov.textContent = `Sewing the ${def.name}…`; ov.hidden = false;
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 30)));
    this.endInteraction();
    let assets = this.assets.get(id);
    if (!assets) { assets = Engine.buildCreatureAssets(def); this.assets.set(id, assets); }
    this.def = def;
    this.creature = new Engine.Creature(def, assets, { seed: 7 });
    this.materialPreset = null;
    this.renderer?.setCreature(this.creature);
    this.camera.configure(def.environment.camera);
    const tools = this.toolsFor(def);
    if (!tools.includes(this.toolId)) this.toolId = tools[0];
    this.buildSpeciesUI();
    this.acc = 0;
    ov.hidden = true;
  }
  toolsFor(def) { return def.interactions.tools.filter((t) => Engine.TOOLS[t] && (Engine.TOOLS[t].requires !== 'fur' || def.fur)); }
  buildSpeciesUI() {
    const def = this.def, D = def.display;
    document.title = `${def.name} — Plush Creatures`;
    document.documentElement.style.setProperty('--accent', D.theme?.accent || '#8a5a33');
    $('kicker').textContent = D.kicker;
    $('title').querySelector('.l1').textContent = D.title[0];
    $('title').querySelector('.l2').textContent = D.title[1];
    $('lede').textContent = D.description;
    $('speciesName').textContent = def.name;
    $('meta').textContent = `WebGPU · XPBD · ${Engine.ENVIRONMENTS[def.environment.preset].label}`;
    for (const b of $('speciesList').children) b.setAttribute('aria-checked', String(b.dataset.id === def.id));
    // tools
    const tools = $('tools');
    tools.textContent = '';
    for (const t of this.toolsFor(def)) {
      const T = Engine.TOOLS[t];
      const b = el('button', { class: 'tool', type: 'button', role: 'radio', 'aria-checked': String(t === this.toolId), title: T.hint },
        el('span', { class: 'g', 'aria-hidden': 'true', text: T.glyph }), T.label);
      b.addEventListener('click', () => this.setTool(t));
      tools.append(b);
    }
    this.toolHintUntil = performance.now() + 6000;
    // controls
    this.refreshSpecies = renderControls($('speciesControls'), def.controls, this.speciesBinding());
    // readouts
    const ro = $('readouts');
    ro.textContent = '';
    this.readoutEls = def.readouts.map((r) => {
      const dd = el('dd', {}, el('span', { text: '—' }), r.unit ? el('small', { text: r.unit }) : null);
      ro.append(el('div', { class: 'readout' }, el('dt', { text: r.label }), dd));
      return { r, span: dd.firstChild };
    });
    this.buildNotes();
    this.refreshUI();
  }
  setTool(t) {
    this.toolId = t;
    for (const b of $('tools').children) b.setAttribute('aria-checked', String(b.textContent.includes(Engine.TOOLS[t].label)));
    this.toolHintUntil = performance.now() + 5000;
  }
  /* ---- bindings */
  speciesBinding() {
    const app = this, def = this.def, regs = Object.keys(def.materials.regions);
    return {
      prefix: def.id,
      labelCtx: () => app.creature.ctx,
      run: (c) => app.runAction(c.action),
      enabled: (c) => app.actionEnabled(c),
      get: (bind) => app.creature.getParam(bind),
      set: (bind, v) => app.creature.setParam(bind, v),
      presets: () => def.materials.presets.map((p) => ({ id: p.id, label: p.label, colors: (def.materials.swatch || regs.slice(0, 2)).map((r) => p.colors[r]) })),
    };
  }
  globalBinding() {
    const app = this;
    return {
      prefix: 'global', labelCtx: () => app, enabled: () => true,
      run: (c) => app.runAction(c.action),
      get: (bind) => app[bind.split('.')[1]],
      set: (bind, v) => { app[bind.split('.')[1]] = v; },
    };
  }
  runAction(id) {
    const c = this.creature;
    if (id === 'app.pause') { this.paused = !this.paused; this.last = performance.now(); }
    else if (id === 'app.reset') { if (c) { this.endInteraction(); c.reset(); this.camera.configure(this.def.environment.camera); } }
    else if (!c) return;
    else if (Engine.ENGINE_ACTIONS[id]) Engine.ENGINE_ACTIONS[id].run(c);
    else if (this.def.actions[id]) this.def.actions[id].run(c.ctx);
    this.refreshUI();
  }
  actionEnabled(control) {
    const c = this.creature;
    if (!c || !this.renderer) return false;
    if (control.enabled && !control.enabled(c.ctx)) return false;
    const a = this.def.actions?.[control.action];
    return !(a && a.enabled && !a.enabled(c.ctx));
  }
  /* ---- notes */
  buildNotes() {
    const def = this.def, A = this.creature.assets, body = $('notesBody');
    body.textContent = '';
    const list = (items, tag, cls) => el('ul', {}, ...items.map((t) => el('li', {}, el('span', { class: `tag ${cls}`, text: tag }), t)));
    body.append(
      el('h3', { text: `${def.name} — anatomy` }), el('p', { text: def.notes.anatomy }),
      el('p', { class: 'kv', text: `${A.vertexCount.toLocaleString()} vertices · ${A.triangles.toLocaleString()} triangles · ${A.lattice.n} particles · ${A.lattice.nc} hexes · ${A.lattice.tets.length / 4} tets · ${A.decalCount} stitch segments · built in ${A.buildMs.toFixed(0)} ms` }),
      el('h3', { text: 'Species behaviour' }), list(def.notes.implemented, 'Implemented', 'ok'), list(def.notes.approximations, 'Approx.', 'approx'),
      el('h3', { text: 'Engine' }), list(ENGINE_NOTES.implemented, 'Implemented', 'ok'), list(ENGINE_NOTES.approximations, 'Approx.', 'approx'), list(ENGINE_NOTES.limitations, 'Limit', 'limit'),
      el('h3', { text: 'Architecture checks' }),
      el('ul', {}, ...Engine.registry.map((s) => el('li', {}, el('span', { class: `tag ${s.validation.ok ? 'ok' : 'limit'}`, text: s.validation.ok ? 'Valid' : 'Invalid' }),
        `${s.name}: ${s.validation.planned ? 'planned, identity only (not playable)' : `${s.validation.checks} definition checks passed against engine capabilities`}${s.validation.errors.length ? ' — ' + s.validation.errors.join('; ') : ''}`)),
        el('li', {}, el('span', { class: `tag ${this.purity.hits.length ? 'limit' : 'ok'}`, text: this.purity.hits.length ? 'Leak' : 'Clean' }),
          this.purity.scanned ? `Engine source scanned for ${this.purity.terms.length} species-specific identifiers (${this.purity.terms.join(', ')}): ${this.purity.hits.length ? 'found ' + this.purity.hits.join(', ') : 'none found'}.` : 'Engine source scan unavailable.')),
    );
  }
  scanEngineSource() {
    const src = (document.getElementById('engine-src')?.textContent || '').replace(/\/\*[\s\S]*?\*\//g, '');
    const terms = new Set();
    for (const s of Engine.registry) {
      terms.add(s.id);
      s.name.split(/\s+/).forEach((w) => { if (w.length > 2 && w.toLowerCase() !== 'plush') terms.add(w.toLowerCase()); });
      (s.anatomy?.parts || []).forEach((p) => { if (p.attach) terms.add(p.id.toLowerCase()); });
      Object.keys(s.anatomy?.joints || {}).forEach((j) => terms.add(j.toLowerCase()));
      Object.keys(s.items || {}).forEach((i) => terms.add(i.toLowerCase()));
    }
    const esc = (t) => t.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
    const hits = [...terms].filter((t) => new RegExp(`\\b${esc(t)}\\b`, 'i').test(src));
    return { scanned: src.length > 0, terms: [...terms], hits };
  }
  /* ---- input */
  bindInput() {
    const cv = this.canvas;
    const ndc = (e) => { const r = cv.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2]; };
    const pick = (e) => {
      if (!this.creature || !this.camera.cache) return { hit: null, ray: null };
      const [x, y] = ndc(e), ray = this.camera.ray(x, y);
      return { ray, hit: this.creature.raycast(ray.o, ray.d) };
    };
    cv.addEventListener('pointerdown', (e) => {
      if (!this.renderer) return;
      cv.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 2) { this.endInteraction(); this.mode = 'pinch'; this.pinch = this.pinchDist(); return; }
      const { hit, ray } = pick(e), tool = Engine.TOOLS[this.toolId];
      if (hit && tool.begin(this.creature, hit, ray)) { this.mode = 'tool'; this.toolHintUntil = 0; }
      else this.mode = 'orbit';
      cv.classList.add('dragging');
    });
    cv.addEventListener('pointermove', (e) => {
      const prev = this.pointers.get(e.pointerId);
      if (!prev) { this.hoverAt(e, pick); return; }
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.mode === 'pinch' && this.pointers.size === 2) {
        const d = this.pinchDist();
        this.camera.dist = clamp(this.camera.dist * (this.pinch / d), 2, 16); this.pinch = d;
      } else if (this.mode === 'orbit') {
        this.camera.yaw -= dx * 0.006; this.camera.pitch = clamp(this.camera.pitch + dy * 0.005, 0.04, 1.35);
      } else if (this.mode === 'tool') {
        const { hit, ray } = pick(e);
        Engine.TOOLS[this.toolId].move(this.creature, hit, ray);
      }
    });
    const up = (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.delete(e.pointerId);
      if (this.mode === 'tool') this.endInteraction();
      if (this.pointers.size === 0) { this.mode = null; cv.classList.remove('dragging'); }
    };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', () => { if (!this.pointers.size) { this.hover = null; cv.classList.remove('on-creature'); } });
    cv.addEventListener('wheel', (e) => { e.preventDefault(); this.camera.dist = clamp(this.camera.dist * Math.exp(e.deltaY * 0.0012), 2, 16); }, { passive: false });
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    addEventListener('keydown', (e) => {
      if (e.target.closest && e.target.closest('input, select, textarea')) return;
      if (e.key === ' ' && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); this.runAction('app.pause'); }
      else if (e.key === 'r' || e.key === 'R') this.runAction('app.reset');
      else if (/^[1-9]$/.test(e.key)) { const t = this.toolsFor(this.def)[+e.key - 1]; if (t) this.setTool(t); }
    });
  }
  pinchDist() { const p = [...this.pointers.values()]; return Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) || 1; }
  hoverAt(e, pick) {
    if (e.pointerType !== 'mouse' || performance.now() - (this.hoverT || 0) < 70) return;
    this.hoverT = performance.now();
    const { hit } = pick(e);
    this.hover = hit;
    this.canvas.classList.toggle('on-creature', !!hit);
  }
  endInteraction() {
    if (this.mode === 'tool' && this.creature) Engine.TOOLS[this.toolId].end(this.creature);
    if (this.mode === 'tool') this.mode = null;
  }
  /* ---- loop */
  frame(now) {
    requestAnimationFrame((t) => this.frame(t));
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    const c = this.creature;
    if (c && this.renderer && !this.paused) {
      this.acc += dt * (this.quarter ? 0.25 : 1);
      let n = 0;
      const t0 = performance.now();
      while (this.acc >= STEP && n < MAX_STEPS) { c.step(STEP); this.acc -= STEP; n++; }
      if (this.acc >= STEP) this.acc = 0;
      if (n) this.stepMs = lerp(this.stepMs || (performance.now() - t0) / n, (performance.now() - t0) / n, 0.08);
    }
    this.uiClock += dt;
    if (this.uiClock > 0.15) { this.uiClock = 0; this.refreshUI(); }
    if (!this.renderer || !c) return;
    if (this.needsResize) this.resize();
    this.camera.update(dt, c.sim.frame.c);
    this.renderer.render(this.frameState());
  }
  resize() {
    this.needsResize = false;
    const r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    let w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    const max = 2.6e6;
    if (w * h > max) { const s = Math.sqrt(max / (w * h)); w = Math.round(w * s); h = Math.round(h * s); }
    const lim = this.renderer.device.limits.maxTextureDimension2D;
    this.renderer.resize(Math.min(w, lim), Math.min(h, lim));
  }
  frameState() {
    const c = this.creature, def = this.def, env = c.env.cfg, b = c.env.bounds, [w, h] = this.renderer.size;
    const cam = this.camera.matrices(w / h);
    if (this.materialPreset !== c.presetId) { this.materialTable = Engine.buildMaterialTable(def, c.presetId); this.materialPreset = c.presetId; }
    const center = [(b.min[0] + b.max[0]) / 2, b.min[1], (b.min[2] + b.max[2]) / 2];
    const lightView = mat4.lookAt(v3.add(center, v3.scale(LIGHTS.key, 14)), center, [0, 1, 0]);
    const lightViewProj = mat4.mul(mat4.ortho(-6, 6, -6, 6, 1, 30), lightView);
    const bg = env.background, bgLin = hexToLinear(bg), bgS = [1, 3, 5].map((i) => parseInt(bg.slice(i, i + 2), 16) / 255);
    const fur = def.fur && c.fur ? [def.fur.length * c.fur.pile, c.fur.pile > 0.02 ? def.fur.shells : 0, def.fur.density, def.fur.droop] : [0, 0, 0, 0];
    return {
      viewProj: cam.viewProj, lightViewProj, eye: cam.eye, time: c.env.time,
      keyDir: LIGHTS.key, shadowStrength: 0.85, keyColor: LIGHTS.keyColor, fillDir: LIGHTS.fill, fillColor: LIGHTS.fillColor,
      skyColor: LIGHTS.sky, groundColor: LIGHTS.ground, bgColor: bgLin, bgClear: bgS, floorColor: hexToLinear(env.floor),
      fur, env: [b.min[1], env.floorPattern === 'caustics' ? 1 : 0, env.fadeRadius, 0], bounds: [b.min[0], b.min[2], b.max[0], b.max[2]],
      materials: this.materialTable, creature: c, debugMesh: this.debugMesh,
      items: c.env.items.map((it) => ({ type: it.type, matrix: c.env.itemMatrix(it) })),
    };
  }
  refreshUI() {
    const c = this.creature;
    this.refreshGlobal?.();
    if (!c) return;
    this.refreshSpecies?.();
    c.updateMetrics();
    const rctx = { metrics: c.metrics, locomotion: c.loco.readouts, state: c.state };
    for (const { r, span } of this.readoutEls || []) {
      const v = r.get(rctx);
      span.textContent = r.format ? r.format(v) : String(v);
    }
    const st = $('status');
    st.classList.toggle('paused', this.paused);
    $('stateLabel').textContent = !this.renderer ? 'Renderer unavailable' : this.paused ? 'Paused' : c.machine.label() + (this.quarter ? ' · ¼×' : '');
    const A = c.assets, sh = def => (def.fur && c.fur.pile > 0.02 ? def.fur.shells : 0);
    $('engineLine').textContent = `lattice ${A.lattice.n} particles · ${A.lattice.nc} hexes · ${A.lattice.tets.length / 4} tets · ${(A.triangles / 1000).toFixed(1)}k triangles × ${1 + sh(this.def)} layers · ${this.stepMs.toFixed(1)} ms / step`;
    const hint = $('hint'), T = Engine.TOOLS[this.toolId];
    if (this.hover && this.hover.regions.length && !this.mode) {
      const labels = this.def.interactions.regions.filter((r) => this.hover.regions.includes(r.id)).map((r) => r.label);
      hint.textContent = `${labels[0]} · ${T.label}`;
    } else hint.textContent = performance.now() < this.toolHintUntil ? T.hint : '';
  }
}

const app = new App();
window.plushApp = app;
app.start().catch((err) => app.fail(err));
})();
