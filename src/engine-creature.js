
/* ---------- 7. Material & fur system ----------------------------------- */
const MATERIAL_TYPES = { plush: 0, felt: 1, glass: 2, embroidery: 3, stitch: 4 };
const MATERIAL_DEFAULTS = {
  plush: { fur: 1, sheen: 0.6, gloss: 0 },
  felt: { fur: 0, sheen: 0.3, gloss: 0 },
  glass: { fur: 0, sheen: 0, gloss: 1 },
  embroidery: { fur: 0, sheen: 0.15, gloss: 0.45 },
  stitch: { fur: 0, sheen: 0.1, gloss: 0.3 },
};
const MAX_REGIONS = 16, MAX_DECALS = 128, MAX_GROUPS = 32, MAX_ATTACH = 4, MAX_SPOTS = 4;

/** Region table (16 × 8 floats) for a fabric preset: colour, type, fur multiplier, sheen, gloss. */
function buildMaterialTable(def, presetId) {
  const regions = def.materials.regions, names = Object.keys(regions);
  const preset = def.materials.presets.find((p) => p.id === presetId) || def.materials.presets[0];
  const out = new Float32Array(MAX_REGIONS * 8);
  names.forEach((name, i) => {
    const r = regions[name], base = MATERIAL_DEFAULTS[r.type];
    const col = hexToLinear(preset.colors[name] || '#ff00ff');
    out.set([col[0], col[1], col[2], MATERIAL_TYPES[r.type], r.fur ?? base.fur, r.sheen ?? base.sheen, r.gloss ?? base.gloss, 0], i * 8);
  });
  return out;
}

/* ---------- 8. Asset building: anatomy → meshes, lattice, embeddings ---- */
/**
 * Converts a species' procedural anatomy into GPU-ready geometry (separate
 * from simulation state, so it can be cached and re-used across resets).
 */
function buildCreatureAssets(def) {
  const A = def.anatomy, t0 = performance.now();
  const regionNames = Object.keys(def.materials.regions);
  const regionIndex = (name) => {
    const i = regionNames.indexOf(name);
    if (i < 0) throw new Error(`Region "${name}" is not declared in materials.regions`);
    return i;
  };
  const parts = A.parts.map((part) => {
    const mesh = surfaceNets(part.sdf, part.bounds.min, part.bounds.max, part.resolution);
    const P = mesh.positions, nv = P.length / 3, r0 = new Uint32Array(nv);
    // Output vertices reference a mesh vertex (src) and carry a region pair + signed boundary field.
    const src = [], reg = [], field = [], I = Array.from(mesh.indices);
    for (let v = 0; v < nv; v++) { r0[v] = regionIndex(part.region(P[3 * v], P[3 * v + 1], P[3 * v + 2])); src.push(v); reg.push(r0[v] | (r0[v] << 8)); field.push(1); }
    // Triangles straddling a species boundary get private vertex copies holding that boundary's
    // field, so the fragment shader can resolve the seam exactly inside the triangle.
    const bds = part.boundaries || [], pick = (r, x, y, z) => regionIndex(typeof r === 'function' ? r(x, y, z) : r);
    if (bds.length) for (let t = 0; t < I.length; t += 3) {
      const ia = I[t], ib = I[t + 1], ic = I[t + 2];
      const cx = (P[3 * ia] + P[3 * ib] + P[3 * ic]) / 3, cy = (P[3 * ia + 1] + P[3 * ib + 1] + P[3 * ic + 1]) / 3, cz = (P[3 * ia + 2] + P[3 * ib + 2] + P[3 * ic + 2]) / 3;
      let best = null, score = 1;
      for (const bd of bds) {
        if (bd.where && !bd.where(cx, cy, cz)) continue;
        const sc = Math.abs(bd.field(cx, cy, cz)) / (bd.band ?? 0.05);
        if (sc < score) { score = sc; best = bd; }
      }
      if (!best) continue;
      const f3 = [ia, ib, ic].map((v) => best.field(P[3 * v], P[3 * v + 1], P[3 * v + 2]));
      const crosses = !((f3[0] >= 0) === (f3[1] >= 0) && (f3[1] >= 0) === (f3[2] >= 0));
      if (!crosses && r0[ia] === r0[ib] && r0[ib] === r0[ic]) continue;
      const packed = pick(best.positive, cx, cy, cz) | (pick(best.negative, cx, cy, cz) << 8);
      [ia, ib, ic].forEach((v, k) => { I[t + k] = src.length; src.push(v); reg.push(packed); field.push(f3[k]); });
    }
    return { part, mesh, src, reg, field, indices: I };
  });
  // Lattice from all lattice-skinned parts.
  const latticeParts = parts.filter((p) => !p.part.attach);
  const latVerts = new Float32Array(latticeParts.reduce((s, p) => s + p.mesh.positions.length, 0));
  let off = 0;
  for (const p of latticeParts) { latVerts.set(p.mesh.positions, off); off += p.mesh.positions.length; }
  const physSdf = A.physicsSdf || latticeParts[0].part.sdf;
  const lattice = buildLattice(physSdf, latVerts, def.physics.latticeSpacing, def.physics.stuffingAt);
  // Articulated joints & rigid attachments (anchor frame read from the lattice).
  const joints = Object.entries(A.joints || {}).map(([id, j]) => ({ id, ...j }));
  const attachments = [];
  for (const p of parts) if (p.part.attach) {
    const key = `${p.part.attach.anchor}|${p.part.attach.joint || ''}`;
    let k = attachments.findIndex((a) => a.key === key);
    if (k < 0) {
      const anchor = A.anchors[p.part.attach.anchor];
      attachments.push({ key, anchor, joint: p.part.attach.joint ? joints.findIndex((j) => j.id === p.part.attach.joint) : -1, emb: lattice.embed(...anchor) });
      k = attachments.length - 1;
    }
    p.attachIndex = k + 1;
  }
  // Decals (embroidery / stitches) flattened to capsule segments grouped spatially.
  const decalSpecs = A.decals ? A.decals({ surfacePoint, sdf: physSdf, v3 }) : [];
  const groupNames = [], segs = [];
  for (const d of decalSpecs) {
    let g = groupNames.indexOf(d.group);
    if (g < 0) { groupNames.push(d.group); g = groupNames.length - 1; }
    let along = 0;
    for (let i = 0; i + 1 < d.points.length; i++) {
      const a = d.points[i], b = d.points[i + 1];
      segs.push({ g, a, b, width: d.width, region: regionIndex(d.region), dash: d.dash || null, cuts: d.cutsFur !== false ? 1 : 0, along });
      along += v3.dist(a, b);
    }
  }
  segs.sort((s, t) => s.g - t.g);
  if (segs.length > MAX_DECALS) throw new Error(`Too many decal segments (${segs.length} > ${MAX_DECALS})`);
  if (groupNames.length > MAX_GROUPS) throw new Error('Too many decal groups');
  const groups = groupNames.map((_, g) => {
    const list = segs.filter((s) => s.g === g), start = segs.indexOf(list[0]);
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const s of list) for (const p of [s.a, s.b]) for (let a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], p[a] - s.width); mx[a] = Math.max(mx[a], p[a] + s.width); }
    return { start, count: list.length, mn, mx, list };
  });
  const segDist2 = (p, a, b) => {
    const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    const t = clamp(((p[0] - a[0]) * abx + (p[1] - a[1]) * aby + (p[2] - a[2]) * abz) / (abx * abx + aby * aby + abz * abz + 1e-12), 0, 1);
    return (p[0] - a[0] - abx * t) ** 2 + (p[1] - a[1] - aby * t) ** 2 + (p[2] - a[2] - abz * t) ** 2;
  };
  // Interleaved vertex buffer: restPos, restNrm, uvw, (cell, attach, regions, decalMask), furDir + boundary field — 68 bytes.
  const total = parts.reduce((n, p) => n + p.src.length, 0);
  const vbuf = new ArrayBuffer(total * 68), vf = new Float32Array(vbuf), vu = new Uint32Array(vbuf);
  const furDir = new Float32Array(total * 3), restPos = new Float32Array(total * 3), restNrm = new Float32Array(total * 3);
  const embCell = new Int32Array(total).fill(-1), embUVW = new Float32Array(total * 3);
  const indices = [];
  let base = 0;
  for (const p of parts) {
    const { positions: P, normals: N } = p.mesh, margin = p.part.resolution * 2.5;
    for (const i of p.indices) indices.push(i + base);
    for (let k = 0; k < p.src.length; k++) {
      const v = p.src[k], gv = base + k, o = gv * 17, x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
      const nx = N[3 * v], ny = N[3 * v + 1], nz = N[3 * v + 2];
      vf[o] = x; vf[o + 1] = y; vf[o + 2] = z; vf[o + 3] = nx; vf[o + 4] = ny; vf[o + 5] = nz;
      restPos.set([x, y, z], 3 * gv); restNrm.set([nx, ny, nz], 3 * gv);
      if (!p.part.attach) {
        const e = lattice.embed(x, y, z);
        vf[o + 6] = e.u; vf[o + 7] = e.v; vf[o + 8] = e.w; vu[o + 9] = e.cell; vu[o + 10] = 0;
        embCell[gv] = e.cell; embUVW.set([e.u, e.v, e.w], 3 * gv);
      } else { vu[o + 9] = 0; vu[o + 10] = p.attachIndex; }
      vu[o + 11] = p.reg[k]; vf[o + 16] = p.field[k];
      let mask = 0;
      groups.forEach((g, gi) => {
        if (x < g.mn[0] - margin || y < g.mn[1] - margin || z < g.mn[2] - margin || x > g.mx[0] + margin || y > g.mx[1] + margin || z > g.mx[2] + margin) return;
        for (const s of g.list) if (segDist2([x, y, z], s.a, s.b) < (s.width + margin) ** 2) { mask |= 1 << gi; break; }
      });
      vu[o + 12] = mask;
      if (def.fur && p.part.fur !== false) {
        const d = def.fur.direction(x, y, z, nx, ny, nz), dn = d[0] * nx + d[1] * ny + d[2] * nz;
        const t = v3.norm([d[0] - nx * dn, d[1] - ny * dn, d[2] - nz * dn]);
        furDir.set(t, 3 * gv); vf[o + 13] = t[0]; vf[o + 14] = t[1]; vf[o + 15] = t[2];
      }
    }
    base += p.src.length;
  }
  // Decal uniform payload.
  const decalData = new Float32Array(MAX_DECALS * 12), groupData = new Uint32Array(MAX_GROUPS * 4);
  segs.forEach((s, i) => decalData.set([...s.a, s.width, ...s.b, s.region, s.dash ? s.dash.period : 0, s.dash ? s.dash.duty : 1, s.cuts, s.along], i * 12));
  groups.forEach((g, i) => groupData.set([g.start, g.count, 0, 0], i * 4));
  // Expression spots.
  const channels = A.expressions?.channels || [];
  const spotData = new Float32Array(MAX_SPOTS * 8);
  (A.expressions?.spots || []).slice(0, MAX_SPOTS).forEach((s, i) => {
    const c = hexToLinear(s.color);
    spotData.set([...s.center, s.radius, c[0], c[1], c[2], channels.indexOf(s.channel)], i * 8);
  });
  // Landmarks embedded in the lattice.
  const landmarks = {};
  for (const [k, p] of Object.entries(A.landmarks || {})) landmarks[k] = lattice.embed(...p);
  // Picking list (lattice-skinned vertices) and a spatial hash for combing.
  const pick = [];
  for (let v = 0; v < total; v++) if (embCell[v] >= 0) pick.push(v);
  const HC = 0.08, hash = new Map();
  for (const v of pick) {
    const key = `${Math.floor(restPos[3 * v] / HC)},${Math.floor(restPos[3 * v + 1] / HC)},${Math.floor(restPos[3 * v + 2] / HC)}`;
    if (!hash.has(key)) hash.set(key, []);
    hash.get(key).push(v);
  }
  // Prop meshes for environment items owned by the species.
  const items = {};
  for (const [id, it] of Object.entries(def.items || {})) {
    const m = surfaceNets(it.sdf, it.bounds.min, it.bounds.max, it.resolution), nv = m.positions.length / 3;
    const data = new Float32Array(nv * 9);
    for (let v = 0; v < nv; v++) {
      const c = hexToLinear(it.color(m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]));
      data.set([m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2], m.normals[3 * v], m.normals[3 * v + 1], m.normals[3 * v + 2], ...c], v * 9);
    }
    items[id] = { data, indices: m.indices };
  }
  return {
    lattice, joints, attachments, vbuf, indices: new Uint32Array(indices), vertexCount: total, furDir, restPos, restNrm,
    embCell, embUVW, decalData, groupData, decalCount: segs.length, spotData, channels, landmarks, pick, hash, HC, items,
    regionNames, buildMs: performance.now() - t0, triangles: indices.length / 3,
  };
}

/* ---------- 9. Interaction tools (species-agnostic) -------------------- */
/**
 * Tools only know how to apply generic physical effects and emit generic
 * events (grab / release / poke / stroke / comb). What a creature *does* about
 * an event is decided by the species' reaction rules.
 */
const TOOLS = {
  hand: {
    label: 'Hand', glyph: '✋', requires: null,
    hint: 'Drag the plush to grab and move it. Drag empty space to orbit.',
    begin(c, hit, ray) {
      if (!c.sim.startGrab(hit.point, c.sim.lat.H * 1.7)) return false;
      this.plane = { p: hit.point.slice(), n: v3.scale(ray.view, -1) };
      c.interaction.grabbed = true; c.interaction.regions = hit.regions; c.interaction.point = hit.point;
      c.emit('grab', { regions: hit.regions, point: hit.point });
      return true;
    },
    move(c, hit, ray) {
      const { p, n } = this.plane, den = v3.dot(ray.d, n);
      if (Math.abs(den) < 1e-5 || !c.sim.grab) return;
      const t = v3.dot(v3.sub(p, ray.o), n) / den, target = v3.add(ray.o, v3.scale(ray.d, t));
      c.sim.grab.target = clampToBounds(target, c.env.bounds, 0.05);
      c.interaction.point = c.sim.grab.target;
    },
    end(c) {
      c.sim.grab = null; c.interaction.grabbed = false;
      c.emit('release', { regions: c.interaction.regions, point: c.interaction.point });
    },
  },
  finger: {
    label: 'Finger', glyph: '☝', requires: null,
    hint: 'Tap to poke or boop. Drag across the plush to pet it.',
    begin(c, hit, ray) {
      c.poke(hit.point, ray.d, 1.6, c.sim.lat.H * 2.2);
      c.emit('poke', { regions: hit.regions, point: hit.point });
      this.last = 0;
      return true;
    },
    move(c, hit, ray) {
      if (!hit || c.time - this.last < 0.12) return;
      this.last = c.time;
      c.poke(hit.point, ray.d, 0.35, c.sim.lat.H * 2);
      c.emit('stroke', { regions: hit.regions, point: hit.point });
    },
    end() {},
  },
  comb: {
    label: 'Comb', glyph: '≋', requires: 'fur',
    hint: 'Drag across the pile to brush it. Smooth fur resets the nap.',
    begin(c, hit) { this.prev = hit ? hit.point : null; this.last = -1; return true; },
    move(c, hit) {
      if (!hit) { this.prev = null; return; }
      if (this.prev) {
        const d = v3.sub(hit.point, this.prev);
        if (v3.len(d) > 0.004) c.combAt(hit, v3.norm(d), 0.55);
      }
      this.prev = hit.point;
      if (c.time - this.last > 0.25) { this.last = c.time; c.emit('comb', { regions: hit.regions, point: hit.point }); }
    },
    end() { this.prev = null; },
  },
};

/** Actions every species can bind to without defining them. */
const ENGINE_ACTIONS = {
  'engine.smoothFur': { label: 'Smooth fur', requires: 'fur', run: (c) => c.smoothFur() },
};

/* ---------- 10. Creature: assembles all systems from a definition ------- */
class Creature {
  constructor(def, assets, opts = {}) {
    this.def = def; this.assets = assets; this.seed = opts.seed || 1;
    this.rng = makeRng(this.seed);
    const envCfg = { ...ENVIRONMENTS[def.environment.preset], ...(def.environment.overrides || {}) };
    this.env = new Environment(envCfg, def.items, this.rng);
    this.sim = new SoftBody(assets.lattice, def.physics, this.spawnTransform());
    this.loco = Locomotion.create(def.locomotion.type, this, def.locomotion);
    this.joints = assets.joints.map((j) => ({ ...j, angle: 0, vel: 0 }));
    this.expr = new Float32Array(4);
    this.flags = {}; this.paramValues = {};
    this.state = def.createState ? def.createState() : {};
    this.fur = def.fur ? { pile: 1, comb: new Float32Array(assets.furDir), dirty: true } : null;
    this.lag = new Float64Array(3 * this.sim.n); this.lagV = new Float64Array(3 * this.sim.n);
    this.attachQ = new Float64Array(4 * MAX_ATTACH);
    for (let k = 0; k < MAX_ATTACH; k++) this.attachQ[4 * k + 3] = 1;
    this.attachMats = new Float32Array(16 * MAX_ATTACH);
    this.interaction = { grabbed: false, regions: [], point: null };
    this.pokes = []; this.time = 0;
    this.presetId = def.materials.defaultPreset;
    this.metrics = { mass: 0, volume: 0 };
    this.ctx = this.makeContext();
    this.machine = new BehaviorMachine(def.behaviors, this.ctx);
    this.ctx.behaviors = this.machine;
    for (const c of def.controls) if (c.bind && c.default !== undefined) this.setParam(c.bind, c.default);
    this.updateMetrics();
  }
  spawnTransform() { return this.def.environment.spawn || { pos: this.env.api.center, yaw: 0 }; }
  makeContext() {
    const self = this;
    return {
      time: 0, dt: 0, intent: null, rand: () => self.rng(),
      flags: this.flags, state: this.state, env: this.env.api, interaction: this.interaction,
      get body() { return self.sim.frame; },
      get locomotion() { return self.loco.readouts; },
      landmark: (name) => self.landmarkWorld(name),
      joint: (id) => self.joints.find((j) => j.id === id),
      behaviors: null,
    };
  }
  freshIntent() { return Object.assign(this.loco.defaultIntent(), { joints: {}, expr: {} }); }
  setParam(path, value) {
    const [ns, key] = path.split('.');
    if (ns === 'physics' && key === 'stuffing') this.sim.setStuffing(value);
    else if (ns === 'fur' && this.fur) this.fur[key] = value;
    else if (ns === 'locomotion') this.loco.params[key] = value;
    else if (ns === 'flags') this.flags[key] = value;
    else if (ns === 'material' && key === 'preset') this.presetId = value;
    this.paramValues[path] = value;
  }
  getParam(path) { return this.paramValues[path]; }
  emit(type, info) {
    for (const rule of this.def.interactions.reactions) {
      if (rule.on !== type) continue;
      if (rule.region && !(info.regions || []).includes(rule.region)) continue;
      if (rule.when && !rule.when(this.ctx, info)) continue;
      if (this.machine.react(rule.behavior, info)) return rule;
    }
    return null;
  }
  poke(point, dir, strength, radius) { this.pokes.push({ point, dir: v3.norm(dir), strength, radius }); }
  step(dt) {
    const ctx = this.ctx;
    this.time += dt; ctx.time = this.time; ctx.dt = dt;
    ctx.intent = this.freshIntent();
    this.machine.tick(dt);
    const I = ctx.intent;
    for (const j of this.joints) {
      const target = j.min + (j.max - j.min) * clamp(I.joints[j.id] ?? 0, 0, 1);
      j.vel += (j.stiffness * (target - j.angle) - j.damping * j.vel) * dt;
      j.angle += j.vel * dt;
      const lo = Math.min(j.min, j.max), hi = Math.max(j.min, j.max);
      if (j.angle < lo) { j.angle = lo; j.vel = 0; } else if (j.angle > hi) { j.angle = hi; j.vel = 0; }
    }
    this.assets.channels.forEach((ch, i) => {
      const t = I.expr[ch] ?? 0;
      this.expr[i] += (t - this.expr[i]) * damp(t > this.expr[i] ? 7 : 1.1, dt);
    });
    this.loco.pose(dt, I, this.sim.posed);
    this.loco.control(dt, I);
    const g = this.env.cfg.gravity * (1 - this.env.cfg.buoyancy), sim = this.sim;
    sim.step(dt, this.env, (sdt, s) => {
      const v = sim.v;
      if (g) for (let p = 1; p < v.length; p += 3) v[p] += g * sdt;
      if (this.env.cfg.medium === 'air') { const k = Math.exp(-0.15 * sdt); for (let p = 0; p < v.length; p++) v[p] *= k; }
      this.loco.applyForces(sdt, s);
      if (s === 0 && this.pokes.length) {
        for (const pk of this.pokes) {
          for (let i = 0; i < sim.n; i++) {
            const d = Math.hypot(sim.x[3 * i] - pk.point[0], sim.x[3 * i + 1] - pk.point[1], sim.x[3 * i + 2] - pk.point[2]);
            if (d > pk.radius) continue;
            const k = pk.strength * (1 - d / pk.radius) ** 2;
            v[3 * i] += pk.dir[0] * k; v[3 * i + 1] += pk.dir[1] * k; v[3 * i + 2] += pk.dir[2] * k;
          }
        }
        this.pokes.length = 0;
      }
    });
    this.env.step(dt);
    this.updateFurLag(dt);
  }
  updateFurLag(dt) {
    if (!this.fur) return;
    const f = this.def.fur, v = this.sim.v, vp = this.sim.vPrev, L = this.lag, LV = this.lagV;
    const k = f.lagStiffness, c = 2 * Math.sqrt(k) * f.lagDamping;
    for (let i = 0; i < L.length; i++) {
      const target = -((v[i] - vp[i]) / dt) * f.inertia - v[i] * f.flow;
      LV[i] += (k * (target - L[i]) - c * LV[i]) * dt;
      L[i] = clamp(L[i] + LV[i] * dt, -1.2, 1.2);
    }
  }
  /** Rigid attachment matrices: lattice frame at the anchor × joint rotation. */
  computeAttachments() {
    const P = [0, 0, 0], Fm = new Float64Array(9), cells = this.sim.lat.cells;
    this.assets.attachments.forEach((att, k) => {
      const e = att.emb;
      ffd(this.sim.x, cells, e.cell, e.u, e.v, e.w, P, Fm);
      extractRotation(Fm, this.attachQ, 4 * k, 6);
      const R = quatToMat3(this.attachQ, new Float64Array(9), 4 * k);
      let L = R;
      if (att.joint >= 0) {
        const j = this.joints[att.joint], J = quatToMat3(quatAxisAngle(v3.norm(j.axis), j.angle));
        L = new Float64Array(9);
        for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) L[c * 3 + r] = R[r] * J[c * 3] + R[3 + r] * J[c * 3 + 1] + R[6 + r] * J[c * 3 + 2];
      }
      const a = att.anchor, M = this.attachMats, o = 16 * k;
      for (let c = 0; c < 3; c++) { M[o + 4 * c] = L[3 * c]; M[o + 4 * c + 1] = L[3 * c + 1]; M[o + 4 * c + 2] = L[3 * c + 2]; M[o + 4 * c + 3] = 0; }
      for (let r = 0; r < 3; r++) M[o + 12 + r] = P[r] - (L[r] * a[0] + L[3 + r] * a[1] + L[6 + r] * a[2]);
      M[o + 15] = 1;
    });
  }
  landmarkWorld(name) {
    const e = this.assets.landmarks[name], P = [0, 0, 0];
    if (!e) return this.sim.frame.c.slice();
    ffd(this.sim.x, this.sim.lat.cells, e.cell, e.u, e.v, e.w, P, null);
    return P;
  }
  regionsAt(p) {
    const out = [];
    for (const r of this.def.interactions.regions) {
      let hit = false;
      if (r.sphere) hit = Math.hypot(p[0] - r.sphere[0], p[1] - r.sphere[1], p[2] - r.sphere[2]) < r.sphere[3];
      else if (r.box) hit = p[0] >= r.box.min[0] && p[0] <= r.box.max[0] && p[1] >= r.box.min[1] && p[1] <= r.box.max[1] && p[2] >= r.box.min[2] && p[2] <= r.box.max[2];
      else if (r.test) hit = r.test(p[0], p[1], p[2]);
      if (hit) out.push(r.id);
    }
    return out;
  }
  /** Ray vs skinned render vertices (CPU FFD of the same embedding the GPU uses). */
  raycast(o, d) {
    const A = this.assets, x = this.sim.x, cells = this.sim.lat.cells, P = [0, 0, 0], r2 = (this.def.interactions.pickRadius || 0.035) ** 2;
    let best = -1, bt = Infinity, bp = null;
    for (let k = 0; k < A.pick.length; k++) {
      const v = A.pick[k];
      ffd(x, cells, A.embCell[v], A.embUVW[3 * v], A.embUVW[3 * v + 1], A.embUVW[3 * v + 2], P, null);
      const px = P[0] - o[0], py = P[1] - o[1], pz = P[2] - o[2], t = px * d[0] + py * d[1] + pz * d[2];
      if (t < 0 || t > bt) continue;
      if (px * px + py * py + pz * pz - t * t < r2) { bt = t; best = v; bp = P.slice(); }
    }
    if (best < 0) return null;
    const rest = [A.restPos[3 * best], A.restPos[3 * best + 1], A.restPos[3 * best + 2]];
    return { point: bp, rest, vertex: best, t: bt, regions: this.regionsAt(rest) };
  }
  combAt(hit, dirWorld, strength) {
    if (!this.fur) return;
    const R = this.sim.frame.R, A = this.assets, comb = this.fur.comb, rad = this.def.fur.combRadius || 0.09;
    const dr = [R[0] * dirWorld[0] + R[1] * dirWorld[1] + R[2] * dirWorld[2], R[3] * dirWorld[0] + R[4] * dirWorld[1] + R[5] * dirWorld[2],
      R[6] * dirWorld[0] + R[7] * dirWorld[1] + R[8] * dirWorld[2]];
    const p = hit.rest, HC = A.HC, cr = Math.ceil(rad / HC);
    const ci = [Math.floor(p[0] / HC), Math.floor(p[1] / HC), Math.floor(p[2] / HC)];
    for (let a = -cr; a <= cr; a++) for (let b = -cr; b <= cr; b++) for (let c = -cr; c <= cr; c++) {
      const list = A.hash.get(`${ci[0] + a},${ci[1] + b},${ci[2] + c}`);
      if (!list) continue;
      for (const v of list) {
        const d = Math.hypot(A.restPos[3 * v] - p[0], A.restPos[3 * v + 1] - p[1], A.restPos[3 * v + 2] - p[2]);
        if (d > rad || A.furDir[3 * v] === 0 && A.furDir[3 * v + 1] === 0 && A.furDir[3 * v + 2] === 0) continue;
        const nx = A.restNrm[3 * v], ny = A.restNrm[3 * v + 1], nz = A.restNrm[3 * v + 2], dn = dr[0] * nx + dr[1] * ny + dr[2] * nz;
        const t = v3.norm([dr[0] - nx * dn, dr[1] - ny * dn, dr[2] - nz * dn]), wgt = strength * (1 - d / rad);
        const m = v3.norm([comb[3 * v] * (1 - wgt) + t[0] * wgt, comb[3 * v + 1] * (1 - wgt) + t[1] * wgt, comb[3 * v + 2] * (1 - wgt) + t[2] * wgt]);
        comb.set(m, 3 * v);
      }
    }
    this.fur.dirty = true;
  }
  smoothFur() { if (this.fur) { this.fur.comb.set(this.assets.furDir); this.fur.dirty = true; } }
  /** Grain alignment of the comb field (1 = smooth nap) — a live read-out. */
  napAlignment() {
    if (!this.fur) return 1;
    const c = this.fur.comb, f = this.assets.furDir;
    let s = 0, n = 0;
    for (let i = 0; i < c.length; i += 9) { const d = c[i] * f[i] + c[i + 1] * f[i + 1] + c[i + 2] * f[i + 2]; if (f[i] || f[i + 1] || f[i + 2]) { s += d; n++; } }
    return n ? s / n : 1;
  }
  updateMetrics() {
    const u = this.def.physics.unitMeters, V = this.sim.measureVolume() * u ** 3 * 1000; // litres
    this.metrics.volume = V;
    this.metrics.mass = V * this.def.physics.fillGramsPerLitre * this.sim.stuffing + this.def.physics.shellGrams;
  }
  reset() {
    this.rng = makeRng(this.seed); this.env.rng = this.rng; this.env.clear();
    this.sim.reset(this.spawnTransform()); this.loco.reset();
    this.joints.forEach((j) => { j.angle = 0; j.vel = 0; });
    this.expr.fill(0); this.lag.fill(0); this.lagV.fill(0);
    Object.keys(this.state).forEach((k) => delete this.state[k]);
    if (this.def.createState) Object.assign(this.state, this.def.createState());
    this.interaction.grabbed = false; this.time = 0; this.ctx.time = 0;
    this.machine.reset();
    this.smoothFur();
  }
}

/* ---------- 11. Species registry & validation -------------------------- */
const registry = [];
const BEHAVIOR_KINDS = ['autonomous', 'user', 'reactive', 'environment'];
const CONTROL_TYPES = ['button', 'toggle', 'slider', 'select', 'preset'];
const BIND_NAMESPACES = ['physics', 'fur', 'locomotion', 'flags', 'material'];

/**
 * Static validation of a SpeciesDefinition against engine capabilities. It
 * deliberately knows nothing about any animal; it checks references only.
 */
function validateSpecies(def) {
  const errors = [], warnings = [];
  let checks = 0;
  const req = (cond, msg) => { checks++; if (!cond) errors.push(msg); };
  req(def.id && def.name && def.category, 'identity: id, name and category are required');
  if (def.status === 'planned') return { ok: errors.length === 0, errors, warnings, checks, planned: true };
  req(def.display && Array.isArray(def.display.title) && def.display.title.length === 2, 'display.title must be two lines');
  req(ENVIRONMENTS[def.environment?.preset], `environment preset "${def.environment?.preset}" is not registered`);
  const A = def.anatomy || {};
  req(Array.isArray(A.parts) && A.parts.length, 'anatomy.parts must be non-empty');
  req((A.parts || []).some((p) => !p.attach), 'anatomy needs at least one lattice-skinned part');
  const regions = Object.keys(def.materials?.regions || {});
  req(regions.length > 0 && regions.length <= MAX_REGIONS, `materials.regions must declare 1–${MAX_REGIONS} regions`);
  for (const r of regions) req(MATERIAL_TYPES[def.materials.regions[r].type] !== undefined, `region "${r}" has unknown material type`);
  for (const p of A.parts || []) {
    req(typeof p.sdf === 'function' && typeof p.region === 'function', `part "${p.id}": sdf() and region() required`);
    req(p.bounds && p.resolution > 0, `part "${p.id}": bounds and resolution required`);
    if (p.attach) {
      req(A.anchors && A.anchors[p.attach.anchor], `part "${p.id}": anchor "${p.attach.anchor}" missing`);
      if (p.attach.joint) req(A.joints && A.joints[p.attach.joint], `part "${p.id}": joint "${p.attach.joint}" missing`);
    }
  }
  for (const [id, j] of Object.entries(A.joints || {})) req(j.axis && j.stiffness > 0 && j.damping >= 0, `joint "${id}": axis/stiffness/damping`);
  for (const s of A.expressions?.spots || []) req((A.expressions.channels || []).includes(s.channel), `expression spot uses unknown channel "${s.channel}"`);
  req((A.expressions?.channels || []).length <= 4, 'at most 4 expression channels');
  for (const p of def.materials?.presets || []) for (const r of regions) req(p.colors[r], `preset "${p.id}" lacks colour for region "${r}"`);
  req(def.materials?.presets?.some((p) => p.id === def.materials.defaultPreset), 'materials.defaultPreset must exist');
  const P = def.physics || {};
  req(P.latticeSpacing > 0 && Array.isArray(P.shapeStiffness) && Array.isArray(P.volumeCompliance) && Array.isArray(P.inflate), 'physics: latticeSpacing, shapeStiffness, volumeCompliance, inflate');
  req(P.unitMeters > 0 && P.fillGramsPerLitre > 0, 'physics: unitMeters and fillGramsPerLitre for read-outs');
  req(Locomotion.registry[def.locomotion?.type], `locomotion "${def.locomotion?.type}" is not a registered capability`);
  const B = def.behaviors || {}, states = B.states || {};
  req(states[B.initial], `initial behaviour "${B.initial}" missing`);
  for (const [id, s] of Object.entries(states)) {
    req(BEHAVIOR_KINDS.includes(s.kind), `behaviour "${id}": kind must be one of ${BEHAVIOR_KINDS.join(', ')}`);
    req(typeof s.priority === 'number', `behaviour "${id}": numeric priority`);
  }
  const I = def.interactions || {}, regionIds = (I.regions || []).map((r) => r.id);
  for (const t of I.tools || []) {
    req(TOOLS[t], `tool "${t}" is not provided by the engine`);
    if (TOOLS[t]?.requires === 'fur') req(!!def.fur, `tool "${t}" requires a fur definition`);
  }
  for (const r of I.reactions || []) {
    req(states[r.behavior], `reaction on "${r.on}" targets unknown behaviour "${r.behavior}"`);
    if (r.region) req(regionIds.includes(r.region), `reaction references unknown region "${r.region}"`);
  }
  if (def.fur) req(typeof def.fur.direction === 'function' && def.fur.shells > 0, 'fur: direction() and shells');
  for (const c of def.controls || []) {
    req(c.id && c.label && CONTROL_TYPES.includes(c.type), `control "${c.id}": id, label and supported type`);
    if (c.action) req((def.actions && def.actions[c.action]) || ENGINE_ACTIONS[c.action], `control "${c.id}": unknown action "${c.action}"`);
    if (ENGINE_ACTIONS[c.action]?.requires === 'fur') req(!!def.fur, `control "${c.id}" needs fur`);
    if (c.bind) req(BIND_NAMESPACES.includes(c.bind.split('.')[0]), `control "${c.id}": bind namespace must be one of ${BIND_NAMESPACES.join(', ')}`);
    if (c.type === 'slider') req(c.min < c.max && c.default >= c.min && c.default <= c.max, `slider "${c.id}": range/default`);
    if (c.bind === 'fur.pile') req(!!def.fur, `slider "${c.id}" binds fur but species has none`);
  }
  for (const r of def.readouts || []) req(r.id && r.label && typeof r.get === 'function', `readout "${r.id}": label and get()`);
  for (const [id, it] of Object.entries(def.items || {})) req(typeof it.sdf === 'function' && it.radius > 0, `item "${id}": sdf and radius`);
  if (!def.notes) warnings.push('no technical notes');
  return { ok: errors.length === 0, errors, warnings, checks };
}

function registerSpecies(def) {
  def.validation = validateSpecies(def);
  registry.push(def);
  if (!def.validation.ok) console.warn(`Species "${def.id}" failed validation`, def.validation.errors);
  return def;
}
