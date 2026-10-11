
/* ---------- 4. Locomotion capabilities --------------------------------- */
/**
 * A locomotion capability turns a behaviour "intent" into (a) a posed rest
 * shape for the lattice (active muscles: shape matching pulls the body toward
 * it) and (b) velocity-level forces (thrust, torques, drag). Species pick one
 * by name and tune it through their definition.
 */
const Locomotion = {
  registry: {},
  register(type, factory) { this.registry[type] = factory; },
  create(type, creature, cfg) {
    const f = this.registry[type];
    if (!f) throw new Error(`Unknown locomotion capability "${type}"`);
    return f(creature, cfg);
  },
};

/** Shared helper: distribute an angular acceleration α about c as particle velocity change. */
function applySpin(sim, alpha, sdt) {
  const x = sim.x, v = sim.v, c = sim.frame.c, ax = alpha[0] * sdt, ay = alpha[1] * sdt, az = alpha[2] * sdt;
  for (let p = 0; p < x.length; p += 3) {
    const rx = x[p] - c[0], ry = x[p + 1] - c[1], rz = x[p + 2] - c[2];
    v[p] += ay * rz - az * ry; v[p + 1] += az * rx - ax * rz; v[p + 2] += ax * ry - ay * rx;
  }
}

/** Keep a target point inside the environment bounds. */
function clampToBounds(p, b, m) {
  return [clamp(p[0], b.min[0] + m, b.max[0] - m), clamp(p[1], b.min[1] + m * 0.4, b.max[1] - m * 0.6), clamp(p[2], b.min[2] + m, b.max[2] - m)];
}

/**
 * SWIM — travelling-wave undulation along a spine (+x = head in rest space).
 * Thrust is partly emergent (anisotropic drag on the undulating lattice) and
 * partly an explicit propulsive force scaled by beat frequency.
 */
Locomotion.register('swim', (creature, cfg) => {
  const sim = creature.sim, n = sim.n, rest = sim.lat.rest, env = creature.env;
  const L = cfg.length, headX = cfg.headX, NS = 48, ds = L / (NS - 1);
  const sx = new Float64Array(NS), sz = new Float64Array(NS), th = new Float64Array(NS), pCos = new Float64Array(n), pSin = new Float64Array(n);
  const params = { beat: 1 };
  let phase = 0, freq = cfg.baseFreq, bend = 0, headYaw = 0, wrig = 0, wrigPhase = 0, amp = 1, time = 0;
  const ctrl = { alpha: [0, 0, 0], thrust: [0, 0, 0] };
  const k = (2 * Math.PI) / (cfg.waveLength * L);
  return {
    params,
    defaultIntent: () => ({ target: null, speed: 0.6, beat: 1, amplitude: 1, turnBend: 1, headYaw: 0, wriggle: 0 }),
    pose(dt, intent, posed) {
      time += dt;
      const tf = cfg.baseFreq * params.beat * intent.beat;
      freq += (tf - freq) * damp(4, dt);
      phase += 2 * Math.PI * freq * dt;
      amp += (intent.amplitude - amp) * damp(5, dt);
      headYaw += (intent.headYaw - headYaw) * damp(7, dt);
      wrig += (intent.wriggle - wrig) * damp(8, dt);
      wrigPhase += 2 * Math.PI * cfg.wriggleFreq * dt;
      let x = headX, z = 0;
      for (let i = 0; i < NS; i++) {
        const s = i * ds, sig = s / L;
        const A = (cfg.amp[0] + cfg.amp[1] * sig + cfg.amp[2] * sig * sig) * amp;
        let t = A * Math.sin(phase - k * s);
        t -= bend * cfg.turnBend * sig;
        t += headYaw * (1 - smoothstep(0, cfg.neck, s));
        t += wrig * cfg.wriggleAmp * Math.sin(wrigPhase - 9 * sig) * (0.35 + sig);
        th[i] = t;
        if (i > 0) { const tm = 0.5 * (th[i] + th[i - 1]); x -= Math.cos(tm) * ds; z += Math.sin(tm) * ds; }
        sx[i] = x; sz[i] = z;
      }
      for (let p = 0; p < n; p++) {
        const rx = rest[3 * p], ry = rest[3 * p + 1], rz = rest[3 * p + 2];
        const s = headX - rx, sc = clamp(s, 0, L), f = (sc / L) * (NS - 1), i0 = Math.min(Math.floor(f), NS - 2), a = f - i0;
        const X = sx[i0] + (sx[i0 + 1] - sx[i0]) * a, Z = sz[i0] + (sz[i0 + 1] - sz[i0]) * a, T = th[i0] + (th[i0 + 1] - th[i0]) * a;
        const c = Math.cos(T), sn = Math.sin(T), e = s - sc;
        posed[3 * p] = X + rz * sn - e * c;
        posed[3 * p + 1] = ry;
        posed[3 * p + 2] = Z + rz * c + e * sn;
        pCos[p] = c; pSin[p] = sn;
      }
    },
    control(dt, intent) {
      const F = sim.frame, f = F.f, u = F.u, l = F.l, c = F.c, w = F.omega;
      const head = [c[0] + f[0] * cfg.headOffset, c[1] + f[1] * cfg.headOffset, c[2] + f[2] * cfg.headOffset];
      const tgt = clampToBounds(intent.target || v3.add(head, v3.scale(f, 2)), env.bounds, cfg.wallMargin);
      const to = v3.sub(tgt, head), dh = Math.hypot(to[0], to[2]);
      const err = dh > 0.05 ? wrapAngle(Math.atan2(-to[2], to[0]) - Math.atan2(-f[2], f[0])) : 0;
      const ay = clamp(cfg.turnGain * err - cfg.turnDamp * w[1], -cfg.maxTurn, cfg.maxTurn);
      bend += (clamp(err * 1.1, -1, 1) * intent.turnBend - bend) * damp(3, dt);
      const pitch = Math.asin(clamp(f[1], -1, 1));
      const pitchDes = clamp(Math.atan2(to[1], Math.max(dh, 0.5)), -cfg.maxPitch, cfg.maxPitch);
      const ap = cfg.pitchGain * (pitchDes - pitch) - cfg.pitchDamp * v3.dot(w, l);
      const rollAxis = v3.cross(u, [0, 1, 0]);
      const ar = cfg.rollGain * v3.dot(rollAxis, f) - cfg.rollDamp * v3.dot(w, f);
      ctrl.alpha = [l[0] * ap + f[0] * ar, ay + l[1] * ap + f[1] * ar, l[2] * ap + f[2] * ar];
      // Thrust grows with beat frequency and eases off while the target is off-axis, tightening turns.
      const th0 = cfg.thrust * intent.speed * Math.sqrt(freq / cfg.baseFreq) * clamp(0.35 + 0.65 * Math.cos(err), 0.15, 1);
      // Pectoral lift / buoyancy trim: a bounded vertical force toward the target depth.
      const lift = clamp((tgt[1] - c[1]) * cfg.liftGain - F.v[1] * cfg.liftDamp, -cfg.maxLift, cfg.maxLift);
      ctrl.thrust = [f[0] * th0, f[1] * th0 + lift, f[2] * th0];
    },
    applyForces(sdt) {
      const x = sim.x, v = sim.v, F = sim.frame, f = F.f, u = F.u, l = F.l, T = ctrl.thrust, c0 = F.c;
      const wx = ctrl.alpha[0] * sdt, wy = ctrl.alpha[1] * sdt, wz = ctrl.alpha[2] * sdt;
      const ct = 1 - 1 / (1 + cfg.drag[0] * sdt), cn = 1 - 1 / (1 + cfg.drag[1] * sdt), cu = 1 - 1 / (1 + cfg.drag[2] * sdt);
      for (let p = 0; p < n; p++) {
        const i = 3 * p;
        const rx = x[i] - c0[0], ry = x[i + 1] - c0[1], rz = x[i + 2] - c0[2];
        v[i] += T[0] * sdt + wy * rz - wz * ry; v[i + 1] += T[1] * sdt + wz * rx - wx * rz; v[i + 2] += T[2] * sdt + wx * ry - wy * rx;
        const cs = pCos[p], sn = pSin[p];
        const tx = f[0] * cs - l[0] * sn, ty = f[1] * cs - l[1] * sn, tz = f[2] * cs - l[2] * sn;
        const nx = f[0] * sn + l[0] * cs, ny = f[1] * sn + l[1] * cs, nz = f[2] * sn + l[2] * cs;
        const vt = (v[i] * tx + v[i + 1] * ty + v[i + 2] * tz) * ct;
        const vn = (v[i] * nx + v[i + 1] * ny + v[i + 2] * nz) * cn;
        const vu = (v[i] * u[0] + v[i + 1] * u[1] + v[i + 2] * u[2]) * cu;
        v[i] -= tx * vt + nx * vn + u[0] * vu; v[i + 1] -= ty * vt + ny * vn + u[1] * vu; v[i + 2] -= tz * vt + nz * vn + u[2] * vu;
      }
    },
    readouts: { get beatsPerMinute() { return freq * 60; }, get speed() { return sim.frame.speed; } },
    reset() { phase = 0; freq = cfg.baseFreq; bend = 0; headYaw = 0; wrig = 0; amp = 1; },
  };
});

/**
 * HOP — ground locomotion with crouch / launch / flight / landing phases.
 * Squash-and-stretch is posed into the lattice rest shape; the jump itself is
 * a ballistic impulse under gravity, and floppy parts react physically.
 */
Locomotion.register('hop', (creature, cfg) => {
  const sim = creature.sim, n = sim.n, rest = sim.lat.rest, env = creature.env;
  const params = { hopHeight: 1 };
  let phase = 'ground', timer = 0, crouch = 0, air = 0, cooldown = 0.5, hops = 0, sniffPh = 0, twist = 0, impulse = null;
  let alpha = [0, 0, 0], plannedScale = 1;
  const com = sim.restCom;
  return {
    params,
    defaultIntent: () => ({ target: null, hop: false, hopNow: false, hopScale: 1, sniff: 0, twist: 0, crouch: 0, turn: true }),
    pose(dt, intent, posed) {
      const cTarget = phase === 'crouch' ? 1 : intent.crouch;
      crouch += (cTarget - crouch) * damp(phase === 'crouch' ? 18 : 6, dt);
      const stretch = phase === 'air' ? Math.max(0, 1 - air * 2.2) : 0;
      twist += ((phase === 'air' ? intent.twist : 0) - twist) * damp(10, dt);
      sniffPh += dt * 2 * Math.PI * 7;
      const sy = 1 - cfg.squash * crouch + cfg.stretch * stretch, sxz = 1 + cfg.squash * 0.45 * crouch - cfg.stretch * 0.3 * stretch;
      for (let p = 0; p < n; p++) {
        const rx = rest[3 * p], ry = rest[3 * p + 1], rz = rest[3 * p + 2];
        let x = com[0] + (rx - com[0]) * sxz, y = cfg.groundY + (ry - cfg.groundY) * sy, z = com[2] + (rz - com[2]) * sxz;
        const hw = smoothstep(cfg.headX - 0.05, cfg.headX + 0.12, rx);
        y += Math.sin(sniffPh) * 0.012 * intent.sniff * hw;
        const tw = twist * 0.55 * smoothstep(com[0] + 0.05, com[0] - 0.3, rx);
        if (tw !== 0) { const c = Math.cos(tw), s = Math.sin(tw), yy = y - com[1]; y = com[1] + yy * c - z * s; z = yy * s + z * c; }
        posed[3 * p] = x; posed[3 * p + 1] = y; posed[3 * p + 2] = z;
      }
    },
    control(dt, intent) {
      const F = sim.frame, f = F.f, u = F.u, c = F.c, w = F.omega, g = Math.abs(env.cfg.gravity) * (1 - env.cfg.buoyancy);
      const grounded = sim.contacts > 0;
      cooldown -= dt;
      const tgt = intent.target ? clampToBounds(intent.target, env.bounds, cfg.wallMargin) : null;
      const to = tgt ? v3.sub(tgt, c) : [f[0], 0, f[2]], dh = Math.hypot(to[0], to[2]);
      const err = dh > 0.05 ? wrapAngle(Math.atan2(-to[2], to[0]) - Math.atan2(-f[2], f[0])) : 0;
      const ay = grounded && intent.turn ? clamp(cfg.turnGain * err - cfg.turnDamp * w[1], -cfg.maxTurn, cfg.maxTurn) : -w[1] * 0.5;
      const rollAxis = v3.cross(u, [0, 1, 0]);
      alpha = [rollAxis[0] * cfg.uprightGain - (w[0]) * cfg.uprightDamp, ay, rollAxis[2] * cfg.uprightGain - (w[2]) * cfg.uprightDamp];
      if (phase === 'ground') {
        const wants = intent.hopNow || (intent.hop && tgt && dh > 0.3);
        if (grounded && wants && cooldown <= 0 && (intent.hopNow || Math.abs(err) < 0.6)) { phase = 'crouch'; timer = cfg.crouchTime; plannedScale = intent.hopScale; }
      } else if (phase === 'crouch') {
        timer -= dt;
        if (timer <= 0) {
          const h = cfg.hopHeight * params.hopHeight * plannedScale, vy = Math.sqrt(2 * g * h), tAir = (2 * vy) / g;
          const dist = Math.min(tgt ? dh : cfg.hopDistance, cfg.hopDistance * plannedScale), hs = dist / tAir;
          const dir = v3.norm([f[0], 0, f[2]]);
          impulse = [dir[0] * hs, vy, dir[2] * hs]; phase = 'air'; air = 0; hops++;
        }
      } else if (phase === 'air') {
        air += dt;
        if (grounded && air > 0.15) { phase = 'ground'; cooldown = cfg.cooldown; }
        if (air > 3) phase = 'ground';
      }
    },
    applyForces(sdt, s) {
      applySpin(sim, alpha, sdt);
      if (impulse && s === 0) {
        const v = sim.v;
        for (let p = 0; p < v.length; p += 3) { v[p] += impulse[0]; v[p + 1] += impulse[1]; v[p + 2] += impulse[2]; }
        impulse = null;
      }
    },
    readouts: { get hops() { return hops; }, get phase() { return phase; }, get airborne() { return phase === 'air'; } },
    reset() { phase = 'ground'; timer = 0; crouch = 0; cooldown = 0.5; hops = 0; impulse = null; },
  };
});

/* ---------- 5. Behaviour engine ---------------------------------------- */
/**
 * Prioritised finite-state machine. Each state declares its kind
 * (autonomous | user | reactive | environment), priority, whether it can be
 * interrupted, whether it resumes after an interruption, and optional
 * enter / update / exit / trigger / retrigger hooks. States write into the
 * per-tick `ctx.intent`, which locomotion and expression systems consume.
 */
class BehaviorMachine {
  constructor(def, ctx) {
    this.def = def; this.ctx = ctx; this.stack = []; this.history = [];
    this.ids = Object.keys(def.states);
    this.triggered = this.ids.filter((id) => def.states[id].trigger).sort((a, b) => def.states[b].priority - def.states[a].priority);
    this.current = null;
    this.enter(def.initial, null, 'start');
  }
  get state() { return this.current; }
  label(id = this.current.id) { return this.def.states[id].label || id; }
  canPreempt(def) {
    const cur = this.current.def;
    if (cur.interruptible === false && def.priority <= (cur.lockPriority ?? 99)) return false;
    return def.priority > cur.priority;
  }
  enter(id, payload, reason) {
    const def = this.def.states[id];
    if (!def) throw new Error(`Unknown behaviour "${id}"`);
    const prev = this.current;
    if (prev) {
      if (prev.def.resumable && reason !== 'next' && reason !== 'done') {
        this.stack = this.stack.filter((s) => s.id !== prev.id);
        this.stack.push(prev);
        if (this.stack.length > 4) { const old = this.stack.shift(); old.def.exit?.(this.ctx, old.data); }
      } else prev.def.exit?.(this.ctx, prev.data);
    }
    this.stack = this.stack.filter((s) => s.id !== id);
    this.current = { id, def, data: {}, time: 0 };
    this.history.unshift({ id, reason, t: this.ctx.time });
    if (this.history.length > 8) this.history.pop();
    def.enter?.(this.ctx, this.current.data, payload);
    return true;
  }
  /** User-triggered request (buttons). */
  request(id, payload) {
    if (this.current.id === id) { this.current.def.retrigger?.(this.ctx, this.current.data, payload); return true; }
    if (!this.canPreempt(this.def.states[id])) return false;
    return this.enter(id, payload, 'user');
  }
  /** Reactive event (touch, grab, comb). Re-triggering an active reaction extends it. */
  react(id, payload) {
    if (this.current.id === id) { this.current.def.retrigger?.(this.ctx, this.current.data, payload); return true; }
    if (!this.canPreempt(this.def.states[id])) return false;
    return this.enter(id, payload, 'reaction');
  }
  cancel(id) {
    this.stack = this.stack.filter((s) => s.id !== id);
    if (this.current.id === id) this.finish(null);
  }
  finish(next) {
    const cur = this.current;
    cur.def.exit?.(this.ctx, cur.data);
    this.current = null;
    if (next) { this.enterFresh(next.id, next.payload, 'next'); return; }
    if (this.stack.length) {
      this.current = this.stack.pop();
      this.history.unshift({ id: this.current.id, reason: 'resume', t: this.ctx.time });
      this.current.def.resume?.(this.ctx, this.current.data);
    } else this.enterFresh(this.def.initial, null, 'done');
  }
  enterFresh(id, payload, reason) { this.current = null; this.enter(id, payload, reason); }
  tick(dt) {
    const ctx = this.ctx;
    for (const id of this.triggered) {
      const d = this.def.states[id];
      if (id === this.current.id || this.stack.some((s) => s.id === id)) continue;
      if (this.canPreempt(d) && d.trigger(ctx)) { this.enter(id, null, d.kind); break; }
    }
    const cur = this.current;
    cur.time += dt;
    let r = cur.def.update ? cur.def.update(ctx, cur.data, dt, cur.time) : undefined;
    if (r === undefined && cur.def.duration && cur.time >= cur.def.duration) r = 'done';
    if (r === 'done') this.finish(null);
    else if (r && r.next) this.finish({ id: r.next, payload: r.payload });
  }
  reset() { this.stack = []; this.history = []; this.current = null; this.enter(this.def.initial, null, 'reset'); }
}

/* ---------- 6. Environment & props ------------------------------------- */
/** Environment presets are engine-level; species choose one and may override fields. */
const ENVIRONMENTS = {
  tank: { label: 'Studio tank', medium: 'water', gravity: -28, buoyancy: 1, friction: 0.15, floorPattern: 'caustics',
    bounds: { min: [-3.4, 0, -2.5], max: [3.4, 2.6, 2.5] }, background: '#d8d2c8', floor: '#cdc3b1', fadeRadius: 11 },
  studio: { label: 'Felt studio', medium: 'air', gravity: -28, buoyancy: 0, friction: 0.55, floorPattern: 'felt',
    bounds: { min: [-3, 0, -2.2], max: [3, 3.4, 2.2] }, background: '#d9d3ca', floor: '#c9bfae', fadeRadius: 10 },
};

class Environment {
  constructor(cfg, itemDefs, rng) {
    this.cfg = cfg; this.itemDefs = itemDefs || {}; this.rng = rng;
    this.bounds = cfg.bounds; this.friction = cfg.friction; this.items = []; this.nextId = 1; this.time = 0;
    const self = this;
    this.api = {
      get bounds() { return self.bounds; },
      get floorY() { return self.bounds.min[1]; },
      get medium() { return self.cfg.medium; },
      items: (type) => self.items.filter((it) => it.state === 'active' && (!type || it.type === type)),
      count: (type) => self.items.filter((it) => !type || it.type === type).length,
      spawn: (type, opts) => self.spawn(type, opts),
      nearest: (type, p) => {
        let best = null, bd = Infinity;
        for (const it of self.items) if (it.type === type && it.state === 'active') {
          const d = v3.dist(it.pos, p); if (d < bd) { bd = d; best = it; }
        }
        return best;
      },
      attract: (it, p, rate, dt) => { const k = damp(rate, dt); for (let a = 0; a < 3; a++) it.pos[a] += (p[a] - it.pos[a]) * k; it.vel = [0, 0, 0]; },
      consume: (it, p) => { it.state = 'consumed'; it.sink = p.slice(); },
      randomPoint: (margin = 0.8, yRange = [0.4, 0.8]) => {
        const b = self.bounds, r = self.rng;
        return [lerp(b.min[0] + margin, b.max[0] - margin, r()), lerp(b.min[1], b.max[1], lerp(yRange[0], yRange[1], r())),
          lerp(b.min[2] + margin, b.max[2] - margin, r())];
      },
      get center() { const b = self.bounds; return [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2]; },
    };
  }
  spawn(type, opts = {}) {
    const def = this.itemDefs[type];
    if (!def) throw new Error(`Unknown item type "${type}"`);
    const it = { id: this.nextId++, type, def, pos: (opts.pos || this.api.randomPoint(1.0, [0.85, 0.9])).slice(), vel: [0, 0, 0],
      yaw: this.rng() * Math.PI * 2, spin: (this.rng() - 0.5) * 2, t: 0, scale: 1, state: 'active' };
    this.items.push(it);
    return it;
  }
  step(dt) {
    this.time += dt;
    const g = this.cfg.gravity, floor = this.bounds.min[1];
    for (const it of this.items) {
      it.t += dt;
      if (it.state === 'consumed') {
        for (let a = 0; a < 3; a++) it.pos[a] += (it.sink[a] - it.pos[a]) * damp(14, dt);
        it.scale -= dt * 5;
        continue;
      }
      const b = this.cfg.medium === 'water' ? it.def.buoyancy : 0;
      it.vel[1] += g * (1 - b) * dt;
      const drag = this.cfg.medium === 'water' ? it.def.drag : 0.1;
      for (let a = 0; a < 3; a++) it.vel[a] *= Math.exp(-drag * dt);
      it.vel[0] += Math.sin(it.t * 3.1 + it.id) * dt * 0.4 * (it.pos[1] > floor + it.def.radius + 0.05 ? 1 : 0);
      for (let a = 0; a < 3; a++) it.pos[a] += it.vel[a] * dt;
      if (it.pos[1] < floor + it.def.radius) { it.pos[1] = floor + it.def.radius; it.vel[1] = 0; it.vel[0] *= 0.8; it.vel[2] *= 0.8; it.landed = true; }
      if (!it.landed) it.yaw += it.spin * dt;
    }
    this.items = this.items.filter((it) => it.scale > 0);
  }
  clear() { this.items = []; }
  /** Model matrix for an item (yaw, flutter tilt, uniform scale). */
  itemMatrix(it) {
    const s = Math.max(it.scale, 0) * (it.def.scale || 1), cy = Math.cos(it.yaw), sy = Math.sin(it.yaw);
    const tilt = it.landed ? 0.0 : Math.sin(it.t * 4 + it.id) * 0.35, ct = Math.cos(tilt), st = Math.sin(tilt);
    // R = Ry(yaw) * Rx(tilt)
    return new Float32Array([cy * s, 0, -sy * s, 0, sy * st * s, ct * s, cy * st * s, 0, sy * ct * s, -st * s, cy * ct * s, 0,
      it.pos[0], it.pos[1], it.pos[2], 1]);
  }
}
