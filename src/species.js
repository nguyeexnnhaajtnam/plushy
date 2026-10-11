'use strict';
/* =========================================================================
   SPECIES DEFINITIONS
   Everything animal-specific lives here: anatomy (SDFs, regions, joints,
   decals), physics tuning, locomotion choice, behaviours, interaction rules,
   materials, controls and read-outs. The engine above never branches on a
   species id.
   ========================================================================= */

/* ---------------------------------------------------------------------- */
/* 1. PLUSH HAMMERHEAD — complete species                                  */
/* ---------------------------------------------------------------------- */
const PlushHammerhead = (() => {
  const { SDF } = Engine, { smoothstep, clamp, lerp, v3 } = Engine.math;
  // Rest space: +x toward the head, +y up, z lateral. 1 unit ≈ 35 cm.
  const W = 0.5;
  const hammerMid = (z) => 0.03 - 0.03 * Math.min((z / W) ** 2, 1.4);
  const DORSAL = [0.17, 0.12, 0.1, 0.32, 0.0, 0.47, -0.12, 0.57, -0.27, 0.64, -0.21, 0.52, -0.16, 0.38, -0.15, 0.24, -0.21, 0.12];
  const CAUDAL = [-0.84, 0.1, -1.02, 0.22, -1.2, 0.38, -1.37, 0.53, -1.3, 0.4, -1.21, 0.22, -1.14, 0.08, -1.19, -0.07, -1.18, -0.2,
    -1.06, -0.12, -0.93, -0.03, -0.84, 0.0];
  const PECT = [0.12, -0.04, 0.08, 0.15, -0.02, 0.31, -0.13, 0.42, -0.18, 0.38, -0.15, 0.2, -0.12, -0.04];
  const fDorsal = SDF.frame([0, 0, 0], [1, 0, 0], [0, 1, 0]);
  const fPectR = SDF.frame([0.1, -0.12, 0.12], [1, 0, 0], [-0.35, -0.42, 1]);
  const fPectL = SDF.frame([0.1, -0.12, -0.12], [1, 0, 0], [-0.35, -0.42, -1]);
  const FISH_TAIL = [-0.06, 0, -0.118, 0.048, -0.102, 0, -0.118, -0.048];
  const fFish = SDF.frame([0, 0, 0], [1, 0, 0], [0, 1, 0]);

  const torso = (x, y, z) => {
    let d = SDF.ellipsoid(x, y, z, 0.4, 0.0, 0, 0.28, 0.14, 0.17);
    d = SDF.smin(d, SDF.ellipsoid(x, y, z, 0.06, 0.0, 0, 0.42, 0.21, 0.21), 0.14);
    d = SDF.smin(d, SDF.ellipsoid(x, y, z, -0.38, 0.01, 0, 0.4, 0.16, 0.15), 0.14);
    return SDF.smin(d, SDF.ellipsoid(x, y, z, -0.8, 0.04, 0, 0.3, 0.075, 0.07), 0.12);
  };
  const hammer = (x, y, z) => {
    const az = Math.abs(z), zn = z / 0.55;
    const xf = 0.8 - 0.16 * zn * zn - 0.035 * Math.exp(-((z / 0.07) ** 2));
    const xb = 0.53 - 0.13 * zn * zn;
    let d2 = SDF.smax(x - xf, xb - x, 0.06);
    d2 = SDF.smax(d2, az - W, 0.1);
    const h = 0.062 - 0.018 * Math.min(az / W, 1);
    const d = SDF.extrude(d2, y - hammerMid(z), h, h * 0.9);
    return SDF.smin(d, SDF.ellipsoid(x, y, az, 0.585, 0.02, 0.5, 0.11, 0.07, 0.085), 0.06);
  };
  const eye = (x, y, z) => SDF.sphere(x, y, Math.abs(z), 0.588, 0.034, 0.6, 0.05);
  const dorsal = (x, y, z) => SDF.planar(x, y, z, fDorsal, DORSAL, 0.045, 0.04, 0.09);
  const caudal = (x, y, z) => SDF.planar(x, y, z, fDorsal, CAUDAL, 0.04, 0.035, 0.08);
  const pect = (x, y, z) => SDF.planar(x, y, z, z > 0 ? fPectR : fPectL, PECT, 0.032, 0.028, 0.08);
  const cavity = (x, y, z) => SDF.ellipsoid(x, y, z, 0.54, -0.085, 0, 0.065, 0.045, 0.14);
  const body = (x, y, z) => {
    let d = SDF.smin(torso(x, y, z), hammer(x, y, z), 0.12);
    d = SDF.smin(d, dorsal(x, y, z), 0.07);
    d = SDF.smin(d, caudal(x, y, z), 0.06);
    d = SDF.smin(d, pect(x, y, z), 0.06);
    d = SDF.smax(d, -cavity(x, y, z), 0.025);
    return Math.min(d, eye(x, y, z));
  };
  const finField = (x, y, z) => Math.min(torso(x, y, z), hammer(x, y, z)) - 0.004 - Math.min(dorsal(x, y, z), caudal(x, y, z), pect(x, y, z));
  const isFin = (x, y, z) => finField(x, y, z) > 0;
  const seamY = (x, z) => lerp(lerp(-0.02, -0.05, smoothstep(-0.9, 0.0, x)), hammerMid(z), smoothstep(0.42, 0.56, x));
  // Lower jaw: a smiling lip arc with felt teeth; upper teeth hang from the palate.
  const arc = (t) => [0.565 - 0.05 * t * t, -0.112 + 0.03 * t * t, 0.12 * t];
  const arcU = (t) => [0.57 - 0.05 * t * t, -0.058 + 0.025 * t * t, 0.115 * t];
  const LOWER_T = [-0.8, -0.53, -0.27, 0, 0.27, 0.53, 0.8], UPPER_T = [-0.67, -0.4, -0.13, 0.13, 0.4, 0.67];
  const lip = (x, y, z) => {
    let d = Infinity;
    for (let i = 0; i < 6; i++) d = Math.min(d, SDF.cone(x, y, z, arc(-1 + i / 3), arc(-1 + (i + 1) / 3), 0.026, 0.026));
    return SDF.smin(d, SDF.ellipsoid(x, y, z, 0.5, -0.128, 0, 0.075, 0.03, 0.11), 0.03);
  };
  const lowerTeeth = (x, y, z) => {
    let d = Infinity;
    for (const t of LOWER_T) { const b = v3.add(arc(t), [0, 0.016, 0]); d = Math.min(d, SDF.cone(x, y, z, b, v3.add(b, [0.004, 0.03, 0]), 0.012, 0.002)); }
    return d;
  };
  const upperTeeth = (x, y, z) => {
    let d = Infinity;
    for (const t of UPPER_T) { const b = arcU(t); d = Math.min(d, SDF.cone(x, y, z, b, v3.add(b, [0.004, -0.028, 0]), 0.011, 0.002)); }
    return d;
  };
  const jaw = (x, y, z) => Math.min(lip(x, y, z), lowerTeeth(x, y, z));

  const finFlow = (x, y, z) => {
    const dT = Math.min(torso(x, y, z), hammer(x, y, z)), dD = dorsal(x, y, z), dC = caudal(x, y, z), dP = pect(x, y, z);
    const m = Math.min(dD, dC, dP);
    if (m > dT) return [-1, 0, 0];
    if (m === dD) return [-0.55, 1, 0];
    if (m === dC) return [-1, 0.35 * Math.sign(y - 0.05), 0];
    return [-0.4, -0.35, Math.sign(z)];
  };

  return {
    id: 'hammerhead', name: 'Plush Hammerhead', category: 'Cartilaginous fish', status: 'complete',
    description: 'Minky hammerhead with a felt-lined smile, articulated jaw and glass eyes on its cephalofoil.',
    display: {
      kicker: 'No. 01 — Sphyrnidae', title: ['Plush', 'Hammerhead'], theme: { accent: '#8a5a33' },
      description: 'A minky hammerhead with a sewn cream belly, a felt-lined smile and glass eyes on the tips of its hammer. ' +
        'It swims on a travelling S-wave, sweeps the floor for felt fish, and wriggles and blushes when you boop the middle of its head.',
    },
    environment: { preset: 'tank', spawn: { pos: [0.2, 1.2, 0.3], yaw: 0.4 }, camera: { target: [0, 0.85, 0], distance: 6.2, yaw: 0.62, pitch: 0.32, follow: 0.55 } },
    createState: () => ({ fishEaten: 0 }),

    anatomy: {
      physicsSdf: body,
      parts: [
        { id: 'body', sdf: body, resolution: 0.0165, bounds: { min: [-1.44, -0.29, -0.7], max: [0.86, 0.7, 0.7] },
          region(x, y, z) {
            if (eye(x, y, z) < 0.006) return 'eye';
            if (cavity(x, y, z) < 0.006) return 'mouth';
            if (isFin(x, y, z)) return 'fin';
            return y < seamY(x, z) ? 'belly' : 'body';
          },
          // Signed fields resolved per fragment so the belly seam and the lip line stay crisp.
          boundaries: [
            { field: (x, y, z) => finField(x, y, z), positive: 'fin', negative: (x, y, z) => (y < seamY(x, z) ? 'belly' : 'body'), band: 0.04 },
            { field: (x, y, z) => cavity(x, y, z) - 0.006, negative: 'mouth', positive: (x, y, z) => (y < seamY(x, z) ? 'belly' : 'body'), band: 0.05 },
            { field: (x, y, z) => y - seamY(x, z), positive: 'body', negative: 'belly', band: 0.05, where: (x, y, z) => !isFin(x, y, z) && eye(x, y, z) > 0.006 },
          ] },
        { id: 'jaw', sdf: jaw, resolution: 0.0055, bounds: { min: [0.38, -0.18, -0.17], max: [0.64, -0.03, 0.17] },
          attach: { anchor: 'hinge', joint: 'jaw' }, fur: false,
          region(x, y, z) {
            if (lowerTeeth(x, y, z) < 0.003) return 'tooth';
            const a = arc(clamp(z / 0.12, -1, 1));
            return y > a[1] + 0.006 && x > 0.47 ? 'mouth' : 'belly';
          } },
        { id: 'upperTeeth', sdf: upperTeeth, resolution: 0.005, bounds: { min: [0.49, -0.1, -0.11], max: [0.6, -0.02, 0.11] },
          attach: { anchor: 'hinge' }, fur: false, region: () => 'tooth' },
      ],
      anchors: { hinge: [0.44, -0.095, 0] },
      joints: { jaw: { axis: [0, 0, 1], min: 0, max: -0.52, stiffness: 320, damping: 22 } },
      landmarks: { mouth: [0.56, -0.09, 0], head: [0.64, 0.03, 0], tail: [-1.0, 0.05, 0] },
      expressions: { channels: ['blush'], spots: [
        { center: [0.69, 0.09, 0.17], radius: 0.085, color: '#f07c93', channel: 'blush' },
        { center: [0.69, 0.09, -0.17], radius: 0.085, color: '#f07c93', channel: 'blush' }] },
      decals({ surfacePoint, sdf }) {
        const out = [];
        for (const side of [1, -1]) {
          const S = side > 0 ? 'R' : 'L';
          for (let i = 0; i < 5; i++) {
            const xg = 0.31 - i * 0.045, pts = [];
            for (const yy of [0.085, 0.03, -0.025, -0.07]) { const p = surfacePoint(sdf, [xg + 0.02 * (yy / 0.07) - 0.006 * (yy / 0.07) ** 2, yy, 0], [0, 0, side], 0.6, 0.004); if (p) pts.push(p); }
            if (pts.length > 1) out.push({ group: 'gills' + S, region: 'gill', width: 0.0095, points: pts });
          }
          for (const [g, a, b] of [['seamF' + S, -0.2, 0.42], ['seamB' + S, -0.88, -0.2]]) {
            const pts = [];
            for (let k = 0; k <= 9; k++) {
              const x = lerp(a, b, k / 9), p = surfacePoint(sdf, [x, seamY(x, 0), 0], [0, 0, side], 0.6, 0.004);
              if (p) pts.push(p);
            }
            out.push({ group: g, region: 'seam', width: 0.0055, points: pts, dash: { period: 0.024, duty: 0.58 } });
          }
        }
        for (const [g, a, b] of [['rimL', -0.5, 0], ['rimR', 0, 0.5]]) {
          const pts = [];
          for (let k = 0; k <= 11; k++) {
            const z = lerp(a, b, k / 11), p = surfacePoint(sdf, [0.6, hammerMid(z), z], [1, 0, 0], 0.4, 0.003);
            if (p) pts.push(p);
          }
          out.push({ group: g, region: 'seam', width: 0.0048, points: pts, dash: { period: 0.024, duty: 0.58 } });
        }
        return out;
      },
    },

    physics: {
      latticeSpacing: 0.08, substeps: 8,
      shapeStiffness: [0.05, 0.42], volumeCompliance: [6e-8, 6e-10], inflate: [0.93, 1.07], grabCompliance: 0.006,
      unitMeters: 0.35, fillGramsPerLitre: 28, shellGrams: 140,
      stuffingAt(x, y, z) {
        const dB = Math.min(torso(x, y, z), hammer(x, y, z)), dF = Math.min(dorsal(x, y, z), caudal(x, y, z), pect(x, y, z));
        if (dF < dB) return { density: 0.45, stiffness: 0.5 };      // fins: lightly stuffed, floppier
        if (x > 0.45) return { density: 1.25, stiffness: 1.5 };     // packed head keeps the hammer flat
        if (x < -0.75) return { density: 0.7, stiffness: 0.75 };
        return { density: 1, stiffness: 1 };
      },
    },

    locomotion: {
      type: 'swim', headX: 0.52, length: 1.75, waveLength: 0.95, amp: [0.03, 0.06, 0.32], baseFreq: 1.1, neck: 0.32,
      turnBend: 0.5, wriggleAmp: 0.22, wriggleFreq: 4.5, headOffset: 0.55, thrust: 2.0, drag: [1.3, 9, 6],
      turnGain: 11, turnDamp: 2, maxTurn: 12, pitchGain: 7, pitchDamp: 3, maxPitch: 0.45, rollGain: 16, rollDamp: 4, wallMargin: 0.95, liftGain: 2.2, liftDamp: 1.2, maxLift: 1.6,
    },

    fur: {
      shells: 10, length: 0.016, density: 320, droop: 0.02, combRadius: 0.09,
      lagStiffness: 140, lagDamping: 0.35, inertia: 0.012, flow: 0.22,
      direction: (x, y, z) => finFlow(x, y, z),
    },

    materials: {
      regions: {
        body: { type: 'plush' }, belly: { type: 'plush', fur: 0.75 }, fin: { type: 'plush', fur: 0.7 },
        eye: { type: 'glass' }, mouth: { type: 'felt' }, tooth: { type: 'felt', sheen: 0.45 },
        gill: { type: 'embroidery' }, seam: { type: 'stitch' },
      },
      defaultPreset: 'bronze',
      presets: [
        { id: 'bronze', label: 'Bronze', colors: { body: '#8d6239', belly: '#efe2c7', fin: '#7c5331', eye: '#1b120e', mouth: '#a8434f', tooth: '#fbf6ea', gill: '#4a2f1d', seam: '#5e4128' } },
        { id: 'lagoon', label: 'Lagoon', colors: { body: '#3f7c86', belly: '#eee6d1', fin: '#336771', eye: '#101618', mouth: '#ad4858', tooth: '#fbf6ea', gill: '#1f4249', seam: '#2b545b' } },
        { id: 'sunset', label: 'Sunset', colors: { body: '#df7a53', belly: '#f7e5c9', fin: '#c95f3e', eye: '#1c110d', mouth: '#9b3646', tooth: '#fbf6ea', gill: '#7a321f', seam: '#8d4128' } },
      ],
    },

    behaviors: {
      initial: 'cruise',
      states: {
        cruise: {
          kind: 'autonomous', priority: 0, resumable: true, label: 'Cruising',
          enter(ctx, d) { d.scanAt = ctx.time + 10 + ctx.rand() * 8; d.wp = null; },
          resume(ctx, d) { d.scanAt = Math.max(d.scanAt, ctx.time + 6); d.wp = null; },
          update(ctx, d) {
            const I = ctx.intent, b = ctx.body, E = ctx.env;
            if (ctx.flags.circle) {
              const c = E.center, a = Math.atan2(b.c[2] - c[2], b.c[0] - c[0]) - 0.75;
              I.target = [c[0] + 1.75 * Math.cos(a), 1.15 + 0.3 * Math.sin(ctx.time * 0.27), c[2] + 1.45 * Math.sin(a)];
            } else {
              if (!d.wp || v3.dist(d.wp, b.c) < 0.8) d.wp = E.randomPoint(1.1, [0.25, 0.7]);
              I.target = d.wp;
            }
            I.speed = 0.55;
            if (ctx.time > d.scanAt) return { next: 'scan' };
          },
        },
        scan: {
          kind: 'autonomous', priority: 2, duration: 6, label: 'Scanning the floor',
          update(ctx, d, dt, t) {
            const I = ctx.intent, b = ctx.body, f = v3.norm([b.f[0], 0, b.f[2]]);
            I.target = [b.c[0] + f[0] * 1.6 - f[2] * 0.4 * Math.sin(t * 0.5), ctx.env.floorY + 0.3, b.c[2] + f[2] * 1.6 + f[0] * 0.4 * Math.sin(t * 0.5)];
            I.headYaw = 0.5 * Math.sin(t * 2.6) * smoothstep(0, 0.8, t);
            I.speed = 0.35; I.beat = 0.85;
          },
        },
        hunt: {
          kind: 'environment', priority: 5, label: 'Hunting a felt fish',
          trigger: (ctx) => !!ctx.env.nearest('feltFish', ctx.body.c),
          update(ctx, d, dt, t) {
            const I = ctx.intent, fish = ctx.env.nearest('feltFish', ctx.body.c);
            if (!fish) return 'done';
            const mouth = ctx.landmark('mouth'), dist = v3.dist(mouth, fish.pos);
            I.target = [fish.pos[0], fish.pos[1] + 0.13, fish.pos[2]];
            if (dist > 1.3) { I.speed = 0.75; I.beat = 1.2; I.headYaw = 0.35 * Math.sin(t * 3.2); }
            else { I.speed = 1.05; I.beat = 1.9; I.joints.jaw = clamp((1.1 - dist) / 0.55, 0, 1); }
            if (dist < 0.34) ctx.env.attract(fish, mouth, 5, dt);
            if (dist < 0.12) { ctx.env.consume(fish, mouth); ctx.state.fishEaten++; return { next: 'chomp' }; }
            if (t > 30) return 'done';
          },
        },
        dash: {
          kind: 'user', priority: 6, duration: 1.6, label: 'Dashing',
          enter(ctx, d) { const b = ctx.body, f = v3.norm([b.f[0], 0, b.f[2]]); d.target = [b.c[0] + f[0] * 4, b.c[1], b.c[2] + f[2] * 4]; },
          update(ctx, d) { const I = ctx.intent; I.target = d.target; I.speed = 2.6; I.beat = 2.6; I.amplitude = 1.25; },
        },
        chomp: {
          kind: 'user', priority: 6, duration: 0.8, label: 'Chomp!',
          update(ctx, d, dt, t) { const I = ctx.intent; I.joints.jaw = t < 0.3 ? 1 : 0; I.speed = 0.3; },
        },
        thrash: {
          kind: 'reactive', priority: 9, label: 'Thrashing',
          update(ctx, d, dt, t) {
            const I = ctx.intent;
            I.speed = 0; I.beat = 3.2; I.amplitude = 2.1; I.wriggle = 0.5; I.joints.jaw = 0.35 + 0.35 * Math.sin(t * 9);
            if (!ctx.interaction.grabbed) return { next: 'flee' };
          },
        },
        flee: {
          kind: 'reactive', priority: 8, duration: 1.8, label: 'Bolting away',
          enter(ctx, d) { const b = ctx.body, f = v3.norm([b.f[0], 0, b.f[2]]); d.target = [b.c[0] + f[0] * 3.5, 1.2, b.c[2] + f[2] * 3.5]; },
          update(ctx, d) { const I = ctx.intent; I.target = d.target; I.speed = 2.4; I.beat = 2.7; I.amplitude = 1.2; },
        },
        held: {
          kind: 'reactive', priority: 7, label: 'Being held',
          update(ctx) { const I = ctx.intent; I.speed = 0; I.beat = 0.6; I.amplitude = 0.6; if (!ctx.interaction.grabbed) return 'done'; },
        },
        boop: {
          kind: 'reactive', priority: 7, label: 'Booped!',
          enter(ctx, d) { d.until = ctx.time + 2.0; },
          retrigger(ctx, d) { d.until = ctx.time + 2.0; },
          update(ctx, d) {
            const I = ctx.intent;
            I.wriggle = 1; I.expr.blush = 1; I.speed = 0.15; I.beat = 1.4; I.amplitude = 0.55; I.joints.jaw = 0.22;
            if (ctx.time > d.until) return 'done';
          },
        },
        flinch: {
          kind: 'reactive', priority: 4, duration: 0.8, label: 'Startled',
          update(ctx) { const I = ctx.intent; I.speed = 1.4; I.beat = 2; },
        },
        soothed: {
          kind: 'reactive', priority: 3, label: 'Enjoying a brush',
          enter(ctx, d) { d.until = ctx.time + 2; },
          retrigger(ctx, d) { d.until = ctx.time + 2; },
          update(ctx, d) { const I = ctx.intent; I.speed = 0.12; I.beat = 0.5; I.amplitude = 0.6; I.expr.blush = 0.35; if (ctx.time > d.until) return 'done'; },
        },
      },
    },

    interactions: {
      tools: ['hand', 'finger', 'comb'], pickRadius: 0.032,
      regions: [
        { id: 'hammerCenter', label: 'Hammer centre — boop zone', sphere: [0.66, 0.03, 0, 0.17] },
        { id: 'hammer', label: 'Cephalofoil', box: { min: [0.45, -0.2, -0.7], max: [0.95, 0.2, 0.7] } },
        { id: 'tail', label: 'Tail', box: { min: [-1.6, -0.5, -0.4], max: [-0.55, 0.8, 0.4] } },
        { id: 'dorsal', label: 'Dorsal fin', box: { min: [-0.35, 0.18, -0.1], max: [0.2, 0.75, 0.1] } },
        { id: 'belly', label: 'Belly', test: (x, y, z) => y < seamY(x, z) },
      ],
      reactions: [
        { on: 'grab', region: 'tail', behavior: 'thrash' },
        { on: 'grab', behavior: 'held' },
        { on: 'poke', region: 'hammerCenter', behavior: 'boop' },
        { on: 'stroke', region: 'hammerCenter', behavior: 'boop' },
        { on: 'poke', behavior: 'flinch' },
        { on: 'comb', behavior: 'soothed' },
      ],
    },

    items: {
      feltFish: {
        label: 'Felt fish', radius: 0.045, buoyancy: 0.8, drag: 2.4, resolution: 0.0075,
        bounds: { min: [-0.135, -0.07, -0.05], max: [0.105, 0.07, 0.05] },
        sdf(x, y, z) {
          let d = SDF.ellipsoid(x, y, z, 0.0, 0, 0, 0.08, 0.045, 0.022);
          d = SDF.smin(d, SDF.planar(x, y, z, fFish, FISH_TAIL, 0.012, 0.01, 0), 0.02);
          return Math.min(d, SDF.sphere(x, y, Math.abs(z), 0.045, 0.012, 0.018, 0.009));
        },
        color(x, y, z) {
          if (SDF.sphere(x, y, Math.abs(z), 0.045, 0.012, 0.018, 0.009) < 0.002) return '#111111';
          return Math.abs(x + 0.01) < 0.012 ? '#f6e6c4' : '#f0a03c';
        },
      },
    },

    actions: {
      feed: { run: (ctx) => ctx.env.spawn('feltFish'), enabled: (ctx) => ctx.env.count('feltFish') < 3 },
      dash: { run: (ctx) => ctx.behaviors.request('dash') },
      scan: { run: (ctx) => ctx.behaviors.request('scan') },
      chomp: { run: (ctx) => ctx.behaviors.request('chomp') },
    },

    controls: [
      { id: 'feed', type: 'button', label: 'Feed', action: 'feed', group: 'Actions', primary: true, hint: 'Drop a felt fish into the tank' },
      { id: 'dash', type: 'button', label: 'Dash', action: 'dash', group: 'Actions', hint: 'Burst of rapid tail beats' },
      { id: 'scan', type: 'button', label: 'Scan', action: 'scan', group: 'Actions', hint: 'Sweep the hammer over the floor' },
      { id: 'chomp', type: 'button', label: 'Chomp', action: 'chomp', group: 'Actions', hint: 'Open and snap the jaw' },
      { id: 'smooth', type: 'button', label: 'Smooth fur', action: 'engine.smoothFur', group: 'Actions', hint: 'Reset the combed nap' },
      { id: 'circle', type: 'toggle', label: 'Circle', bind: 'flags.circle', default: true, group: 'Behaviour', hint: 'Swim laps around the tank instead of wandering' },
      { id: 'stuffing', type: 'slider', label: 'Stuffing', bind: 'physics.stuffing', min: 0.3, max: 1.5, step: 0.01, default: 1, group: 'Body', format: (v) => `${Math.round(v * 100)}%` },
      { id: 'pile', type: 'slider', label: 'Pile', bind: 'fur.pile', min: 0, max: 2, step: 0.01, default: 1, group: 'Body', format: (v) => `${(v * 0.016 * 350).toFixed(1)} mm` },
      { id: 'tailBeat', type: 'slider', label: 'Tail beat', bind: 'locomotion.beat', min: 0.4, max: 2, step: 0.01, default: 1, group: 'Body', format: (v) => `${v.toFixed(2)}×` },
      { id: 'fabric', type: 'preset', label: 'Fabric', bind: 'material.preset', default: 'bronze', group: 'Fabric' },
    ],

    readouts: [
      { id: 'mass', label: 'Mass', unit: 'g', get: (r) => r.metrics.mass, format: (v) => v.toFixed(0) },
      { id: 'volume', label: 'Volume', unit: 'L', get: (r) => r.metrics.volume, format: (v) => v.toFixed(2) },
      { id: 'bpm', label: 'Tail beat', unit: 'bpm', get: (r) => r.locomotion.beatsPerMinute, format: (v) => v.toFixed(0) },
      { id: 'fish', label: 'Fish eaten', unit: '', get: (r) => r.state.fishEaten, format: (v) => String(v) },
    ],

    notes: {
      anatomy: 'Body, hammer, fins and mouth cavity are one blended signed-distance field meshed with surface nets (~1.6 cm grid). ' +
        'The lower jaw and the upper teeth are separate finer meshes rigidly attached at a hinge.',
      implemented: [
        'Swimming is driven by a travelling S-wave posed into the lattice rest shape; the body follows it through shape matching, so the head yaws and the fins lag on their own.',
        'Steering bends the body into the turn and turns the head; scanning sweeps only the head segment.',
        'Hunting is environment-triggered by a felt fish in the tank: the shark approaches while scanning, speeds up, opens its jaw, then pulls the fish in by suction and eats it.',
        'When you grab it by the tail it thrashes (beat ×3.2), then bolts away once you let go. A boop on the centre of the hammer makes it wriggle and blush.',
        'Gills, the belly seam and the hammer rim are drawn as embroidery and stitch decals that flatten the pile under them.',
      ],
      approximations: [
        'There is no fluid simulation. Thrust comes partly from anisotropic drag on the undulating lattice and partly from a propulsive force along the body axis, scaled by beat frequency.',
        'The jaw is a 1-DoF damped joint, rigidly skinned to the lattice frame at its hinge. It is not part of the soft lattice.',
        'Suction feeding pulls the fish toward the mouth kinematically once it is within 0.34 units.',
      ],
    },
  };
})();

/* ---------------------------------------------------------------------- */
/* 2. DUST BUNNY — minimal second species (validation fixture)            */
/*    Different anatomy (ears, no fins, no jaw), a different locomotion    */
/*    capability (hop under gravity), different environment, behaviours,  */
/*    reactions and controls — with zero engine changes.                  */
/* ---------------------------------------------------------------------- */
const DustBunny = (() => {
  const { SDF } = Engine, { smoothstep, v3 } = Engine.math;
  const EAR = [0.06, 0, 0.068, 0.17, 0.045, 0.33, 0, 0.38, -0.045, 0.33, -0.068, 0.17, -0.06, 0];
  const fEarR = SDF.frame([0.2, 0.56, 0.075], [0, 0, 1], [-0.3, 1, 0.22]);
  const fEarL = SDF.frame([0.2, 0.56, -0.075], [0, 0, -1], [-0.3, 1, -0.22]);
  const core = (x, y, z) => {
    const az = Math.abs(z);
    let d = SDF.ellipsoid(x, y, z, -0.04, 0.27, 0, 0.34, 0.26, 0.27);
    d = SDF.smin(d, SDF.ellipsoid(x, y, az, -0.14, 0.2, 0.13, 0.2, 0.17, 0.12), 0.08);
    d = SDF.smin(d, SDF.ellipsoid(x, y, z, 0.27, 0.47, 0, 0.19, 0.17, 0.18), 0.1);
    d = SDF.smin(d, SDF.ellipsoid(x, y, az, 0.33, 0.42, 0.08, 0.1, 0.08, 0.08), 0.05);
    d = SDF.smin(d, SDF.ellipsoid(x, y, az, 0.13, 0.05, 0.11, 0.13, 0.05, 0.065), 0.06);
    return SDF.smin(d, SDF.ellipsoid(x, y, az, -0.13, 0.045, 0.17, 0.18, 0.045, 0.07), 0.06);
  };
  const ears = (x, y, z) => SDF.planar(x, y, z, z > 0 ? fEarR : fEarL, EAR, 0.024, 0.021, 0.04);
  const puff = (x, y, z) => SDF.sphere(x, y, z, -0.38, 0.28, 0, 0.085);
  const eye = (x, y, z) => SDF.sphere(x, y, Math.abs(z), 0.385, 0.5, 0.105, 0.031);
  const body = (x, y, z) => Math.min(SDF.smin(SDF.smin(core(x, y, z), ears(x, y, z), 0.05), puff(x, y, z), 0.04), eye(x, y, z));
  const earLocal = (x, y, z) => {
    const F = z > 0 ? fEarR : fEarL, dx = x - F.o[0], dy = y - F.o[1], dz = z - F.o[2];
    return [dx * F.u[0] + dy * F.u[1] + dz * F.u[2], dx * F.v[0] + dy * F.v[1] + dz * F.v[2]];
  };

  return {
    id: 'dust-bunny', name: 'Dust Bunny', category: 'Lagomorph', status: 'prototype',
    description: 'Minimal second species that proves the engine is not shark-shaped.',
    display: {
      kicker: 'Prototype — validation fixture', title: ['Dust', 'Bunny'], theme: { accent: '#7d6a8c' },
      description: 'A deliberately small second species: floppy lattice ears, a hop-based locomotion capability under gravity, ' +
        'and its own reactions — built from a definition alone, with no changes to the engine.',
    },
    environment: { preset: 'studio', spawn: { pos: [0, 0.31, 0], yaw: 0.6 }, camera: { target: [0, 0.4, 0], distance: 4.6, yaw: 0.7, pitch: 0.36, follow: 0.7 } },
    createState: () => ({}),
    anatomy: {
      physicsSdf: body,
      parts: [{ id: 'body', sdf: body, resolution: 0.0135, bounds: { min: [-0.5, -0.02, -0.32], max: [0.52, 0.98, 0.32] },
        region(x, y, z) {
          if (eye(x, y, z) < 0.005) return 'eye';
          if (puff(x, y, z) < core(x, y, z) - 0.005) return 'puff';
          if (ears(x, y, z) < core(x, y, z) - 0.01) {
            const [u, v] = earLocal(x, y, z);
            return x > (z > 0 ? fEarR : fEarL).o[0] - 0.3 * v * 0.29 && Math.abs(u) < 0.04 && v > 0.04 && v < 0.33 ? 'earInner' : 'body';
          }
          return y < 0.3 && x > 0.05 && Math.abs(z) < 0.16 ? 'belly' : 'body';
        } }],
      anchors: {}, joints: {},
      landmarks: { nose: [0.45, 0.45, 0] },
      expressions: { channels: ['content'], spots: [
        { center: [0.37, 0.42, 0.13], radius: 0.06, color: '#f2a1a8', channel: 'content' },
        { center: [0.37, 0.42, -0.13], radius: 0.06, color: '#f2a1a8', channel: 'content' }] },
      decals({ surfacePoint, sdf }) {
        const tip = surfacePoint(sdf, [0.25, 0.445, 0], [1, 0, 0], 0.4, 0.003);
        const ys = surfacePoint(sdf, [0.25, 0.405, 0], [1, 0, 0], 0.4, 0.003);
        const l = surfacePoint(sdf, [0.25, 0.385, 0.03], [1, 0, 0.25], 0.4, 0.003), r = surfacePoint(sdf, [0.25, 0.385, -0.03], [1, 0, -0.25], 0.4, 0.003);
        return [
          { group: 'nose', region: 'nose', width: 0.011, points: [tip, ys] },
          { group: 'nose', region: 'nose', width: 0.005, points: [ys, l] },
          { group: 'nose', region: 'nose', width: 0.005, points: [ys, r] },
        ];
      },
    },
    physics: {
      latticeSpacing: 0.058, substeps: 8,
      shapeStiffness: [0.06, 0.45], volumeCompliance: [8e-8, 8e-10], inflate: [0.94, 1.06], grabCompliance: 0.005,
      unitMeters: 0.25, fillGramsPerLitre: 30, shellGrams: 45,
      stuffingAt(x, y, z) {
        if (ears(x, y, z) < core(x, y, z)) return { density: 0.35, stiffness: 0.22 + 0.5 * smoothstep(0.85, 0.6, y) };
        if (puff(x, y, z) < core(x, y, z)) return { density: 0.5, stiffness: 0.5 };
        return { density: 1, stiffness: 1 };
      },
    },
    locomotion: {
      type: 'hop', groundY: 0, headX: 0.2, squash: 0.2, stretch: 0.12, crouchTime: 0.17, hopHeight: 0.3, hopDistance: 0.6,
      cooldown: 0.25, turnGain: 16, turnDamp: 4, maxTurn: 20, uprightGain: 34, uprightDamp: 6, wallMargin: 0.7,
    },
    fur: {
      shells: 14, length: 0.03, density: 200, droop: 0.12, combRadius: 0.08,
      lagStiffness: 90, lagDamping: 0.3, inertia: 0.01, flow: 0.06,
      direction(x, y, z) {
        if (ears(x, y, z) < core(x, y, z)) return [0, 1, 0];
        if (puff(x, y, z) < core(x, y, z)) return [x + 0.38, y - 0.28, z];
        return [-1, -0.45, 0];
      },
    },
    materials: {
      regions: { body: { type: 'plush' }, belly: { type: 'plush', fur: 0.85 }, puff: { type: 'plush', fur: 1.7 },
        earInner: { type: 'felt' }, eye: { type: 'glass' }, nose: { type: 'embroidery' } },
      defaultPreset: 'oat',
      presets: [
        { id: 'oat', label: 'Oat', colors: { body: '#d2bc98', belly: '#f2e9d8', puff: '#f7f1e6', earInner: '#e7a7a4', eye: '#1b1410', nose: '#9c5560' } },
        { id: 'cocoa', label: 'Cocoa', colors: { body: '#7b5741', belly: '#c8a98b', puff: '#eadccb', earInner: '#d99592', eye: '#140e0b', nose: '#4e2c28' } },
        { id: 'snow', label: 'Snow', colors: { body: '#efebe4', belly: '#f8f6f1', puff: '#ffffff', earInner: '#f1b6b8', eye: '#22140f', nose: '#c46d79' } },
      ],
    },
    behaviors: {
      initial: 'sit',
      states: {
        sit: {
          kind: 'autonomous', priority: 0, resumable: true, label: 'Sitting',
          enter(ctx, d) { d.until = ctx.time + 1.5 + ctx.rand() * 2.5; },
          resume(ctx, d) { d.until = ctx.time + 1 + ctx.rand() * 2; },
          update(ctx, d) { ctx.intent.sniff = 0.3; if (ctx.flags.wander && ctx.time > d.until) return { next: ctx.rand() < 0.35 ? 'sniff' : 'hopAbout' }; },
        },
        hopAbout: {
          kind: 'autonomous', priority: 1, label: 'Hopping about',
          enter(ctx, d) { d.target = ctx.env.randomPoint(0.8, [0.1, 0.1]); },
          update(ctx, d, dt, t) {
            const c = ctx.body.c; ctx.intent.target = d.target; ctx.intent.hop = true;
            if (Math.hypot(c[0] - d.target[0], c[2] - d.target[2]) < 0.35 || t > 7) return 'done';
          },
        },
        sniff: { kind: 'autonomous', priority: 2, duration: 2.4, label: 'Sniffing', update(ctx) { ctx.intent.sniff = 1; ctx.intent.crouch = 0.2; } },
        hop: {
          kind: 'user', priority: 5, label: 'Hop',
          enter(ctx, d) { const b = ctx.body; d.target = [b.c[0] + b.f[0] * 0.7, 0, b.c[2] + b.f[2] * 0.7]; },
          update(ctx, d, dt, t) {
            if (ctx.locomotion.airborne) d.launched = true;
            ctx.intent.target = d.target; ctx.intent.hopNow = !d.launched;
            if ((d.launched && !ctx.locomotion.airborne) || t > 2.5) return 'done';
          },
        },
        binky: {
          kind: 'user', priority: 6, label: 'Binky!',
          update(ctx, d, dt, t) {
            const I = ctx.intent;
            if (ctx.locomotion.airborne) d.launched = true;
            I.hopNow = !d.launched; I.hopScale = 1.7; I.twist = 1; I.expr.content = 0.6; I.turn = false;
            if ((d.launched && !ctx.locomotion.airborne) || t > 2.5) return 'done';
          },
        },
        startle: {
          kind: 'reactive', priority: 7, label: 'Startled',
          enter(ctx, d, info) {
            const c = ctx.body.c, p = info?.point || c, a = v3.norm([c[0] - p[0] + 1e-3, 0, c[2] - p[2]]);
            d.target = [c[0] + a[0] * 1.2, 0, c[2] + a[2] * 1.2];
          },
          update(ctx, d, dt, t) {
            if (ctx.locomotion.airborne) d.launched = true;
            ctx.intent.target = d.target; ctx.intent.hopNow = !d.launched; ctx.intent.hopScale = 1.2;
            if ((d.launched && !ctx.locomotion.airborne) || t > 2) return 'done';
          },
        },
        held: { kind: 'reactive', priority: 8, label: 'Dangling', update(ctx) { ctx.intent.turn = false; if (!ctx.interaction.grabbed) return 'done'; } },
        groomed: {
          kind: 'reactive', priority: 3, label: 'Melting into the brush',
          enter(ctx, d) { d.until = ctx.time + 2.2; }, retrigger(ctx, d) { d.until = ctx.time + 2.2; },
          update(ctx, d) { ctx.intent.crouch = 0.5; ctx.intent.expr.content = 1; if (ctx.time > d.until) return 'done'; },
        },
      },
    },
    interactions: {
      tools: ['hand', 'finger', 'comb'], pickRadius: 0.03,
      regions: [
        { id: 'ears', label: 'Ears', test: (x, y) => y > 0.62 },
        { id: 'head', label: 'Head', sphere: [0.27, 0.47, 0, 0.21] },
        { id: 'tail', label: 'Tail puff', sphere: [-0.38, 0.28, 0, 0.12] },
      ],
      reactions: [
        { on: 'grab', behavior: 'held' },
        { on: 'poke', region: 'head', behavior: 'groomed' },
        { on: 'poke', behavior: 'startle' },
        { on: 'stroke', behavior: 'groomed' },
        { on: 'comb', behavior: 'groomed' },
      ],
    },
    actions: {
      hop: { run: (ctx) => ctx.behaviors.request('hop') },
      binky: { run: (ctx) => ctx.behaviors.request('binky') },
      sniff: { run: (ctx) => ctx.behaviors.request('sniff') },
    },
    controls: [
      { id: 'hop', type: 'button', label: 'Hop', action: 'hop', group: 'Actions', primary: true },
      { id: 'binky', type: 'button', label: 'Binky', action: 'binky', group: 'Actions', hint: 'A joyful twisting leap' },
      { id: 'sniff', type: 'button', label: 'Sniff', action: 'sniff', group: 'Actions' },
      { id: 'smooth', type: 'button', label: 'Smooth fur', action: 'engine.smoothFur', group: 'Actions' },
      { id: 'wander', type: 'toggle', label: 'Wander', bind: 'flags.wander', default: true, group: 'Behaviour' },
      { id: 'stuffing', type: 'slider', label: 'Stuffing', bind: 'physics.stuffing', min: 0.3, max: 1.5, step: 0.01, default: 1, group: 'Body', format: (v) => `${Math.round(v * 100)}%` },
      { id: 'pile', type: 'slider', label: 'Pile', bind: 'fur.pile', min: 0, max: 2, step: 0.01, default: 1, group: 'Body', format: (v) => `${(v * 0.03 * 250).toFixed(1)} mm` },
      { id: 'hopHeight', type: 'slider', label: 'Hop height', bind: 'locomotion.hopHeight', min: 0.4, max: 1.8, step: 0.01, default: 1, group: 'Body', format: (v) => `${v.toFixed(2)}×` },
      { id: 'fabric', type: 'preset', label: 'Fabric', bind: 'material.preset', default: 'oat', group: 'Fabric' },
    ],
    readouts: [
      { id: 'mass', label: 'Mass', unit: 'g', get: (r) => r.metrics.mass, format: (v) => v.toFixed(0) },
      { id: 'volume', label: 'Volume', unit: 'L', get: (r) => r.metrics.volume, format: (v) => v.toFixed(2) },
      { id: 'hops', label: 'Hops', unit: '', get: (r) => r.locomotion.hops, format: (v) => String(v) },
    ],
    notes: {
      anatomy: 'One blended SDF (body, haunches, head, cheeks, feet, puff tail, planar ears) with no attached parts and no joints.',
      implemented: [
        'Hop locomotion: crouch, ballistic launch under gravity, flight with stretch, and landing with floor friction.',
        'The ears are lightly stuffed lattice cells, so they flop and bounce on their own. They are not animated.',
        'A poke anywhere except the head startles it into a hop away; the comb, petting or a poke on the head make it relax.',
      ],
      approximations: [
        'This prototype only checks that the architecture works. Its behaviour set and anatomy are intentionally small.',
      ],
    },
  };
})();

/* Planned species: listed for the selector, never playable. */
const PlannedCat = { id: 'cat', name: 'Plush Cat', category: 'Felid', status: 'planned', description: 'Walk, stretch, pounce and groom.' };
const PlannedSongbird = { id: 'songbird', name: 'Plush Songbird', category: 'Passerine', status: 'planned', description: 'Fly, land, flap and preen.' };

[PlushHammerhead, DustBunny, PlannedCat, PlannedSongbird].forEach((s) => Engine.registerSpecies(s));
if (typeof module !== 'undefined') module.exports = { PlushHammerhead, DustBunny };
