'use strict';
/* =========================================================================
   PLUSH CREATURES — ENGINE
   Species-independent systems. Nothing in the engine may reference a
   particular animal: anatomy, physics tuning, locomotion choice, behaviours,
   interaction regions, materials and controls all arrive through a
   SpeciesDefinition object (see the SPECIES script further down).
   ========================================================================= */
const Engine = (() => {

/* ---------- 0. Math ---------------------------------------------------- */
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const wrapAngle = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};

/** Deterministic PRNG (mulberry32). */
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Column-major 3x3 from quaternion [x,y,z,w]. */
function quatToMat3(q, out = new Float64Array(9), o = 0) {
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y + w * z); out[2] = 2 * (x * z - w * y);
  out[3] = 2 * (x * y - w * z); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z + w * x);
  out[6] = 2 * (x * z + w * y); out[7] = 2 * (y * z - w * x); out[8] = 1 - 2 * (x * x + y * y);
  return out;
}
function quatAxisAngle(ax, ang) {
  const s = Math.sin(ang / 2);
  return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)];
}

/**
 * Rotation extraction (Müller et al. 2016, "A Robust Method to Extract the
 * Rotational Part of Deformations"). A is column-major 3x3, Q holds a warm-
 * started quaternion at offset o which is refined in place.
 */
function extractRotation(A, Q, o, iters) {
  let x = Q[o], y = Q[o + 1], z = Q[o + 2], w = Q[o + 3];
  for (let it = 0; it < iters; it++) {
    const r00 = 1 - 2 * (y * y + z * z), r10 = 2 * (x * y + w * z), r20 = 2 * (x * z - w * y);
    const r01 = 2 * (x * y - w * z), r11 = 1 - 2 * (x * x + z * z), r21 = 2 * (y * z + w * x);
    const r02 = 2 * (x * z + w * y), r12 = 2 * (y * z - w * x), r22 = 1 - 2 * (x * x + y * y);
    let ox = (r10 * A[2] - r20 * A[1]) + (r11 * A[5] - r21 * A[4]) + (r12 * A[8] - r22 * A[7]);
    let oy = (r20 * A[0] - r00 * A[2]) + (r21 * A[3] - r01 * A[5]) + (r22 * A[6] - r02 * A[8]);
    let oz = (r00 * A[1] - r10 * A[0]) + (r01 * A[4] - r11 * A[3]) + (r02 * A[7] - r12 * A[6]);
    const den = Math.abs(r00 * A[0] + r10 * A[1] + r20 * A[2] + r01 * A[3] + r11 * A[4] + r21 * A[5] +
      r02 * A[6] + r12 * A[7] + r22 * A[8]) + 1e-9;
    ox /= den; oy /= den; oz /= den;
    const ang = Math.sqrt(ox * ox + oy * oy + oz * oz);
    if (ang < 1e-9) break;
    const s = Math.sin(ang * 0.5) / ang, c = Math.cos(ang * 0.5);
    const ax = ox * s, ay = oy * s, az = oz * s;
    const nx = c * x + ax * w + ay * z - az * y;
    const ny = c * y - ax * z + ay * w + az * x;
    const nz = c * z + ax * y - ay * x + az * w;
    const nw = c * w - ax * x - ay * y - az * z;
    const il = 1 / Math.hypot(nx, ny, nz, nw);
    x = nx * il; y = ny * il; z = nz * il; w = nw * il;
  }
  Q[o] = x; Q[o + 1] = y; Q[o + 2] = z; Q[o + 3] = w;
}

/** Column-major 4x4 matrices (WebGPU clip space: z in [0,1]). */
const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[10] = far / (near - far); o[11] = -1; o[14] = (near * far) / (near - far);
    return o;
  },
  ortho(l, r, b, t, n, f) {
    const o = new Float32Array(16);
    o[0] = 2 / (r - l); o[5] = 2 / (t - b); o[10] = 1 / (n - f);
    o[12] = (l + r) / (l - r); o[13] = (t + b) / (b - t); o[14] = n / (n - f); o[15] = 1;
    return o;
  },
  lookAt(eye, target, up) {
    const z = v3.norm(v3.sub(eye, target)), x = v3.norm(v3.cross(up, z)), y = v3.cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -v3.dot(x, eye), -v3.dot(y, eye), -v3.dot(z, eye), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  invert(m) {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3], a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11], a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    const id = 1 / det, o = new Float32Array(16);
    o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * id; o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * id;
    o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * id; o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * id;
    o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * id; o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * id;
    o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * id; o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * id;
    o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * id; o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * id;
    o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * id; o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * id;
    o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * id; o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * id;
    o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * id; o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * id;
    return o;
  },
  xform(m, p) {
    const x = p[0], y = p[1], z = p[2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w, (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
      (m[2] * x + m[6] * y + m[10] * z + m[14]) / w];
  },
};

/** sRGB hex → linear [r,g,b]. */
function hexToLinear(hex) {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((i) => Math.pow(parseInt(h.slice(i, i + 2), 16) / 255, 2.2));
}

/* ---------- 1. Signed-distance toolkit (used by species anatomy) ------- */
const SDF = {
  smin(a, b, k) { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; },
  smax(a, b, k) { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.max(a, b) + h * h * k * 0.25; },
  sphere(x, y, z, cx, cy, cz, r) { const dx = x - cx, dy = y - cy, dz = z - cz; return Math.sqrt(dx * dx + dy * dy + dz * dz) - r; },
  ellipsoid(x, y, z, cx, cy, cz, rx, ry, rz) {
    const px = x - cx, py = y - cy, pz = z - cz;
    const ax = px / rx, ay = py / ry, az = pz / rz, bx = ax / rx, by = ay / ry, bz = az / rz;
    const k0 = Math.sqrt(ax * ax + ay * ay + az * az), k1 = Math.sqrt(bx * bx + by * by + bz * bz);
    return k1 < 1e-9 ? -Math.min(rx, ry, rz) : (k0 * (k0 - 1)) / k1;
  },
  /** Capsule with radius interpolated from ra (at a) to rb (at b). */
  cone(x, y, z, a, b, ra, rb) {
    const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
    const px = x - a[0], py = y - a[1], pz = z - a[2];
    const t = clamp((px * bx + py * by + pz * bz) / (bx * bx + by * by + bz * bz), 0, 1);
    const qx = px - bx * t, qy = py - by * t, qz = pz - bz * t;
    return Math.sqrt(qx * qx + qy * qy + qz * qz) - (ra + (rb - ra) * t);
  },
  /** Exact 2D polygon distance (Inigo Quilez). pts is flat [u0,v0,u1,v1,...]. */
  polygon(u, v, pts) {
    const N = pts.length / 2;
    let d = (u - pts[0]) ** 2 + (v - pts[1]) ** 2, s = 1;
    for (let i = 0, j = N - 1; i < N; j = i, i++) {
      const vix = pts[2 * i], viy = pts[2 * i + 1], vjx = pts[2 * j], vjy = pts[2 * j + 1];
      const ex = vjx - vix, ey = vjy - viy, wx = u - vix, wy = v - viy;
      const t = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1);
      const bx = wx - ex * t, by = wy - ey * t;
      d = Math.min(d, bx * bx + by * by);
      const c1 = v >= viy, c2 = v < vjy, c3 = ex * wy > ey * wx;
      if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
    }
    return s * Math.sqrt(d);
  },
  /** Rounded extrusion of a 2D distance d2 along w with half thickness h. */
  extrude(d2, w, h, r) {
    const qx = d2 + r, qy = Math.abs(w) - h + r;
    const mx = qx > 0 ? qx : 0, my = qy > 0 ? qy : 0;
    return Math.min(Math.max(qx, qy), 0) + Math.sqrt(mx * mx + my * my) - r;
  },
  /** Orthonormal frame for planar shapes: u (in-plane), v (in-plane), w (normal). */
  frame(origin, uDir, vDir) {
    const v = v3.norm(vDir);
    const u = v3.norm(v3.sub(uDir, v3.scale(v, v3.dot(uDir, v))));
    return { o: origin, u, v, w: v3.cross(u, v) };
  },
  /** Puffy planar appendage: polygon in a frame, extruded with a pillow taper toward its edges. */
  planar(x, y, z, F, poly, h, round, taper) {
    const dx = x - F.o[0], dy = y - F.o[1], dz = z - F.o[2];
    const w = dx * F.w[0] + dy * F.w[1] + dz * F.w[2];
    if (Math.abs(w) > h + 0.25) return Math.abs(w) - h;
    const u = dx * F.u[0] + dy * F.u[1] + dz * F.u[2];
    const v = dx * F.v[0] + dy * F.v[1] + dz * F.v[2];
    const d2 = SDF.polygon(u, v, poly);
    const hh = taper > 0 ? h * (0.4 + 0.6 * smoothstep(0, taper, -d2)) : h;
    return SDF.extrude(d2, w, hh, Math.min(round, hh * 0.92));
  },
};

/* ---------- 2. Procedural geometry: surface nets ----------------------- */
const CUBE_EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

/**
 * Naive surface nets over a sampled SDF. A coarse pre-pass skips exact
 * evaluation far from the surface. Vertices are then projected onto the
 * zero set and given analytic-gradient normals.
 */
function surfaceNets(fn, bmin, bmax, h) {
  const nx = Math.ceil((bmax[0] - bmin[0]) / h) + 1, ny = Math.ceil((bmax[1] - bmin[1]) / h) + 1;
  const nz = Math.ceil((bmax[2] - bmin[2]) / h) + 1, sy = nx, sz = nx * ny;
  const val = new Float32Array(nx * ny * nz);
  const C = 4, cnx = Math.ceil((nx - 1) / C) + 1, cny = Math.ceil((ny - 1) / C) + 1, cnz = Math.ceil((nz - 1) / C) + 1;
  const cval = new Float32Array(cnx * cny * cnz);
  for (let k = 0; k < cnz; k++) for (let j = 0; j < cny; j++) for (let i = 0; i < cnx; i++) {
    cval[i + cnx * (j + cny * k)] = fn(bmin[0] + i * C * h, bmin[1] + j * C * h, bmin[2] + k * C * h);
  }
  const thr = 1.6 * C * h;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const dc = cval[Math.round(i / C) + cnx * (Math.round(j / C) + cny * Math.round(k / C))];
    val[i + sy * j + sz * k] = Math.abs(dc) > thr ? dc : fn(bmin[0] + i * h, bmin[1] + j * h, bmin[2] + k * h);
  }
  const cx = nx - 1, cy = ny - 1;
  const cubeId = (i, j, k) => i + cx * (j + cy * k);
  const vid = new Int32Array(cx * cy * (nz - 1)).fill(-1);
  const pos = [], cv = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      const v = val[(i + (c & 1)) + sy * (j + ((c >> 1) & 1)) + sz * (k + (c >> 2))];
      cv[c] = v; if (v < 0) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let ax = 0, ay = 0, az = 0, cnt = 0;
    for (const [a, b] of CUBE_EDGES) {
      if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
      const t = cv[a] / (cv[a] - cv[b]);
      ax += (a & 1) + t * ((b & 1) - (a & 1));
      ay += ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1));
      az += (a >> 2) + t * ((b >> 2) - (a >> 2));
      cnt++;
    }
    vid[cubeId(i, j, k)] = pos.length / 3;
    pos.push(bmin[0] + (i + ax / cnt) * h, bmin[1] + (j + ay / cnt) * h, bmin[2] + (k + az / cnt) * h);
  }
  const idx = [];
  const quad = (a, b, c, d) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    const dac = (pos[3 * a] - pos[3 * c]) ** 2 + (pos[3 * a + 1] - pos[3 * c + 1]) ** 2 + (pos[3 * a + 2] - pos[3 * c + 2]) ** 2;
    const dbd = (pos[3 * b] - pos[3 * d]) ** 2 + (pos[3 * b + 1] - pos[3 * d + 1]) ** 2 + (pos[3 * b + 2] - pos[3 * d + 2]) ** 2;
    if (dac < dbd) idx.push(a, b, c, a, c, d); else idx.push(a, b, d, b, c, d);
  };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const v0 = val[i + sy * j + sz * k], in0 = v0 < 0;
    if (in0 !== (val[i + 1 + sy * j + sz * k] < 0)) {
      const a = vid[cubeId(i, j - 1, k - 1)], b = vid[cubeId(i, j, k - 1)], c = vid[cubeId(i, j, k)], d = vid[cubeId(i, j - 1, k)];
      if (in0) quad(a, b, c, d); else quad(d, c, b, a);
    }
    if (in0 !== (val[i + sy * (j + 1) + sz * k] < 0)) {
      const a = vid[cubeId(i - 1, j, k - 1)], b = vid[cubeId(i - 1, j, k)], c = vid[cubeId(i, j, k)], d = vid[cubeId(i, j, k - 1)];
      if (in0) quad(a, b, c, d); else quad(d, c, b, a);
    }
    if (in0 !== (val[i + sy * j + sz * (k + 1)] < 0)) {
      const a = vid[cubeId(i - 1, j - 1, k)], b = vid[cubeId(i, j - 1, k)], c = vid[cubeId(i, j, k)], d = vid[cubeId(i - 1, j, k)];
      if (in0) quad(a, b, c, d); else quad(d, c, b, a);
    }
  }
  const P = new Float32Array(pos), N = new Float32Array(pos.length);
  const e = h * 0.25;
  const grad = (x, y, z, out) => { // tetrahedral gradient (4 evaluations)
    const a = fn(x + e, y - e, z - e), b = fn(x - e, y - e, z + e), c = fn(x - e, y + e, z - e), d = fn(x + e, y + e, z + e);
    out[0] = a - b - c + d; out[1] = -a - b + c + d; out[2] = -a + b - c + d;
  };
  const g = [0, 0, 0];
  for (let v = 0; v < P.length; v += 3) {
    let x = P[v], y = P[v + 1], z = P[v + 2];
    grad(x, y, z, g);
    for (let it = 0; it < 2; it++) { // Newton projection onto the zero set: ∇f ≈ g / 4e
      const gl = Math.hypot(g[0], g[1], g[2]);
      if (gl < 1e-12) break;
      const s = clamp((fn(x, y, z) * 4 * e) / gl, -h * 0.6, h * 0.6);
      x -= (g[0] / gl) * s; y -= (g[1] / gl) * s; z -= (g[2] / gl) * s;
      grad(x, y, z, g);
    }
    const l2 = Math.hypot(g[0], g[1], g[2]) || 1;
    P[v] = x; P[v + 1] = y; P[v + 2] = z;
    N[v] = g[0] / l2; N[v + 1] = g[1] / l2; N[v + 2] = g[2] / l2;
  }
  return { positions: P, normals: N, indices: new Uint32Array(idx) };
}

/** March from `from` along `dir` and bisect to the first zero crossing of fn. */
function surfacePoint(fn, from, dir, maxDist = 2, step = 0.01) {
  const d = v3.norm(dir);
  let prev = fn(from[0], from[1], from[2]), t0 = 0;
  for (let t = step; t <= maxDist; t += step) {
    const p = [from[0] + d[0] * t, from[1] + d[1] * t, from[2] + d[2] * t];
    const cur = fn(p[0], p[1], p[2]);
    if ((prev < 0) !== (cur < 0)) {
      let a = t0, b = t;
      for (let i = 0; i < 24; i++) {
        const m = (a + b) / 2, fm = fn(from[0] + d[0] * m, from[1] + d[1] * m, from[2] + d[2] * m);
        if ((fm < 0) === (prev < 0)) a = m; else b = m;
      }
      const m = (a + b) / 2;
      return [from[0] + d[0] * m, from[1] + d[1] * m, from[2] + d[2] * m];
    }
    prev = cur; t0 = t;
  }
  return null;
}

/* ---------- 3. Soft body: lattice construction + XPBD solver ----------- */
/**
 * Builds a hexahedral lattice that covers the SDF interior and every render
 * vertex. Each hex gets 8 shared particles, a fill fraction (for mass and
 * volume read-outs), a stiffness multiplier from the species' stuffing
 * distribution, and a 5-tet split for volume constraints.
 */
function buildLattice(fn, verts, H, stuffingAt) {
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < verts.length; i += 3) for (let a = 0; a < 3; a++) {
    mn[a] = Math.min(mn[a], verts[i + a]); mx[a] = Math.max(mx[a], verts[i + a]);
  }
  const origin = mn.map((m) => m - H * 0.5);
  const dims = [0, 1, 2].map((a) => Math.ceil((mx[a] - origin[a]) / H + 0.5));
  const [cx, cy, cz] = dims;
  const cid = (i, j, k) => i + cx * (j + cy * k);
  const active = new Uint8Array(cx * cy * cz);
  for (let k = 0; k < cz; k++) for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) {
    if (fn(origin[0] + (i + 0.5) * H, origin[1] + (j + 0.5) * H, origin[2] + (k + 0.5) * H) < -H * 0.05) active[cid(i, j, k)] = 1;
  }
  const cellOfPoint = (x, y, z) => [
    clamp(Math.floor((x - origin[0]) / H), 0, cx - 1), clamp(Math.floor((y - origin[1]) / H), 0, cy - 1),
    clamp(Math.floor((z - origin[2]) / H), 0, cz - 1)];
  for (let v = 0; v < verts.length; v += 3) {
    const [i, j, k] = cellOfPoint(verts[v], verts[v + 1], verts[v + 2]);
    active[cid(i, j, k)] = 1;
  }
  const px = cx + 1, py = cy + 1;
  const pid = new Int32Array(px * py * (cz + 1)).fill(-1);
  const rest = [], cells = [], cellOf = new Int32Array(cx * cy * cz).fill(-1), cellIJK = [];
  for (let k = 0; k < cz; k++) for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) {
    if (!active[cid(i, j, k)]) continue;
    cellOf[cid(i, j, k)] = cells.length / 8;
    cellIJK.push(i, j, k);
    for (let c = 0; c < 8; c++) {
      const ci = i + (c & 1), cj = j + ((c >> 1) & 1), ck = k + (c >> 2), key = ci + px * (cj + py * ck);
      if (pid[key] < 0) { pid[key] = rest.length / 3; rest.push(origin[0] + ci * H, origin[1] + cj * H, origin[2] + ck * H); }
      cells.push(pid[key]);
    }
  }
  const n = rest.length / 3, nc = cells.length / 8;
  const fill = new Float32Array(nc), stiff = new Float32Array(nc), mass = new Float64Array(n);
  for (let c = 0; c < nc; c++) {
    const i = cellIJK[3 * c], j = cellIJK[3 * c + 1], k = cellIJK[3 * c + 2];
    let inside = 0;
    for (let s = 0; s < 27; s++) {
      const x = origin[0] + (i + ((s % 3) + 0.5) / 3) * H, y = origin[1] + (j + ((((s / 3) | 0) % 3) + 0.5) / 3) * H;
      const z = origin[2] + (k + (((s / 9) | 0) + 0.5) / 3) * H;
      if (fn(x, y, z) < 0) inside++;
    }
    fill[c] = Math.max(inside / 27, 0.06);
    const ccx = origin[0] + (i + 0.5) * H, ccy = origin[1] + (j + 0.5) * H, ccz = origin[2] + (k + 0.5) * H;
    const st = stuffingAt ? stuffingAt(ccx, ccy, ccz) : { density: 1, stiffness: 1 };
    stiff[c] = st.stiffness;
    for (let q = 0; q < 8; q++) mass[cells[8 * c + q]] += (fill[c] * st.density) / 8;
  }
  const surfOff = new Float32Array(n);
  for (let p = 0; p < n; p++) surfOff[p] = clamp(fn(rest[3 * p], rest[3 * p + 1], rest[3 * p + 2]), 0, H);
  // 5-tet split, alternating parity so neighbouring hexes share face diagonals.
  const TA = [[1, 2, 4, 7], [0, 1, 2, 4], [3, 1, 7, 2], [5, 1, 4, 7], [6, 2, 7, 4]];
  const TB = [[0, 3, 5, 6], [1, 0, 5, 3], [2, 0, 3, 6], [4, 0, 6, 5], [7, 3, 6, 5]];
  const tets = [];
  for (let c = 0; c < nc; c++) {
    const par = (cellIJK[3 * c] + cellIJK[3 * c + 1] + cellIJK[3 * c + 2]) & 1;
    for (const t of par ? TB : TA) tets.push(cells[8 * c + t[0]], cells[8 * c + t[1]], cells[8 * c + t[2]], cells[8 * c + t[3]]);
  }
  // Unique edges for the debug overlay.
  const edgeSet = new Set(), edges = [];
  for (let c = 0; c < nc; c++) for (const [a, b] of CUBE_EDGES) {
    const p = cells[8 * c + a], q = cells[8 * c + b], key = Math.min(p, q) * n + Math.max(p, q);
    if (!edgeSet.has(key)) { edgeSet.add(key); edges.push(p, q); }
  }
  /** Embed a rest-space point: returns containing cell and trilinear coordinates. */
  function embed(x, y, z) {
    let [i, j, k] = cellOfPoint(x, y, z);
    let c = cellOf[cid(i, j, k)];
    if (c < 0) { // nearest active cell (only happens for anchors outside the lattice)
      let best = Infinity;
      for (let q = 0; q < nc; q++) {
        const d = (origin[0] + (cellIJK[3 * q] + 0.5) * H - x) ** 2 + (origin[1] + (cellIJK[3 * q + 1] + 0.5) * H - y) ** 2 +
          (origin[2] + (cellIJK[3 * q + 2] + 0.5) * H - z) ** 2;
        if (d < best) { best = d; c = q; }
      }
      i = cellIJK[3 * c]; j = cellIJK[3 * c + 1]; k = cellIJK[3 * c + 2];
    }
    return { cell: c, u: (x - origin[0]) / H - i, v: (y - origin[1]) / H - j, w: (z - origin[2]) / H - k };
  }
  return { H, origin, dims, n, nc, rest: new Float64Array(rest), cells: new Int32Array(cells), fill, stiff, mass,
    surfOff, tets: new Int32Array(tets), edges: new Uint32Array(edges), embed };
}

/** Trilinear position + deformation-gradient columns for a lattice cell. */
function ffd(x, cells, c, u, v, w, outP, outF) {
  const b = 8 * c;
  const i0 = 3 * cells[b], i1 = 3 * cells[b + 1], i2 = 3 * cells[b + 2], i3 = 3 * cells[b + 3];
  const i4 = 3 * cells[b + 4], i5 = 3 * cells[b + 5], i6 = 3 * cells[b + 6], i7 = 3 * cells[b + 7];
  for (let a = 0; a < 3; a++) {
    const x0 = x[i0 + a], x1 = x[i1 + a], x2 = x[i2 + a], x3 = x[i3 + a], x4 = x[i4 + a], x5 = x[i5 + a], x6 = x[i6 + a], x7 = x[i7 + a];
    const a0 = x0 + (x1 - x0) * u, a1 = x2 + (x3 - x2) * u, a2 = x4 + (x5 - x4) * u, a3 = x6 + (x7 - x6) * u;
    const b0 = a0 + (a1 - a0) * v, b1 = a2 + (a3 - a2) * v;
    outP[a] = b0 + (b1 - b0) * w;
    if (outF) {
      outF[a] = (1 - v) * (1 - w) * (x1 - x0) + v * (1 - w) * (x3 - x2) + (1 - v) * w * (x5 - x4) + v * w * (x7 - x6);
      outF[3 + a] = (1 - u) * (1 - w) * (x2 - x0) + u * (1 - w) * (x3 - x1) + (1 - u) * w * (x6 - x4) + u * w * (x7 - x5);
      outF[6 + a] = (1 - u) * (1 - v) * (x4 - x0) + u * (1 - v) * (x5 - x1) + (1 - u) * v * (x6 - x2) + u * v * (x7 - x3);
    }
  }
}

/**
 * XPBD soft body: per-hex shape matching (goal constraints with compliance)
 * plus per-tet volume constraints, point attachments for grabbing, and
 * environment collision with per-particle surface offsets.
 */
class SoftBody {
  constructor(lat, cfg, spawn) {
    this.lat = lat; this.cfg = cfg; this.n = lat.n;
    const n = lat.n;
    this.x = new Float64Array(3 * n); this.xp = new Float64Array(3 * n); this.v = new Float64Array(3 * n);
    this.posed = new Float64Array(lat.rest); this.vPrev = new Float64Array(3 * n);
    const total = lat.mass.reduce((a, b) => a + b, 0), meanM = total / n;
    this.mass = new Float64Array(n); this.w = new Float64Array(n);
    for (let i = 0; i < n; i++) { this.mass[i] = Math.max(lat.mass[i], meanM * 0.25) / total; this.w[i] = 1 / this.mass[i]; }
    this.wMean = this.w.reduce((a, b) => a + b, 0) / n;
    this.cellInvM = new Float64Array(lat.nc); this.cellW = new Float64Array(lat.nc);
    for (let c = 0; c < lat.nc; c++) {
      let s = 0;
      for (let q = 0; q < 8; q++) s += this.mass[lat.cells[8 * c + q]];
      this.cellInvM[c] = 1 / s; this.cellW[c] = 8 / s;
    }
    // Rest centre of mass (unposed) used for the global body frame.
    let cm = [0, 0, 0], ms = 0;
    for (let i = 0; i < n; i++) { for (let a = 0; a < 3; a++) cm[a] += lat.rest[3 * i + a] * this.mass[i]; ms += this.mass[i]; }
    this.restCom = cm.map((c) => c / ms);
    this.cellQ = new Float64Array(4 * lat.nc);
    this.tetV0 = new Float64Array(lat.tets.length / 4);
    for (let t = 0; t < this.tetV0.length; t++) this.tetV0[t] = this.tetVolume(lat.rest, t);
    this.grab = null; this.contacts = 0;
    this.frame = { c: [0, 0, 0], R: new Float64Array(9), q: new Float64Array([0, 0, 0, 1]), v: [0, 0, 0], omega: [0, 0, 0], speed: 0 };
    this.substeps = cfg.substeps || 8;
    this.setStuffing(1);
    this.reset(spawn);
  }
  reset(spawn) {
    const { rest } = this.lat, n = this.n, cy = Math.cos(spawn.yaw), sy = Math.sin(spawn.yaw);
    for (let i = 0; i < n; i++) {
      const rx = rest[3 * i] - this.restCom[0], ry = rest[3 * i + 1] - this.restCom[1], rz = rest[3 * i + 2] - this.restCom[2];
      this.x[3 * i] = spawn.pos[0] + cy * rx + sy * rz;
      this.x[3 * i + 1] = spawn.pos[1] + ry;
      this.x[3 * i + 2] = spawn.pos[2] - sy * rx + cy * rz;
    }
    this.xp.set(this.x); this.v.fill(0); this.vPrev.fill(0); this.posed.set(rest);
    const q = quatAxisAngle([0, 1, 0], spawn.yaw);
    for (let c = 0; c < this.lat.nc; c++) this.cellQ.set(q, 4 * c);
    this.frame.q.set(q); this.grab = null;
    this.computeFrame(1 / 60);
  }
  /** Stuffing changes stiffness, volume compliance and the inflation target. */
  setStuffing(s) {
    const cfg = this.cfg, sdt = 1 / 60 / this.substeps, t = clamp((s - 0.3) / 1.2, 0, 1);
    this.stuffing = s;
    const k0 = lerp(cfg.shapeStiffness[0], cfg.shapeStiffness[1], t);
    this.shapeAlpha = (this.wMean * (1 - k0)) / k0 * sdt * sdt;          // XPBD compliance for goal constraints
    this.volAlpha = lerp(cfg.volumeCompliance[0], cfg.volumeCompliance[1], t);
    this.inflate = lerp(cfg.inflate[0], cfg.inflate[1], t);
    this.shapeScale = Math.cbrt(this.inflate);
  }
  tetVolume(X, t) {
    const T = this.lat.tets, a = 3 * T[4 * t], b = 3 * T[4 * t + 1], c = 3 * T[4 * t + 2], d = 3 * T[4 * t + 3];
    const e1x = X[b] - X[a], e1y = X[b + 1] - X[a + 1], e1z = X[b + 2] - X[a + 2];
    const e2x = X[c] - X[a], e2y = X[c + 1] - X[a + 1], e2z = X[c + 2] - X[a + 2];
    const e3x = X[d] - X[a], e3y = X[d + 1] - X[a + 1], e3z = X[d + 2] - X[a + 2];
    return ((e1y * e2z - e1z * e2y) * e3x + (e1z * e2x - e1x * e2z) * e3y + (e1x * e2y - e1y * e2x) * e3z) / 6;
  }
  /** Live enclosed volume: deformed hex volume weighted by SDF fill. */
  measureVolume() {
    let V = 0;
    for (let c = 0; c < this.lat.nc; c++) {
      let cv = 0;
      for (let t = 5 * c; t < 5 * c + 5; t++) cv += Math.abs(this.tetVolume(this.x, t));
      V += cv * this.lat.fill[c];
    }
    return V;
  }
  step(dt, env, forces) {
    const ns = this.substeps, sdt = dt / ns;
    this.vPrev.set(this.v);
    for (let s = 0; s < ns; s++) {
      forces(sdt, s);
      const x = this.x, xp = this.xp, v = this.v;
      for (let i = 0; i < x.length; i++) { xp[i] = x[i]; x[i] += v[i] * sdt; }
      this.solveShapes(sdt);
      this.solveVolumes(sdt);
      if (this.grab) this.solveGrab(sdt);
      this.collide(env);
      const inv = 1 / sdt;
      for (let i = 0; i < x.length; i++) v[i] = (x[i] - xp[i]) * inv;
    }
    this.computeFrame(dt);
  }
  /** Per-hex shape matching; rotation extraction is inlined (hot path). */
  solveShapes(sdt) {
    const cells = this.lat.cells, nc = this.lat.nc, stiff = this.lat.stiff, x = this.x, r = this.posed, w = this.w, Q = this.cellQ, sc = this.shapeScale;
    const aBase = this.shapeAlpha / (sdt * sdt), P = this._P || (this._P = new Int32Array(8)), M = this._M || (this._M = new Float64Array(8));
    const m = this.mass, cellInvM = this.cellInvM, cellW = this.cellW;
    for (let c = 0; c < nc; c++) {
      // Mass-weighted centroids + one gain per cell keep each projection momentum-conserving.
      const b = 8 * c, im = cellInvM[c];
      let cxp = 0, cyp = 0, czp = 0, rx = 0, ry = 0, rz = 0;
      for (let q = 0; q < 8; q++) {
        const pi = cells[b + q], p = 3 * pi, mq = m[pi]; P[q] = p; M[q] = mq;
        cxp += x[p] * mq; cyp += x[p + 1] * mq; czp += x[p + 2] * mq; rx += r[p] * mq; ry += r[p + 1] * mq; rz += r[p + 2] * mq;
      }
      cxp *= im; cyp *= im; czp *= im; rx *= im; ry *= im; rz *= im;
      let A0 = 0, A1 = 0, A2 = 0, A3 = 0, A4 = 0, A5 = 0, A6 = 0, A7 = 0, A8 = 0;
      for (let q = 0; q < 8; q++) {
        const p = P[q], mq = M[q];
        const dx = (x[p] - cxp) * mq, dy = (x[p + 1] - cyp) * mq, dz = (x[p + 2] - czp) * mq;
        const ex = r[p] - rx, ey = r[p + 1] - ry, ez = r[p + 2] - rz;
        A0 += dx * ex; A1 += dy * ex; A2 += dz * ex;
        A3 += dx * ey; A4 += dy * ey; A5 += dz * ey;
        A6 += dx * ez; A7 += dy * ez; A8 += dz * ez;
      }
      let qx = Q[4 * c], qy = Q[4 * c + 1], qz = Q[4 * c + 2], qw = Q[4 * c + 3];
      let r00, r10, r20, r01, r11, r21, r02, r12, r22;
      for (let it = 0; it < 3; it++) {
        r00 = 1 - 2 * (qy * qy + qz * qz); r10 = 2 * (qx * qy + qw * qz); r20 = 2 * (qx * qz - qw * qy);
        r01 = 2 * (qx * qy - qw * qz); r11 = 1 - 2 * (qx * qx + qz * qz); r21 = 2 * (qy * qz + qw * qx);
        r02 = 2 * (qx * qz + qw * qy); r12 = 2 * (qy * qz - qw * qx); r22 = 1 - 2 * (qx * qx + qy * qy);
        if (it === 2) break;
        let ox = (r10 * A2 - r20 * A1) + (r11 * A5 - r21 * A4) + (r12 * A8 - r22 * A7);
        let oy = (r20 * A0 - r00 * A2) + (r21 * A3 - r01 * A5) + (r22 * A6 - r02 * A8);
        let oz = (r00 * A1 - r10 * A0) + (r01 * A4 - r11 * A3) + (r02 * A7 - r12 * A6);
        const den = 1 / (Math.abs(r00 * A0 + r10 * A1 + r20 * A2 + r01 * A3 + r11 * A4 + r21 * A5 + r02 * A6 + r12 * A7 + r22 * A8) + 1e-12);
        ox *= den; oy *= den; oz *= den;
        const ang = Math.sqrt(ox * ox + oy * oy + oz * oz);
        if (ang < 1e-9) continue;
        const s = Math.sin(ang * 0.5) / ang, co = Math.cos(ang * 0.5), ax = ox * s, ay = oy * s, az = oz * s;
        const nx = co * qx + ax * qw + ay * qz - az * qy, ny = co * qy - ax * qz + ay * qw + az * qx;
        const nz = co * qz + ax * qy - ay * qx + az * qw, nw = co * qw - ax * qx - ay * qy - az * qz;
        const il = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        qx = nx * il; qy = ny * il; qz = nz * il; qw = nw * il;
      }
      Q[4 * c] = qx; Q[4 * c + 1] = qy; Q[4 * c + 2] = qz; Q[4 * c + 3] = qw;
      const k = cellW[c] / (cellW[c] + aBase / stiff[c]);
      for (let q = 0; q < 8; q++) {
        const p = P[q];
        const ex = (r[p] - rx) * sc, ey = (r[p + 1] - ry) * sc, ez = (r[p + 2] - rz) * sc;
        const gx = cxp + r00 * ex + r01 * ey + r02 * ez, gy = cyp + r10 * ex + r11 * ey + r12 * ez, gz = czp + r20 * ex + r21 * ey + r22 * ez;
        x[p] += (gx - x[p]) * k; x[p + 1] += (gy - x[p + 1]) * k; x[p + 2] += (gz - x[p + 2]) * k;
      }
    }
  }
  solveVolumes(sdt) {
    const T = this.lat.tets, x = this.x, w = this.w, V0 = this.tetV0, inf = this.inflate, at = this.volAlpha / (sdt * sdt);
    for (let t = 0; t < V0.length; t++) {
      const ia = T[4 * t], ib = T[4 * t + 1], ic = T[4 * t + 2], id = T[4 * t + 3];
      const a = 3 * ia, b = 3 * ib, c = 3 * ic, d = 3 * id;
      const e1x = x[b] - x[a], e1y = x[b + 1] - x[a + 1], e1z = x[b + 2] - x[a + 2];
      const e2x = x[c] - x[a], e2y = x[c + 1] - x[a + 1], e2z = x[c + 2] - x[a + 2];
      const e3x = x[d] - x[a], e3y = x[d + 1] - x[a + 1], e3z = x[d + 2] - x[a + 2];
      const g1x = (e2y * e3z - e2z * e3y) / 6, g1y = (e2z * e3x - e2x * e3z) / 6, g1z = (e2x * e3y - e2y * e3x) / 6;
      const g2x = (e3y * e1z - e3z * e1y) / 6, g2y = (e3z * e1x - e3x * e1z) / 6, g2z = (e3x * e1y - e3y * e1x) / 6;
      const g3x = (e1y * e2z - e1z * e2y) / 6, g3y = (e1z * e2x - e1x * e2z) / 6, g3z = (e1x * e2y - e1y * e2x) / 6;
      const g0x = -g1x - g2x - g3x, g0y = -g1y - g2y - g3y, g0z = -g1z - g2z - g3z;
      const V = e3x * g3x + e3y * g3y + e3z * g3z;
      const C = V - V0[t] * inf;
      const wsum = w[ia] * (g0x * g0x + g0y * g0y + g0z * g0z) + w[ib] * (g1x * g1x + g1y * g1y + g1z * g1z) +
        w[ic] * (g2x * g2x + g2y * g2y + g2z * g2z) + w[id] * (g3x * g3x + g3y * g3y + g3z * g3z);
      if (wsum < 1e-14) continue;
      const dl = -C / (wsum + at);
      x[a] += dl * w[ia] * g0x; x[a + 1] += dl * w[ia] * g0y; x[a + 2] += dl * w[ia] * g0z;
      x[b] += dl * w[ib] * g1x; x[b + 1] += dl * w[ib] * g1y; x[b + 2] += dl * w[ib] * g1z;
      x[c] += dl * w[ic] * g2x; x[c + 1] += dl * w[ic] * g2y; x[c + 2] += dl * w[ic] * g2z;
      x[d] += dl * w[id] * g3x; x[d + 1] += dl * w[id] * g3y; x[d + 2] += dl * w[id] * g3z;
    }
  }
  /** Grab: XPBD attachments of several particles to offsets around a target. */
  startGrab(point, radius) {
    const list = [], x = this.x;
    for (let i = 0; i < this.n; i++) {
      const d = Math.hypot(x[3 * i] - point[0], x[3 * i + 1] - point[1], x[3 * i + 2] - point[2]);
      if (d < radius) list.push({ i, off: [x[3 * i] - point[0], x[3 * i + 1] - point[1], x[3 * i + 2] - point[2]], k: 1 - d / radius });
    }
    if (!list.length) return false;
    this.grab = { list, target: point.slice() };
    return true;
  }
  solveGrab(sdt) {
    const { list, target } = this.grab, x = this.x, alpha = this.cfg.grabCompliance / (sdt * sdt);
    for (const g of list) {
      const p = 3 * g.i, k = (this.w[g.i] / (this.w[g.i] + alpha)) * (0.35 + 0.65 * g.k);
      for (let a = 0; a < 3; a++) x[p + a] += (target[a] + g.off[a] - x[p + a]) * k;
    }
  }
  collide(env) {
    const x = this.x, xp = this.xp, off = this.lat.surfOff, b = env.bounds, mu = env.friction;
    let contacts = 0;
    for (let i = 0; i < this.n; i++) {
      const p = 3 * i, o = off[i] * 0.9;
      if (x[p + 1] < b.min[1] - o) {
        x[p + 1] = b.min[1] - o; contacts++;
        x[p] = xp[p] + (x[p] - xp[p]) * (1 - mu); x[p + 2] = xp[p + 2] + (x[p + 2] - xp[p + 2]) * (1 - mu);
      } else if (x[p + 1] > b.max[1] + o) x[p + 1] = b.max[1] + o;
      if (x[p] < b.min[0] - o) x[p] = b.min[0] - o; else if (x[p] > b.max[0] + o) x[p] = b.max[0] + o;
      if (x[p + 2] < b.min[2] - o) x[p + 2] = b.min[2] - o; else if (x[p + 2] > b.max[2] + o) x[p + 2] = b.max[2] + o;
    }
    this.contacts = contacts;
  }
  /** Global body frame via whole-body shape matching (centre, rotation, ω). */
  computeFrame(dt) {
    const x = this.x, v = this.v, m = this.mass, rest = this.lat.rest, rc = this.restCom, F = this.frame;
    let cx = 0, cy = 0, cz = 0, vx = 0, vy = 0, vz = 0;
    for (let i = 0; i < this.n; i++) {
      cx += x[3 * i] * m[i]; cy += x[3 * i + 1] * m[i]; cz += x[3 * i + 2] * m[i];
      vx += v[3 * i] * m[i]; vy += v[3 * i + 1] * m[i]; vz += v[3 * i + 2] * m[i];
    }
    const A = new Float64Array(9);
    for (let i = 0; i < this.n; i++) {
      const dx = x[3 * i] - cx, dy = x[3 * i + 1] - cy, dz = x[3 * i + 2] - cz;
      const ex = rest[3 * i] - rc[0], ey = rest[3 * i + 1] - rc[1], ez = rest[3 * i + 2] - rc[2];
      A[0] += m[i] * dx * ex; A[1] += m[i] * dy * ex; A[2] += m[i] * dz * ex;
      A[3] += m[i] * dx * ey; A[4] += m[i] * dy * ey; A[5] += m[i] * dz * ey;
      A[6] += m[i] * dx * ez; A[7] += m[i] * dy * ez; A[8] += m[i] * dz * ez;
    }
    const q0 = Array.from(F.q);
    extractRotation(A, F.q, 0, 8);
    quatToMat3(F.q, F.R);
    // ω from quaternion change: dq = q * conj(q0)
    const q = F.q, cx0 = -q0[0], cy0 = -q0[1], cz0 = -q0[2], cw0 = q0[3];
    let dxq = q[3] * cx0 + q[0] * cw0 + q[1] * cz0 - q[2] * cy0;
    let dyq = q[3] * cy0 - q[0] * cz0 + q[1] * cw0 + q[2] * cx0;
    let dzq = q[3] * cz0 + q[0] * cy0 - q[1] * cx0 + q[2] * cw0;
    let dwq = q[3] * cw0 - q[0] * cx0 - q[1] * cy0 - q[2] * cz0;
    if (dwq < 0) { dxq = -dxq; dyq = -dyq; dzq = -dzq; dwq = -dwq; }
    const sl = Math.hypot(dxq, dyq, dzq), ang = 2 * Math.atan2(sl, dwq), k = sl > 1e-9 ? ang / sl / dt : 0;
    F.omega = [dxq * k, dyq * k, dzq * k];
    F.c = [cx, cy, cz]; F.v = [vx, vy, vz]; F.speed = Math.hypot(vx, vy, vz);
    F.f = [F.R[0], F.R[1], F.R[2]]; F.u = [F.R[3], F.R[4], F.R[5]]; F.l = [F.R[6], F.R[7], F.R[8]];
  }
}
