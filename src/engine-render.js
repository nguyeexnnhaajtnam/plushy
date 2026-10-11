
/* ---------- 12. Rendering system (WebGPU) ------------------------------ */
const WGSL = /* wgsl */`
struct Frame {
  viewProj: mat4x4f, lightViewProj: mat4x4f,
  eye: vec4f, keyDir: vec4f, keyColor: vec4f, fillDir: vec4f, fillColor: vec4f,
  skyColor: vec4f, groundColor: vec4f, bgColor: vec4f,
  fur: vec4f,      // pile length, shell count, strand density, droop
  env: vec4f,      // floor y, caustics, fade radius, unused
  bounds: vec4f,   // min x, min z, max x, max z
  floorColor: vec4f,
};
struct Material { color: vec4f, props: vec4f };
struct Spot { center: vec4f, color: vec4f };
struct Decal { a: vec4f, b: vec4f, p: vec4f };
struct CreatureU {
  attach: array<mat4x4f, 4>,
  materials: array<Material, 16>,
  spots: array<Spot, 4>,
  expr: vec4f,
  groups: array<vec4u, 32>,
  decals: array<Decal, 128>,
};
struct Props { model: array<mat4x4f, 8> };

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;
@group(0) @binding(2) var<storage, read> lagBuf: array<vec4f>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<uniform> C: CreatureU;
@group(0) @binding(5) var<uniform> P: Props;
@group(1) @binding(0) var shadowTex: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;

struct VIn {
  @location(0) restPos: vec3f,
  @location(1) restNrm: vec3f,
  @location(2) uvw: vec3f,
  @location(3) info: vec4u,     // lattice cell, attachment, region pair (lo: +side, hi: -side), decal-group mask
  @location(4) furDir: vec4f,   // xyz: default nap direction, w: signed region-boundary field
  @location(5) comb: vec3f,
};
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) nrm: vec3f,
  @location(2) rest: vec3f,
  @location(3) @interpolate(flat) info: vec2u,
  @location(4) shell: f32,
  @location(5) grain: f32,
  @location(6) restN: vec3f,
  @location(7) field: f32,
};
struct Skinned { pos: vec3f, nrm: vec3f, comb: vec3f, lag: vec3f };

fn skin(v: VIn) -> Skinned {
  var s: Skinned;
  if (v.info.y > 0u) {
    let M = C.attach[v.info.y - 1u];
    s.pos = (M * vec4f(v.restPos, 1.0)).xyz;
    s.nrm = (M * vec4f(v.restNrm, 0.0)).xyz;
    s.comb = (M * vec4f(v.comb, 0.0)).xyz;
    s.lag = vec3f(0.0);
    return s;
  }
  let b = v.info.x * 8u;
  let x0 = particles[cells[b]].xyz;      let x1 = particles[cells[b + 1u]].xyz;
  let x2 = particles[cells[b + 2u]].xyz; let x3 = particles[cells[b + 3u]].xyz;
  let x4 = particles[cells[b + 4u]].xyz; let x5 = particles[cells[b + 5u]].xyz;
  let x6 = particles[cells[b + 6u]].xyz; let x7 = particles[cells[b + 7u]].xyz;
  let u = v.uvw.x; let w = v.uvw.z; let t = v.uvw.y;
  s.pos = mix(mix(mix(x0, x1, u), mix(x2, x3, u), t), mix(mix(x4, x5, u), mix(x6, x7, u), t), w);
  let du = (1.0 - t) * (1.0 - w) * (x1 - x0) + t * (1.0 - w) * (x3 - x2) + (1.0 - t) * w * (x5 - x4) + t * w * (x7 - x6);
  let dv = (1.0 - u) * (1.0 - w) * (x2 - x0) + u * (1.0 - w) * (x3 - x1) + (1.0 - u) * w * (x6 - x4) + u * w * (x7 - x5);
  let dw = (1.0 - u) * (1.0 - t) * (x4 - x0) + u * (1.0 - t) * (x5 - x1) + (1.0 - u) * t * (x6 - x2) + u * t * (x7 - x3);
  s.nrm = v.restNrm.x * cross(dv, dw) + v.restNrm.y * cross(dw, du) + v.restNrm.z * cross(du, dv);
  s.comb = du * v.comb.x + dv * v.comb.y + dw * v.comb.z;
  let l0 = lagBuf[cells[b]].xyz;      let l1 = lagBuf[cells[b + 1u]].xyz;
  let l2 = lagBuf[cells[b + 2u]].xyz; let l3 = lagBuf[cells[b + 3u]].xyz;
  let l4 = lagBuf[cells[b + 4u]].xyz; let l5 = lagBuf[cells[b + 5u]].xyz;
  let l6 = lagBuf[cells[b + 6u]].xyz; let l7 = lagBuf[cells[b + 7u]].xyz;
  s.lag = mix(mix(mix(l0, l1, u), mix(l2, l3, u), t), mix(mix(l4, l5, u), mix(l6, l7, u), t), w);
  return s;
}

fn safeNorm(v: vec3f, fallback: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-8) { return fallback; }
  return v / l;
}

@vertex fn vsCreature(v: VIn, @builtin(instance_index) inst: u32) -> VOut {
  let s = skin(v);
  let n = safeNorm(s.nrm, vec3f(0.0, 1.0, 0.0));
  let mat = C.materials[select(v.info.z >> 8u, v.info.z & 255u, v.furDir.w >= 0.0)];
  let t = f32(inst) / max(F.fur.y, 1.0);
  var len = F.fur.x * mat.props.x;
  if (v.info.w != 0u && inst > 0u) { len *= decalPileScale(v.restPos, v.info.w); }
  var comb = s.comb - n * dot(s.comb, n);
  comb = safeNorm(comb, vec3f(0.0));
  let bend = comb * 0.6 + s.lag + vec3f(0.0, -F.fur.w, 0.0);
  let p = s.pos + n * (len * t) + bend * (len * t * t);
  var o: VOut;
  o.pos = F.viewProj * vec4f(p, 1.0);
  o.world = p;
  o.nrm = safeNorm(n + bend * t * 0.35, n);
  o.rest = v.restPos;
  o.info = vec2u(v.info.z, v.info.w);
  o.shell = t;
  o.grain = dot(safeNorm(v.comb, vec3f(0.0)), safeNorm(v.furDir.xyz, vec3f(0.0)));
  if (length(v.furDir.xyz) < 0.5) { o.grain = 1.0; }
  o.restN = v.restNrm;
  o.field = v.furDir.w;
  return o;
}

@vertex fn vsShadow(v: VIn) -> @builtin(position) vec4f {
  return F.lightViewProj * vec4f(skin(v).pos, 1.0);
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash3(c: vec3f) -> vec3f {
  let i = vec3u(vec3i(c) + vec3i(8192));
  let a = pcg(i.x ^ pcg(i.y ^ pcg(i.z)));
  return vec3f(f32(a & 1023u), f32((a >> 10u) & 1023u), f32((a >> 20u) & 1023u)) / 1023.0;
}

fn encode(c: vec3f) -> vec3f { return pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)); }

fn shadowAt(world: vec3f, n: vec3f) -> f32 {
  let lp = F.lightViewProj * vec4f(world + n * 0.015, 1.0);
  let uv = vec2f(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || lp.z > 1.0) { return 1.0; }
  let texel = 1.0 / 2048.0;
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(shadowTex, shadowSamp, uv + vec2f(f32(x), f32(y)) * texel * 1.6, lp.z - 0.0015);
    }
  }
  return mix(1.0, sum / 9.0, F.keyDir.w);
}

fn studioEnv(r: vec3f) -> vec3f {
  var c = mix(F.groundColor.rgb * 0.6, F.skyColor.rgb * 1.15, smoothstep(-0.3, 0.8, r.y));
  c += F.keyColor.rgb * pow(max(dot(r, normalize(F.keyDir.xyz)), 0.0), 60.0) * 3.0;
  c += F.fillColor.rgb * pow(max(dot(r, normalize(F.fillDir.xyz)), 0.0), 16.0) * 0.6;
  return c;
}

struct Surf { color: vec3f, kind: f32, furMul: f32, sheen: f32, gloss: f32, cut: f32 };

// x: hit, y: region, z: distance along stitch line, w: cuts fur
fn decalAt(p: vec3f, mask: u32) -> vec4f {
  var res = vec4f(0.0);
  var m = mask;
  for (var guard = 0; guard < 32; guard++) {
    if (m == 0u) { break; }
    let g = firstTrailingBit(m);
    m = m & (m - 1u);
    let gr = C.groups[g];
    for (var k = gr.x; k < gr.x + gr.y; k++) {
      let d = C.decals[k];
      let ab = d.b.xyz - d.a.xyz;
      let L = max(length(ab), 1e-6);
      let h = clamp(dot(p - d.a.xyz, ab) / (L * L), 0.0, 1.0);
      let dist = length(p - d.a.xyz - ab * h);
      if (dist < d.a.w) {
        let along = h * L + d.p.w;
        var on = true;
        if (d.p.x > 0.0) { on = fract(along / d.p.x) < d.p.y; }
        if (on) { res = vec4f(1.0, d.b.w, along, d.p.z); }
      }
    }
  }
  return res;
}

// Embroidery and stitches pull the pile down around them (quilted groove).
fn decalPileScale(p: vec3f, mask: u32) -> f32 {
  var k = 1.0;
  var m = mask;
  for (var guard = 0; guard < 32; guard++) {
    if (m == 0u) { break; }
    let g = firstTrailingBit(m);
    m = m & (m - 1u);
    let gr = C.groups[g];
    for (var i = gr.x; i < gr.x + gr.y; i++) {
      let d = C.decals[i];
      if (d.p.z < 0.5) { continue; }
      let ab = d.b.xyz - d.a.xyz;
      let h = clamp(dot(p - d.a.xyz, ab) / max(dot(ab, ab), 1e-10), 0.0, 1.0);
      let dist = length(p - d.a.xyz - ab * h);
      k = min(k, mix(0.15, 1.0, smoothstep(d.a.w * 0.5, d.a.w + 0.02, dist)));
    }
  }
  return k;
}

fn surfaceAt(rest: vec3f, regions: u32, field: f32, mask: u32) -> Surf {
  var s: Surf;
  var region = select(regions >> 8u, regions & 255u, field >= 0.0);
  let dh = decalAt(rest, mask);
  s.cut = 0.0;
  if (dh.x > 0.5) { region = u32(dh.y); s.cut = dh.w; }
  let m = C.materials[region];
  s.color = m.color.rgb; s.kind = m.color.a; s.furMul = m.props.x; s.sheen = m.props.y; s.gloss = m.props.z;
  if (dh.x > 0.5 && s.kind > 2.5 && s.kind < 3.5) { s.color *= 0.8 + 0.2 * sin(dh.z * 1800.0); }
  if (s.kind < 1.5) {
    for (var i = 0; i < 4; i++) {
      let sp = C.spots[i];
      if (sp.center.w <= 0.0) { continue; }
      let amt = C.expr[u32(sp.color.w)] * (1.0 - smoothstep(sp.center.w * 0.3, sp.center.w, distance(rest, sp.center.xyz)));
      s.color = mix(s.color, sp.color.rgb, clamp(amt, 0.0, 1.0) * 0.8);
    }
  }
  return s;
}

fn shade(s: Surf, world: vec3f, n: vec3f, ao: f32, grain: f32) -> vec3f {
  let v = normalize(F.eye.xyz - world);
  let l = normalize(F.keyDir.xyz);
  let sh = shadowAt(world, n);
  let ndl = dot(n, l);
  let ndv = clamp(dot(n, v), 0.0, 1.0);
  let hemi = mix(F.groundColor.rgb, F.skyColor.rgb, n.y * 0.5 + 0.5);
  if (s.kind > 1.5 && s.kind < 2.5) {           // glass eye
    let r = reflect(-v, n);
    let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let h = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), 260.0) * 7.0 * sh;
    return s.color * (0.25 + 0.6 * max(ndl, 0.0) * sh) + studioEnv(r) * (0.06 + fres * 0.75) + F.keyColor.rgb * spec;
  }
  var base = s.color;
  if (s.kind < 0.5) { base *= mix(0.74, 1.04, grain * 0.5 + 0.5); }   // minky nap: against-grain reads darker
  let wrap = clamp((ndl + 0.45) / 1.45, 0.0, 1.0);
  let key = F.keyColor.rgb * wrap * mix(0.35, 1.0, sh);
  let fill = F.fillColor.rgb * clamp(dot(n, normalize(F.fillDir.xyz)) * 0.5 + 0.5, 0.0, 1.0);
  var col = base * (key + fill + hemi * ao);
  let rim = pow(1.0 - ndv, 2.5) * s.sheen;
  col += rim * (hemi * 0.5 + F.keyColor.rgb * 0.25 * sh) * (base * 1.4 + vec3f(0.12));
  if (s.gloss > 0.0) {
    let h = normalize(l + v);
    col += F.keyColor.rgb * pow(max(dot(n, h), 0.0), 36.0) * s.gloss * sh * 0.5;
  }
  return col;
}

fn fog(c: vec3f, world: vec3f) -> vec3f {
  let d = distance(world, F.eye.xyz);
  return mix(c, F.bgColor.rgb, smoothstep(9.0, 26.0, d) * 0.7);
}

@fragment fn fsCreature(in: VOut) -> @location(0) vec4f {
  let s = surfaceAt(in.rest, in.info.x, in.field, in.info.y);
  let n = normalize(in.nrm);
  let furred = s.furMul > 0.0 && F.fur.y > 0.0 && s.cut < 0.5;
  var ao = 1.0;
  if (furred) { ao = 0.6; }
  var col = shade(s, in.world, n, ao, in.grain);
  if (furred) { col *= 0.66; }
  return vec4f(encode(fog(col, in.world)), 1.0);
}

@fragment fn fsFur(in: VOut) -> @location(0) vec4f {
  let s = surfaceAt(in.rest, in.info.x, in.field, in.info.y);
  let t = in.shell;
  let q = in.rest * F.fur.z;
  let h1 = hash3(floor(q));
  let f = fract(q) - 0.5 - (h1 - 0.5) * 0.45;
  let rn = normalize(in.restN);
  let r = length(f - rn * dot(f, rn));
  let height = 0.5 + 0.5 * h1.x;
  let radius = 0.62 * (1.0 - t / height);
  var cov = (1.0 - smoothstep(radius - 0.1, radius + 0.06, r)) * step(t, height);
  if (s.furMul <= 0.0 || s.cut > 0.5) { cov = 0.0; }
  if (cov < 0.03) { discard; }
  let n = normalize(in.nrm);
  var col = shade(s, in.world, n, mix(0.62, 1.0, t), in.grain) * mix(0.66, 1.0, t) * (0.9 + 0.2 * h1.y);
  return vec4f(encode(fog(col, in.world)), cov);
}

// ---- floor -------------------------------------------------------------
struct FOut { @builtin(position) pos: vec4f, @location(0) world: vec3f };
@vertex fn vsFloor(@builtin(vertex_index) vi: u32) -> FOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi] * 40.0;
  var o: FOut;
  o.world = vec3f(c.x, F.env.x, c.y);
  o.pos = F.viewProj * vec4f(o.world, 1.0);
  return o;
}
fn caustic(p: vec2f, t: f32) -> f32 {
  var q = p;
  var c = 0.0;
  for (var i = 0; i < 3; i++) {
    let fi = f32(i);
    q = vec2f(q.x + 0.6 * sin(q.y * 1.3 + t * 0.7 + fi), q.y + 0.6 * cos(q.x * 1.1 - t * 0.6 + fi * 1.7));
    c += 1.0 / (1.0 + 7.0 * abs(sin(q.x * 1.7) + sin(q.y * 1.9)));
  }
  return c / 3.0;
}
@fragment fn fsFloor(in: FOut) -> @location(0) vec4f {
  let p = in.world;
  let center = vec2f(F.bounds.x + F.bounds.z, F.bounds.y + F.bounds.w) * 0.5;
  let half = vec2f(F.bounds.z - F.bounds.x, F.bounds.w - F.bounds.y) * 0.5;
  let q = abs(p.xz - center) - half + vec2f(0.45);
  let dm = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - 0.45;
  var base = F.floorColor.rgb;
  let fib = hash3(vec3f(floor(p.xz * 140.0), 3.0));
  base *= 0.96 + 0.06 * fib.x;
  let inMat = 1.0 - smoothstep(-0.01, 0.01, dm);
  base = mix(base * 0.9, base, inMat);
  let stitch = (1.0 - smoothstep(0.0, 0.012, abs(dm + 0.09))) * step(0.45, fract((p.x + p.z) * 9.0));
  base = mix(base, base * 0.72, stitch * inMat);
  if (F.env.y > 0.5) { base += vec3f(0.85, 0.95, 1.0) * caustic(p.xz * 1.3, F.eye.w) * 0.16 * inMat; }
  let n = vec3f(0.0, 1.0, 0.0);
  let sh = shadowAt(p, n);
  let col = base * (F.keyColor.rgb * max(dot(n, normalize(F.keyDir.xyz)), 0.0) * mix(0.3, 1.0, sh) + F.skyColor.rgb * 0.75 + F.fillColor.rgb * 0.2);
  let r = length(p.xz - center);
  let fade = smoothstep(F.env.z * 0.45, F.env.z, r);
  return vec4f(encode(mix(col, F.bgColor.rgb, fade)), 1.0);
}

// ---- props (environment items) ------------------------------------------
struct PIn { @location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) col: vec3f };
struct POut { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) nrm: vec3f, @location(2) col: vec3f };
@vertex fn vsProp(v: PIn, @builtin(instance_index) i: u32) -> POut {
  let M = P.model[i];
  var o: POut;
  o.world = (M * vec4f(v.pos, 1.0)).xyz;
  o.nrm = (M * vec4f(v.nrm, 0.0)).xyz;
  o.col = v.col;
  o.pos = F.viewProj * vec4f(o.world, 1.0);
  return o;
}
@vertex fn vsPropShadow(v: PIn, @builtin(instance_index) i: u32) -> @builtin(position) vec4f {
  return F.lightViewProj * (P.model[i] * vec4f(v.pos, 1.0));
}
@fragment fn fsProp(in: POut) -> @location(0) vec4f {
  var s: Surf;
  s.color = in.col; s.kind = 1.0; s.furMul = 0.0; s.sheen = 0.35; s.gloss = 0.0; s.cut = 0.0;
  if (in.col.r + in.col.g + in.col.b < 0.05) { s.kind = 2.0; }
  let n = safeNorm(in.nrm, vec3f(0.0, 1.0, 0.0));
  return vec4f(encode(fog(shade(s, in.world, n, 1.0, 1.0), in.world)), 1.0);
}

// ---- lattice debug lines -------------------------------------------------
@vertex fn vsLines(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return F.viewProj * vec4f(particles[vi].xyz, 1.0);
}
@fragment fn fsLines() -> @location(0) vec4f { return vec4f(0.12, 0.1, 0.09, 0.5); }
`;

class GPUUnavailable extends Error {}

const FRAME_FLOATS = 80, CREATURE_BYTES = 7568, PROPS_BYTES = 512, SHADOW_SIZE = 2048;

class Renderer {
  static async create(canvas) {
    if (!('gpu' in navigator)) throw new GPUUnavailable('This browser does not expose WebGPU (navigator.gpu is missing).');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new GPUUnavailable('WebGPU is present, but no compatible GPU adapter was found.');
    const device = await adapter.requestDevice();
    const r = new Renderer(canvas, device);
    await r.checkShaders();
    return r;
  }
  constructor(canvas, device) {
    this.canvas = canvas; this.device = device;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.sampleCount = 4;
    const U = GPUBufferUsage;
    this.frameData = new Float32Array(FRAME_FLOATS);
    this.frameBuf = device.createBuffer({ size: FRAME_FLOATS * 4, usage: U.UNIFORM | U.COPY_DST });
    this.creatureBuf = device.createBuffer({ size: CREATURE_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.creatureData = new ArrayBuffer(CREATURE_BYTES);
    this.propsBuf = device.createBuffer({ size: PROPS_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.propsData = new Float32Array(PROPS_BYTES / 4);
    this.placeholder = device.createBuffer({ size: 64, usage: U.STORAGE });
    this.shadowTex = device.createTexture({ size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.module = device.createShaderModule({ code: WGSL });
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.bgl0 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ] });
    this.bgl1 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ] });
    this.bg1 = device.createBindGroup({ layout: this.bgl1, entries: [{ binding: 0, resource: this.shadowTex.createView() }, { binding: 1, resource: this.shadowSampler }] });
    const mainLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, this.bgl1] });
    const shadowLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0] });
    const creatureBuffers = [
      { arrayStride: 68, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x3' }, { shaderLocation: 3, offset: 36, format: 'uint32x4' },
        { shaderLocation: 4, offset: 52, format: 'float32x4' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x3' }] },
    ];
    const propBuffers = [{ arrayStride: 36, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
      { shaderLocation: 2, offset: 24, format: 'float32x3' }] }];
    const ms = { count: this.sampleCount };
    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    const target = [{ format: this.format }];
    const m = this.module;
    this.pipes = {
      creature: device.createRenderPipeline({ layout: mainLayout, vertex: { module: m, entryPoint: 'vsCreature', buffers: creatureBuffers },
        fragment: { module: m, entryPoint: 'fsCreature', targets: target }, primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: depth, multisample: ms }),
      fur: device.createRenderPipeline({ layout: mainLayout, vertex: { module: m, entryPoint: 'vsCreature', buffers: creatureBuffers },
        fragment: { module: m, entryPoint: 'fsFur', targets: target }, primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: depth, multisample: { count: this.sampleCount, alphaToCoverageEnabled: true } }),
      shadow: device.createRenderPipeline({ layout: shadowLayout, vertex: { module: m, entryPoint: 'vsShadow', buffers: creatureBuffers },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 } }),
      propShadow: device.createRenderPipeline({ layout: shadowLayout, vertex: { module: m, entryPoint: 'vsPropShadow', buffers: propBuffers },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 } }),
      floor: device.createRenderPipeline({ layout: mainLayout, vertex: { module: m, entryPoint: 'vsFloor' },
        fragment: { module: m, entryPoint: 'fsFloor', targets: target }, primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depth, multisample: ms }),
      prop: device.createRenderPipeline({ layout: mainLayout, vertex: { module: m, entryPoint: 'vsProp', buffers: propBuffers },
        fragment: { module: m, entryPoint: 'fsProp', targets: target }, primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: depth, multisample: ms }),
      lines: device.createRenderPipeline({ layout: mainLayout, vertex: { module: m, entryPoint: 'vsLines' },
        fragment: { module: m, entryPoint: 'fsLines', targets: [{ format: this.format, blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] },
        primitive: { topology: 'line-list' }, depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' }, multisample: ms }),
    };
    this.creature = null; this.props = {};
    this.rebuildBindGroup();
    this.size = [0, 0];
  }
  async checkShaders() {
    const info = await this.module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('Shader compilation failed:\n' + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
  }
  rebuildBindGroup() {
    const c = this.creature;
    this.bg0 = this.device.createBindGroup({ layout: this.bgl0, entries: [
      { binding: 0, resource: { buffer: this.frameBuf } },
      { binding: 1, resource: { buffer: c ? c.particleBuf : this.placeholder } },
      { binding: 2, resource: { buffer: c ? c.lagBuf : this.placeholder } },
      { binding: 3, resource: { buffer: c ? c.cellBuf : this.placeholder } },
      { binding: 4, resource: { buffer: this.creatureBuf } },
      { binding: 5, resource: { buffer: this.propsBuf } },
    ] });
  }
  resize(w, h) {
    if (w === this.size[0] && h === this.size[1]) return;
    this.size = [w, h];
    this.canvas.width = w; this.canvas.height = h;
    this.msaaTex?.destroy(); this.depthTex?.destroy();
    this.msaaTex = this.device.createTexture({ size: [w, h], format: this.format, sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = this.device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: this.sampleCount, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }
  /** Upload static creature geometry; dynamic buffers are written each frame. */
  setCreature(creature) {
    this.releaseCreature();
    if (!creature) { this.rebuildBindGroup(); return; }
    const d = this.device, U = GPUBufferUsage, A = creature.assets, lat = A.lattice;
    const mk = (data, usage) => {
      const b = d.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | U.COPY_DST });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    const c = {
      vbuf: mk(A.vbuf, U.VERTEX), combBuf: mk(creature.fur ? creature.fur.comb : new Float32Array(A.vertexCount * 3), U.VERTEX),
      ibuf: mk(A.indices, U.INDEX), indexCount: A.indices.length,
      particleBuf: d.createBuffer({ size: lat.n * 16, usage: U.STORAGE | U.COPY_DST }),
      lagBuf: d.createBuffer({ size: lat.n * 16, usage: U.STORAGE | U.COPY_DST }),
      cellBuf: mk(new Uint32Array(lat.cells), U.STORAGE),
      edgeBuf: mk(lat.edges, U.INDEX), edgeCount: lat.edges.length,
      pos32: new Float32Array(lat.n * 4), lag32: new Float32Array(lat.n * 4),
    };
    // Static parts of the creature uniform: decals, groups, spots.
    const f = new Float32Array(this.creatureData), u = new Uint32Array(this.creatureData);
    f.fill(0);
    f.set(A.spotData, 192);
    u.set(A.groupData, 228);
    f.set(A.decalData, 356);
    this.creature = c;
    // Prop meshes for this species' environment items.
    for (const [id, m] of Object.entries(A.items)) this.props[id] = { vbuf: mk(m.data, U.VERTEX), ibuf: mk(m.indices, U.INDEX), count: m.indices.length };
    this.rebuildBindGroup();
  }
  releaseCreature() {
    const c = this.creature;
    if (c) for (const k of ['vbuf', 'combBuf', 'ibuf', 'particleBuf', 'lagBuf', 'cellBuf', 'edgeBuf']) c[k].destroy();
    for (const p of Object.values(this.props)) { p.vbuf.destroy(); p.ibuf.destroy(); }
    this.creature = null; this.props = {};
  }
  render(S) {
    const d = this.device, c = this.creature, F = this.frameData;
    if (!this.msaaTex) return;
    // ---- per-frame uniforms
    F.set(S.viewProj, 0); F.set(S.lightViewProj, 16);
    F.set([...S.eye, S.time], 32); F.set([...S.keyDir, S.shadowStrength], 36); F.set([...S.keyColor, 0], 40);
    F.set([...S.fillDir, 0], 44); F.set([...S.fillColor, 0], 48); F.set([...S.skyColor, 0], 52); F.set([...S.groundColor, 0], 56);
    F.set([...S.bgColor, 1], 60); F.set(S.fur, 64); F.set(S.env, 68); F.set(S.bounds, 72); F.set([...S.floorColor, 0], 76);
    d.queue.writeBuffer(this.frameBuf, 0, F);
    const cr = S.creature;
    if (c && cr) {
      const x = cr.sim.x, n = cr.sim.n, lag = cr.lag;
      for (let i = 0; i < n; i++) {
        c.pos32[4 * i] = x[3 * i]; c.pos32[4 * i + 1] = x[3 * i + 1]; c.pos32[4 * i + 2] = x[3 * i + 2];
        c.lag32[4 * i] = lag[3 * i]; c.lag32[4 * i + 1] = lag[3 * i + 1]; c.lag32[4 * i + 2] = lag[3 * i + 2];
      }
      d.queue.writeBuffer(c.particleBuf, 0, c.pos32);
      d.queue.writeBuffer(c.lagBuf, 0, c.lag32);
      if (cr.fur && cr.fur.dirty) { d.queue.writeBuffer(c.combBuf, 0, cr.fur.comb); cr.fur.dirty = false; }
      const cf = new Float32Array(this.creatureData);
      cr.computeAttachments();
      cf.set(cr.attachMats, 0);
      cf.set(S.materials, 64);
      cf.set(cr.expr, 224);
      d.queue.writeBuffer(this.creatureBuf, 0, this.creatureData);
    }
    const items = S.items || [];
    items.slice(0, 8).forEach((it, i) => this.propsData.set(it.matrix, i * 16));
    if (items.length) d.queue.writeBuffer(this.propsBuf, 0, this.propsData);
    const enc = d.createCommandEncoder();
    // ---- shadow pass
    const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    sp.setBindGroup(0, this.bg0);
    if (c && cr) {
      sp.setPipeline(this.pipes.shadow);
      sp.setVertexBuffer(0, c.vbuf); sp.setVertexBuffer(1, c.combBuf); sp.setIndexBuffer(c.ibuf, 'uint32');
      sp.drawIndexed(c.indexCount, 1, 0, 0, 0);
    }
    if (items.length) {
      sp.setPipeline(this.pipes.propShadow);
      items.slice(0, 8).forEach((it, i) => { const m = this.props[it.type]; if (!m) return; sp.setVertexBuffer(0, m.vbuf); sp.setIndexBuffer(m.ibuf, 'uint32'); sp.drawIndexed(m.count, 1, 0, 0, i); });
    }
    sp.end();
    // ---- main pass
    const bg = S.bgClear;
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.msaaTex.createView(), resolveTarget: this.ctx.getCurrentTexture().createView(), clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    rp.setBindGroup(0, this.bg0); rp.setBindGroup(1, this.bg1);
    rp.setPipeline(this.pipes.floor); rp.draw(6);
    if (c && cr) {
      rp.setPipeline(this.pipes.creature);
      rp.setVertexBuffer(0, c.vbuf); rp.setVertexBuffer(1, c.combBuf); rp.setIndexBuffer(c.ibuf, 'uint32');
      rp.drawIndexed(c.indexCount, 1, 0, 0, 0);
    }
    if (items.length) {
      rp.setPipeline(this.pipes.prop);
      items.slice(0, 8).forEach((it, i) => { const m = this.props[it.type]; if (!m) return; rp.setVertexBuffer(0, m.vbuf); rp.setIndexBuffer(m.ibuf, 'uint32'); rp.drawIndexed(m.count, 1, 0, 0, i); });
    }
    if (c && cr && S.fur[1] > 0 && S.fur[0] > 0) {
      rp.setPipeline(this.pipes.fur);
      rp.setVertexBuffer(0, c.vbuf); rp.setVertexBuffer(1, c.combBuf); rp.setIndexBuffer(c.ibuf, 'uint32');
      rp.drawIndexed(c.indexCount, S.fur[1], 0, 0, 1);
    }
    if (c && cr && S.debugMesh) {
      rp.setPipeline(this.pipes.lines); rp.setIndexBuffer(c.edgeBuf, 'uint32'); rp.drawIndexed(c.edgeCount);
    }
    rp.end();
    d.queue.submit([enc.finish()]);
  }
  destroy() {
    this.releaseCreature();
    for (const t of [this.msaaTex, this.depthTex, this.shadowTex]) t?.destroy();
    for (const b of [this.frameBuf, this.creatureBuf, this.propsBuf, this.placeholder]) b.destroy();
    this.device.destroy();
  }
}

/* ---------- exports ---------------------------------------------------- */
return {
  math: { clamp, lerp, smoothstep, wrapAngle, damp, v3, mat4, makeRng, hexToLinear },
  SDF, surfaceNets, surfacePoint, buildLattice, SoftBody, Locomotion, BehaviorMachine, Environment, ENVIRONMENTS,
  MATERIAL_TYPES, buildMaterialTable, buildCreatureAssets, TOOLS, ENGINE_ACTIONS, Creature, Renderer, GPUUnavailable,
  registry, registerSpecies, validateSpecies,
};
})();
if (typeof module !== 'undefined') module.exports = Engine;
