/* kinesplat viewer — a custom WebGL2 instanced Gaussian-splat renderer.
 *
 * The scene format is the compact 32-byte-per-splat layout:
 *   3 x float32 position | 3 x float32 scale | 4 x uint8 rgba | 4 x uint8 quat
 *
 * Splats are uploaded once into floating-point textures. Each frame the CPU
 * builds a far-to-near order (16-bit counting sort) and writes only the order
 * texture, so interaction stays smooth even with a few hundred thousand
 * splats.
 */

import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const SPLAT_BYTES = 32;

function formatCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
  return String(n);
}

const VERT = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 aCorner;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform int uTexW;
uniform int uOrderW;
uniform float uScaleBoost;

uniform highp sampler2D uOrderTex;
uniform highp sampler2D uPosTex;
uniform highp sampler2D uScaleTex;
uniform highp sampler2D uQuatTex;
uniform highp sampler2D uColorTex;

out vec2 vUv;
out vec3 vColor;
out float vOpacity;

mat3 quatToMat(vec4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;
  return mat3(
    1.0 - (yy + zz), xy + wz, xz - wy,
    xy - wz, 1.0 - (xx + zz), yz + wx,
    xz + wy, yz - wx, 1.0 - (xx + yy)
  );
}

void main() {
  ivec2 ocoord = ivec2(gl_InstanceID % uOrderW, gl_InstanceID / uOrderW);
  int idx = int(texelFetch(uOrderTex, ocoord, 0).r + 0.5);
  ivec2 dc = ivec2(idx % uTexW, idx / uTexW);

  vec4 p = texelFetch(uPosTex, dc, 0);
  vec4 sc = texelFetch(uScaleTex, dc, 0);
  vec4 q = texelFetch(uQuatTex, dc, 0);
  vec4 col = texelFetch(uColorTex, dc, 0);

  vec4 vc = uView * vec4(p.xyz, 1.0);
  float z = -vc.z;
  if (z <= 0.015) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(2.0);
    vColor = vec3(0.0);
    vOpacity = 0.0;
    return;
  }

  vec4 clip = uProj * vc;
  float fx = 0.5 * uProj[0][0] * uViewport.x;
  float fy = 0.5 * uProj[1][1] * uViewport.y;

  vec4 qn = q / max(length(q), 1e-6);
  vec3 s = max(sc.xyz * uScaleBoost, vec3(0.00002));
  mat3 Rq = quatToMat(qn);
  mat3 M = Rq * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 Sigma = M * transpose(M);

  // View (three.js, y up) -> OpenCV camera axes (y down, z forward).
  mat3 Rcv = mat3(uView);
  Rcv[0].y = -Rcv[0].y; Rcv[0].z = -Rcv[0].z;
  Rcv[1].y = -Rcv[1].y; Rcv[1].z = -Rcv[1].z;
  Rcv[2].y = -Rcv[2].y; Rcv[2].z = -Rcv[2].z;
  mat3 Sc = Rcv * Sigma * transpose(Rcv);

  float x = vc.x;
  float y = -vc.y;
  mat3 J = mat3(
    fx / z, 0.0, -fx * x / (z * z),
    0.0, fy / z, -fy * y / (z * z),
    0.0, 0.0, 0.0
  );
  mat3 cov = J * Sc * transpose(J);

  float c0 = cov[0][0] + 0.22;
  float c1 = cov[0][1];
  float c2 = cov[1][1] + 0.22;
  float mid = 0.5 * (c0 + c2);
  float det = max(c0 * c2 - c1 * c1, 0.0);
  float disc = sqrt(max(mid * mid - det, 0.0));
  float l1 = max(mid + disc, 1e-6);
  float l2 = max(mid - disc, 0.0);
  vec2 e1 = abs(c1) < 1e-6 ? vec2(1.0, 0.0) : normalize(vec2(c1, l1 - c0));
  vec2 e2 = vec2(-e1.y, e1.x);
  float r1 = min(3.0 * sqrt(l1), 512.0);
  float r2 = min(3.0 * sqrt(l2), 512.0);
  vec2 off = e1 * (aCorner.x * r1) + e2 * (aCorner.y * r2);

  clip.xy += vec2(off.x * 2.0 / uViewport.x, -off.y * 2.0 / uViewport.y) * clip.w;
  gl_Position = clip;

  vUv = aCorner;
  vColor = col.rgb;
  vOpacity = sc.w;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
in vec3 vColor;
in float vOpacity;
out vec4 outColor;
void main() {
  float r2 = dot(vUv, vUv);
  if (r2 > 1.0) discard;
  float a = exp(-4.5 * r2) * vOpacity;
  if (a < 0.0022) discard;
  outColor = vec4(vColor * a, a);
}`;

const LINE_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uViewProj;
out vec4 vColor;
void main() {
  gl_Position = uViewProj * vec4(aPos, 1.0);
  vColor = aColor;
}`;

const LINE_FRAG = `#version 300 es
precision highp float;
in vec4 vColor;
out vec4 outColor;
void main() { outColor = vec4(vColor.rgb * vColor.a, vColor.a); }`;

/* Mesh pass: single-image TripoSR reconstructions arrive as a GLB with vertex
 * colours.  Positions are already in world space (identity node transform), so
 * the shader can light with world-space directions and no normal matrix. */
const MESH_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;
void main() {
  vNormal = aNormal;
  vColor = aColor;
  vWorld = aPosition;
  gl_Position = uProj * uView * vec4(aPosition, 1.0);
}`;

const MESH_FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
uniform vec3 uEye;
uniform float uExposure;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;             // open surfaces: light both sides
  vec3 keyDir = normalize(vec3(0.42, 0.78, 0.46));
  vec3 fillDir = normalize(vec3(-0.65, 0.2, -0.35));
  vec3 rimDir = normalize(vec3(0.0, -0.35, -0.9));
  float key = max(dot(n, keyDir), 0.0);
  float fill = max(dot(n, fillDir), 0.0);
  float rim = max(dot(n, rimDir), 0.0);
  float fres = pow(1.0 - max(dot(n, normalize(uEye - vWorld)), 0.0), 3.0);
  vec3 lit = vColor * (0.40 + 0.74 * key + 0.24 * fill)
           + vec3(0.20) * (rim * 0.35 + fres * 0.45);
  outColor = vec4(clamp(lit * uExposure, 0.0, 1.0), 1.0);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Shader compile failed: ' + log);
  }
  return shader;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('Program link failed: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

function makeTexture(gl, width, height, internal, format, data) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, gl.FLOAT, data);
  return tex;
}

async function fetchBuffer(url, onProgress) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Could not load ${url.split('/').pop()} (${response.status})`);
  const total = Number(response.headers.get('Content-Length') || 0);
  if (!response.body || !total) {
    const buf = await response.arrayBuffer();
    onProgress?.(1);
    return buf;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(Math.min(1, received / total));
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

export class SplatViewer {
  constructor(canvas, container) {
    this.canvas = canvas;
    this.container = container;
    this.gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: true,
      stencil: false,
      premultipliedAlpha: true,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('WebGL2 is required for the splat viewer.');

    this.renderer = null;
    this.scene = null;
    this.count = 0;
    this.mesh = null;
    this.cameras = [];
    this.texW = 1;
    this.texH = 1;
    this.orderW = 1;
    this.orderH = 1;
    this.scaleBoost = 1;
    this.showCameras = false;
    this.ready = false;
    this.dirty = true;
    this.lastTime = 0;
    this.onStatus = null;
    this.onFrame = null;
    this.activeCamera = -1;

    this._initGL();
    this._initThree();
    this._initLines();
    this._initMesh();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this._loop = this._loop.bind(this);
    this.frameHandle = requestAnimationFrame(this._loop);
  }

  _initGL() {
    const gl = this.gl;
    this.prog = program(gl, VERT, FRAG);
    this.uniforms = {
      view: gl.getUniformLocation(this.prog, 'uView'),
      proj: gl.getUniformLocation(this.prog, 'uProj'),
      viewport: gl.getUniformLocation(this.prog, 'uViewport'),
      texW: gl.getUniformLocation(this.prog, 'uTexW'),
      orderW: gl.getUniformLocation(this.prog, 'uOrderW'),
      scaleBoost: gl.getUniformLocation(this.prog, 'uScaleBoost'),
      orderTex: gl.getUniformLocation(this.prog, 'uOrderTex'),
      posTex: gl.getUniformLocation(this.prog, 'uPosTex'),
      scaleTex: gl.getUniformLocation(this.prog, 'uScaleTex'),
      quatTex: gl.getUniformLocation(this.prog, 'uQuatTex'),
      colorTex: gl.getUniformLocation(this.prog, 'uColorTex'),
    };

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  }

  _initThree() {
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
    this.camera = camera;
    this.controls = new OrbitControls(camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.rotateSpeed = 0.72;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.8;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.02;
    this.controls.maxDistance = 120;
    this.controls.autoRotateSpeed = 0.85;
    this.controls.addEventListener('change', () => { this.dirty = true; });
  }

  _initLines() {
    const gl = this.gl;
    this.lineProg = program(gl, LINE_VERT, LINE_FRAG);
    this.lineUniforms = {
      viewProj: gl.getUniformLocation(this.lineProg, 'uViewProj'),
    };
    this.lineSets = [];
  }

  _initMesh() {
    const gl = this.gl;
    this.meshProg = program(gl, MESH_VERT, MESH_FRAG);
    this.meshUniforms = {
      view: gl.getUniformLocation(this.meshProg, 'uView'),
      proj: gl.getUniformLocation(this.meshProg, 'uProj'),
      eye: gl.getUniformLocation(this.meshProg, 'uEye'),
      exposure: gl.getUniformLocation(this.meshProg, 'uExposure'),
    };
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.pixelRatio = dpr;
    this.width = w;
    this.height = h;
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  // ------------------------------------------------------------------ model
  async loadScene(sceneUrl, { onProgress, onStage } = {}) {
    this.ready = false;
    this.canvas.classList.remove('is-ready');
    this.dirty = true;

    onStage?.('Reading scene…', 0.04);
    const response = await fetch(sceneUrl, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Scene not found (${response.status})`);
    const scene = await response.json();
    const base = new URL('.', new URL(sceneUrl, window.location.href));
    const splatUrl = scene.splat ? new URL(scene.splat, base).href : null;
    const meshUrl = scene.mesh ? new URL(scene.mesh, base).href : null;
    if (!splatUrl && !meshUrl) {
      throw new Error('This scene has neither splats nor a mesh.');
    }

    this.count = 0;
    this._disposeMesh();

    if (meshUrl) {
      onStage?.('Downloading mesh…', 0.12);
      const meshBuffer = await fetchBuffer(meshUrl, (p) => {
        onStage?.(`Downloading mesh… ${Math.round(p * 100)}%`, 0.12 + p * 0.55);
      });
      onStage?.('Preparing mesh…', 0.7);
      await new Promise((r) => setTimeout(r, 16));
      this._parseGLB(meshBuffer);
      this._uploadMesh();
    }

    if (splatUrl) {
      onStage?.('Downloading splats…', 0.12);
      const buffer = await fetchBuffer(splatUrl, (p) => {
        onStage?.(`Downloading splats… ${Math.round(p * 100)}%`, 0.12 + p * 0.5);
      });

      onStage?.('Preparing splats…', 0.66);
      // Yield so the progress label paints.  A timer rather than rAF: background
      // tabs pause animation frames, which would stall the load indefinitely.
      await new Promise((r) => setTimeout(r, 16));
      this._parseSplat(buffer);
      this._uploadModel();
    }

    onStage?.('Placing camera…', 0.9);
    this._setupScene(scene);
    this.ready = true;
    this.canvas.classList.add('is-ready');
    onStage?.('Ready', 1);
    return scene;
  }

  _parseSplat(buffer) {
    const bytes = buffer.byteLength;
    const n = Math.floor(bytes / SPLAT_BYTES);
    if (!n) throw new Error('The splat file is empty.');
    const f32 = new Float32Array(buffer);
    const u8 = new Uint8Array(buffer);

    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const pz = new Float32Array(n);
    const pos = new Float32Array(n * 4);
    const scale = new Float32Array(n * 4);
    const quat = new Float32Array(n * 4);
    const color = new Float32Array(n * 4);

    for (let i = 0; i < n; i++) {
      const f = i * 8;
      const x = f32[f], y = f32[f + 1], z = f32[f + 2];
      const sx = f32[f + 3], sy = f32[f + 4], sz = f32[f + 5];
      const b = i * SPLAT_BYTES;
      const o = i * 4;
      px[i] = x; py[i] = y; pz[i] = z;
      pos[o] = x; pos[o + 1] = y; pos[o + 2] = z; pos[o + 3] = 1;
      scale[o] = Math.max(sx, 1e-5);
      scale[o + 1] = Math.max(sy, 1e-5);
      scale[o + 2] = Math.max(sz, 1e-5);
      scale[o + 3] = u8[b + 27] / 255;
      let qw = (u8[b + 28] - 128) / 128;
      let qx = (u8[b + 29] - 128) / 128;
      let qy = (u8[b + 30] - 128) / 128;
      let qz = (u8[b + 31] - 128) / 128;
      const qn = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= qn; qx /= qn; qy /= qn; qz /= qn;
      quat[o] = qw; quat[o + 1] = qx; quat[o + 2] = qy; quat[o + 3] = qz;
      color[o] = u8[b + 24] / 255;
      color[o + 1] = u8[b + 25] / 255;
      color[o + 2] = u8[b + 26] / 255;
      color[o + 3] = 1;
    }

    this.count = n;
    this.px = px; this.py = py; this.pz = pz;
    this.posData = pos;
    this.scaleData = scale;
    this.quatData = quat;
    this.colorData = color;
  }

  _uploadModel() {
    const gl = this.gl;
    const n = this.count;
    const width = 1024;
    const height = Math.ceil(n / width);
    const pad = width * height;
    this.texW = width;
    this.texH = height;
    this.orderW = width;
    this.orderH = height;
    this.orderData = new Float32Array(pad);
    this.depth = new Float32Array(n);
    this.counts = new Uint32Array(65536);

    // Textures must be fully populated for their declared width x height.
    const padArray = (arr) => {
      if (arr.length === pad * 4) return arr;
      const out = new Float32Array(pad * 4);
      out.set(arr);
      return out;
    };
    this._disposeTextures();
    this.posTex = makeTexture(gl, width, height, gl.RGBA32F, gl.RGBA, padArray(this.posData));
    this.scaleTex = makeTexture(gl, width, height, gl.RGBA32F, gl.RGBA, padArray(this.scaleData));
    this.quatTex = makeTexture(gl, width, height, gl.RGBA32F, gl.RGBA, padArray(this.quatData));
    this.colorTex = makeTexture(gl, width, height, gl.RGBA32F, gl.RGBA, padArray(this.colorData));
    this.orderTex = makeTexture(gl, width, height, gl.R32F, gl.RED, this.orderData);
    this.dirty = true;
  }

  _disposeTextures() {
    const gl = this.gl;
    for (const key of ['posTex', 'scaleTex', 'quatTex', 'colorTex', 'orderTex']) {
      if (this[key]) gl.deleteTexture(this[key]);
      this[key] = null;
    }
  }

  // ------------------------------------------------------------------- mesh
  /* Minimal GLB reader: walks the JSON chunk for the first primitive and pulls
   * POSITION / NORMAL / COLOR_0 / indices out of the binary chunk. */
  static _readAccessor(gltf, bin, index) {
    const acc = gltf.accessors[index];
    if (!acc) throw new Error('Mesh attribute is missing from the file.');
    const view = gltf.bufferViews[acc.bufferView] || {};
    const componentSize = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[acc.componentType];
    if (!componentSize) throw new Error(`Unsupported component type ${acc.componentType}`);
    const itemSize = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[acc.type] || 1;
    const stride = view.byteStride || componentSize * itemSize;
    const start = (view.byteOffset || 0) + (acc.byteOffset || 0);
    const out = new Float32Array(acc.count * itemSize);
    const dv = new DataView(bin);
    const scale = acc.normalized
      ? ({ 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 }[acc.componentType] || 1)
      : 1;
    for (let i = 0; i < acc.count; i++) {
      const base = start + i * stride;
      for (let c = 0; c < itemSize; c++) {
        const o = base + c * componentSize;
        let v = 0;
        switch (acc.componentType) {
          case 5126: v = dv.getFloat32(o, true); break;
          case 5121: v = dv.getUint8(o); break;
          case 5120: v = dv.getInt8(o); break;
          case 5123: v = dv.getUint16(o, true); break;
          case 5122: v = dv.getInt16(o, true); break;
          case 5125: v = dv.getUint32(o, true); break;
          default: break;
        }
        out[i * itemSize + c] = v / scale;
      }
    }
    return { data: out, itemSize };
  }

  static _vertexNormals(positions, indices) {
    const normals = new Float32Array(positions.length);
    for (let i = 0; i + 2 < indices.length; i += 3) {
      const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
      const ux = positions[b] - positions[a];
      const uy = positions[b + 1] - positions[a + 1];
      const uz = positions[b + 2] - positions[a + 2];
      const vx = positions[c] - positions[a];
      const vy = positions[c + 1] - positions[a + 1];
      const vz = positions[c + 2] - positions[a + 2];
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
      normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
      normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
    }
    for (let i = 0; i < normals.length; i += 3) {
      const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
      normals[i] /= len;
      normals[i + 1] /= len;
      normals[i + 2] /= len;
    }
    return normals;
  }

  _parseGLB(buffer) {
    const dv = new DataView(buffer);
    if (buffer.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67) {
      throw new Error('The mesh file is not a GLB.');
    }
    if (dv.getUint32(4, true) !== 2) throw new Error('Only GLB version 2 is supported.');
    const end = Math.min(buffer.byteLength, dv.getUint32(8, true));
    let offset = 12;
    let gltf = null;
    let bin = null;
    while (offset + 8 <= end) {
      const length = dv.getUint32(offset, true);
      const type = dv.getUint32(offset + 4, true);
      const start = offset + 8;
      if (type === 0x4e4f534a && !gltf) {
        const text = new TextDecoder().decode(new Uint8Array(buffer, start, length));
        gltf = JSON.parse(text);
      } else if (type === 0x004e4942 && !bin) {
        bin = buffer.slice(start, start + length);
      }
      offset = start + length + ((4 - (length % 4)) % 4);
    }
    if (!gltf?.meshes?.length) throw new Error('The GLB has no meshes.');
    if (!bin) throw new Error('The GLB has no binary chunk.');

    const prim = gltf.meshes[0].primitives[0];
    const pos = SplatViewer._readAccessor(gltf, bin, prim.attributes.POSITION);
    const positions = pos.data;
    const vertexCount = positions.length / 3;
    let indices;
    if (prim.indices != null) {
      const raw = SplatViewer._readAccessor(gltf, bin, prim.indices).data;
      indices = Uint32Array.from(raw);
    } else {
      indices = new Uint32Array(vertexCount);
      for (let i = 0; i < vertexCount; i++) indices[i] = i;
    }

    let normals;
    if (prim.attributes.NORMAL != null) {
      normals = SplatViewer._readAccessor(gltf, bin, prim.attributes.NORMAL).data;
    } else {
      normals = SplatViewer._vertexNormals(positions, indices);
    }

    let colors = new Float32Array(vertexCount * 3).fill(0.78);
    if (prim.attributes.COLOR_0 != null) {
      const raw = SplatViewer._readAccessor(gltf, bin, prim.attributes.COLOR_0);
      for (let i = 0; i < vertexCount; i++) {
        const o = i * raw.itemSize;
        colors[i * 3] = raw.data[o];
        colors[i * 3 + 1] = raw.data[o + 1];
        colors[i * 3 + 2] = raw.data[o + 2];
      }
    }

    this.mesh = { positions, normals, colors, indices, vertexCount };
    this.meshCount = vertexCount;
    this.meshFaces = indices.length / 3;
  }

  _uploadMesh() {
    const gl = this.gl;
    if (!this.mesh) return;
    // Keep the parsed arrays across the dispose call: _disposeMesh frees the GPU
    // handles *and* the CPU copy, which we are about to upload.
    const mesh = this.mesh;
    const vertexCount = this.meshCount;
    const faces = this.meshFaces;
    this._disposeMesh();
    const { positions, normals, colors, indices } = mesh;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const attach = (data, loc, size) => {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return buffer;
    };
    const buffers = [
      attach(positions, 0, 3),
      attach(normals, 1, 3),
      attach(colors, 2, 3),
    ];
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.mesh = mesh;
    this.meshCount = vertexCount;
    this.meshFaces = faces;
    this.meshVao = vao;
    this.meshBuffers = { buffers, indexBuffer };
    this.dirty = true;
  }

  _disposeMesh() {
    const gl = this.gl;
    if (this.meshVao) gl.deleteVertexArray(this.meshVao);
    if (this.meshBuffers) {
      for (const buffer of this.meshBuffers.buffers) gl.deleteBuffer(buffer);
      gl.deleteBuffer(this.meshBuffers.indexBuffer);
    }
    this.meshVao = null;
    this.meshBuffers = null;
    this.mesh = null;
    this.meshCount = 0;
    this.meshFaces = 0;
  }

  _renderMesh() {
    const gl = this.gl;
    if (!this.meshVao) return;
    gl.useProgram(this.meshProg);
    gl.bindVertexArray(this.meshVao);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.uniformMatrix4fv(this.meshUniforms.view, false, this.camera.matrixWorldInverse.elements);
    gl.uniformMatrix4fv(this.meshUniforms.proj, false, this.camera.projectionMatrix.elements);
    gl.uniform3f(this.meshUniforms.eye,
      this.camera.position.x, this.camera.position.y, this.camera.position.z);
    gl.uniform1f(this.meshUniforms.exposure, 1.0);
    gl.drawElements(gl.TRIANGLES, this.mesh.indices.length, gl.UNSIGNED_INT, 0);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(null);
  }

  // ------------------------------------------------------------------ scene
  _setupScene(scene) {
    this.scene = scene;
    this.cameras = Array.isArray(scene.cameras) ? scene.cameras : [];
    const bounds = scene.bounds || {};
    const center = new THREE.Vector3(
      bounds.center?.[0] ?? 0, bounds.center?.[1] ?? 0, bounds.center?.[2] ?? 0);
    let radius = Number(bounds.radius) || 0;
    if (!(radius > 0.01) || !Number.isFinite(radius)) radius = 2;
    this.center = center;
    this.radius = radius;
    const bg = Array.isArray(scene.background) && scene.background.length === 3
      ? scene.background.map((v) => Math.max(0, Math.min(1, Number(v) || 0)))
      : [0.035, 0.05, 0.06];
    this.background = bg;

    this.camera.near = Math.max(0.005, radius * 0.004);
    this.camera.far = Math.max(50, radius * 60);
    this.camera.fov = 50;
    this.homeFov = 50;
    this.camera.updateProjectionMatrix();

    const first = this.cameras[0];
    let dir = new THREE.Vector3(0, 0.25, 1).normalize();
    if (first && first.position) {
      const p = new THREE.Vector3(...first.position);
      const d = p.clone().sub(center);
      if (d.lengthSq() > 1e-8) dir = d.normalize();
    }
    const distance = radius * 2.15;
    this.camera.position.copy(center).addScaledVector(dir, distance);
    this.controls.target.copy(center);
    this.controls.update();
    this.home = {
      position: this.camera.position.clone(),
      target: center.clone(),
    };
    this.controls.autoRotate = false;
    this.activeCamera = -1;
    this._buildCameraHelpers();
    this.dirty = true;
  }

  // Camera orientation helpers.  Exported scenes carry `target` and `up`
  // (world space, y up); older files may only carry the OpenCV camera-to-world
  // matrix `R`, whose columns are right / down / forward.
  static _cameraAxes(cam) {
    const pos = new THREE.Vector3(...(cam.position || [0, 0, 0]));
    const R = cam.R;
    let forward;
    if (Array.isArray(cam.target)) {
      forward = new THREE.Vector3(...cam.target).sub(pos);
    } else if (Array.isArray(R) && R.length === 3) {
      forward = new THREE.Vector3(R[0][2], R[1][2], R[2][2]);
    } else {
      return null;
    }
    if (forward.lengthSq() < 1e-12) return null;
    forward.normalize();
    let up;
    if (Array.isArray(cam.up)) up = new THREE.Vector3(...cam.up);
    else if (Array.isArray(R) && R.length === 3) up = new THREE.Vector3(-R[0][1], -R[1][1], -R[2][1]);
    else up = new THREE.Vector3(0, 1, 0);
    up.normalize();
    // right = forward x up keeps the frustum corners the right way round.
    const right = new THREE.Vector3().crossVectors(forward, up);
    if (right.lengthSq() < 1e-12) return null;
    return { pos, forward, up, right: right.normalize() };
  }

  _buildCameraHelpers() {
    const gl = this.gl;
    for (const set of this.lineSets) {
      gl.deleteBuffer(set.positions);
      gl.deleteBuffer(set.colors);
      gl.deleteVertexArray(set.vao);
    }
    this.lineSets = [];
    if (!this.cameras.length) return;

    const radius = this.radius || 2;
    const depth = Math.max(radius * 0.1, 0.04);
    const pathPos = [];
    const pathCol = [];
    const linePos = [];
    const lineCol = [];
    const lime = [0.72, 1.0, 0.24, 0.6];
    const cyan = [0.35, 0.88, 1.0, 0.36];

    const positions = this.cameras.map((c) => new THREE.Vector3(...(c.position || [0, 0, 0])));
    for (let i = 0; i < positions.length; i++) {
      if (i > 0) {
        const a = positions[i - 1], b = positions[i];
        pathPos.push(a.x, a.y, a.z, b.x, b.y, b.z);
        pathCol.push(...lime, ...lime);
      }
    }

    for (let i = 0; i < this.cameras.length; i++) {
      const cam = this.cameras[i];
      const axes = SplatViewer._cameraAxes(cam);
      if (!axes) continue;
      const { pos: apex, forward, up, right } = axes;
      const hw = (0.5 * cam.width / Math.max(cam.fx, 1e-6)) * depth;
      const hh = (0.5 * cam.height / Math.max(cam.fy, 1e-6)) * depth;
      const corners = [
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
      ].map(([x, y]) => apex.clone()
        .addScaledVector(right, x)
        .addScaledVector(up, y)
        .addScaledVector(forward, depth));
      for (let k = 0; k < 4; k++) {
        const a = corners[k];
        const b = corners[(k + 1) % 4];
        linePos.push(a.x, a.y, a.z, b.x, b.y, b.z);
        lineCol.push(...cyan, ...cyan);
        linePos.push(apex.x, apex.y, apex.z, a.x, a.y, a.z);
        lineCol.push(...cyan, ...cyan);
      }
    }

    const addSet = (pos, col, mode) => {
      if (!pos.length) return;
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const pb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      const cb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, cb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      this.lineSets.push({ vao, positions: pb, colors: cb, count: pos.length / 3, mode });
    };
    addSet(pathPos, pathCol, gl.LINES);
    addSet(linePos, lineCol, gl.LINES);
  }

  // ------------------------------------------------------------------ order
  _sortSplats() {
    const n = this.count;
    if (!n) return;
    const e = this.camera.matrixWorldInverse.elements;
    const px = this.px, py = this.py, pz = this.pz;
    const depth = this.depth;
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = -(e[2] * px[i] + e[6] * py[i] + e[10] * pz[i] + e[14]);
      depth[i] = d;
      if (d < min) min = d;
      if (d > max) max = d;
    }
    const span = Math.max(max - min, 1e-7);
    const counts = this.counts;
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      let k = (depth[i] - min) / span;
      k = k < 0 ? 0 : k > 1 ? 1 : k;
      counts[((1 - k) * 65535) | 0]++;
    }
    let acc = 0;
    for (let b = 0; b < 65536; b++) {
      const c = counts[b];
      counts[b] = acc;
      acc += c;
    }
    const order = this.orderData;
    for (let i = 0; i < n; i++) {
      let k = (depth[i] - min) / span;
      k = k < 0 ? 0 : k > 1 ? 1 : k;
      const b = ((1 - k) * 65535) | 0;
      order[counts[b]++] = i;
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.orderTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.orderW, this.orderH,
      gl.RED, gl.FLOAT, order);
  }

  // ------------------------------------------------------------------ frame
  _loop(time) {
    this.frameHandle = requestAnimationFrame(this._loop);
    if (!this.ready) return;
    if (this.flight) {
      this._updateFlight(time);
      this.dirty = true;
    } else {
      const changed = this.controls.update();
      if (changed) this.dirty = true;
    }
    if (!this.dirty) {
      return;
    }
    this.lastTime = time;
    this._render();
    this.dirty = false;
  }

  _render() {
    const gl = this.gl;
    gl.viewport(0, 0, this.width, this.height);
    const bg = this.background || [0, 0, 0];
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count && !this.meshVao) return;

    this.camera.updateMatrixWorld();
    this.camera.matrixWorldInverse.copy(this.camera.matrixWorld).invert();

    // Mesh first: it writes depth, and the splats then composite over it.
    if (this.meshVao) this._renderMesh();
    if (!this.count) return;

    this._sortSplats();

    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);

    gl.useProgram(this.prog);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.uniformMatrix4fv(this.uniforms.view, false, this.camera.matrixWorldInverse.elements);
    gl.uniformMatrix4fv(this.uniforms.proj, false, this.camera.projectionMatrix.elements);
    gl.uniform2f(this.uniforms.viewport, this.width, this.height);
    gl.uniform1i(this.uniforms.texW, this.texW);
    gl.uniform1i(this.uniforms.orderW, this.orderW);
    gl.uniform1f(this.uniforms.scaleBoost, this.scaleBoost);

    const bind = (tex, unit, loc) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(this.orderTex, 0, this.uniforms.orderTex);
    bind(this.posTex, 1, this.uniforms.posTex);
    bind(this.scaleTex, 2, this.uniforms.scaleTex);
    bind(this.quatTex, 3, this.uniforms.quatTex);
    bind(this.colorTex, 4, this.uniforms.colorTex);

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);

    if (this.showCameras && this.lineSets.length) {
      this._renderLines();
    }
    gl.bindVertexArray(null);
  }

  _renderLines() {
    const gl = this.gl;
    const viewProj = new THREE.Matrix4().multiplyMatrices(
      this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    gl.useProgram(this.lineProg);
    gl.uniformMatrix4fv(this.lineUniforms.viewProj, false, viewProj.elements);
    for (const set of this.lineSets) {
      gl.bindVertexArray(set.vao);
      gl.drawArrays(set.mode, 0, set.count);
    }
    gl.bindVertexArray(null);
  }

  _updateFlight(time) {
    const f = this.flight;
    const t = Math.min(1, (time - f.start) / f.duration);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    this.camera.position.lerpVectors(f.fromPos, f.toPos, e);
    this.controls.target.lerpVectors(f.fromTarget, f.toTarget, e);
    this.camera.lookAt(this.controls.target);
    if (t >= 1) {
      this.flight = null;
      this.controls.enabled = true;
    }
  }

  // ------------------------------------------------------------------ public
  setAutoRotate(on) {
    this.controls.autoRotate = !!on;
    if (on) this.flight = null;
    this.dirty = true;
  }

  getAutoRotate() {
    return this.controls.autoRotate;
  }

  setCameraHelpers(show) {
    this.showCameras = !!show;
    this.dirty = true;
  }

  cycleSplatScale() {
    const values = [1, 0.62, 1.45, 2.1];
    const idx = values.indexOf(this.scaleBoost);
    this.scaleBoost = values[(idx + 1) % values.length];
    this.dirty = true;
    return this.scaleBoost;
  }

  resetView() {
    if (!this.home) return;
    this._setFov(this.homeFov || 50);
    this._flyTo(this.home.position, this.home.target, 700);
    this.activeCamera = -1;
  }

  setCameraPose(index) {
    const cam = this.cameras[index];
    if (!cam) return;
    const axes = SplatViewer._cameraAxes(cam);
    if (!axes) return;
    // Matching the original lens lets the jump reproduce the captured frame.
    if (Number.isFinite(cam.fy) && cam.fy > 0 && cam.height > 0) {
      const vfov = 2 * Math.atan(0.5 * cam.height / cam.fy) * 180 / Math.PI;
      this._setFov(Math.max(12, Math.min(90, vfov)));
    }
    const target = Array.isArray(cam.target)
      ? new THREE.Vector3(...cam.target)
      : axes.pos.clone().addScaledVector(axes.forward, Math.max(0.5, this.radius * 0.5));
    this._flyTo(axes.pos, target, 720);
    this.activeCamera = index;
  }

  _setFov(deg) {
    if (!(deg > 0) || Math.abs(this.camera.fov - deg) < 1e-3) return;
    this.camera.fov = deg;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  _flyTo(position, target, duration) {
    this.controls.enabled = false;
    this.controls.autoRotate = false;
    this.flight = {
      fromPos: this.camera.position.clone(),
      toPos: position.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: target.clone(),
      start: performance.now(),
      duration,
    };
    this.dirty = true;
  }

  snapshot() {
    this._render();
    const gl = this.gl;
    const width = this.width;
    const height = this.height;
    // Read back explicitly: the drawing buffer is not preserved across frames,
    // so canvas.toDataURL() alone can capture an empty image.
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(width, height);
    const row = width * 4;
    for (let y = 0; y < height; y++) {
      const src = (height - 1 - y) * row;
      image.data.set(pixels.subarray(src, src + row), y * row);
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  }

  dispose() {
    cancelAnimationFrame(this.frameHandle);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this._disposeTextures();
    this._disposeMesh();
    const gl = this.gl;
    for (const set of this.lineSets) {
      gl.deleteBuffer(set.positions);
      gl.deleteBuffer(set.colors);
      gl.deleteVertexArray(set.vao);
    }
    gl.deleteBuffer(this.quadBuffer);
    gl.deleteProgram(this.prog);
    gl.deleteProgram(this.lineProg);
    gl.deleteProgram(this.meshProg);
  }
}

export { formatCount };
