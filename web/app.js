/* kinesplat studio UI — reconstruction queue, library and viewer wiring. */

import { SplatViewer, formatCount } from './viewer.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  config: null,
  presets: [],
  preset: 'balanced',
  files: [],
  jobs: [],
  demo: null,
  activeJobId: null,
  currentSceneId: null,
  pollTimer: null,
  toastTimer: null,
  viewer: null,
  busy: false,
  modelStatus: 'idle',
  device: null,
  gpu: false,
};

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|tiff?|heic|heif)$/i;
const VIDEO_RE = /\.(mp4|mov|avi|mkv|webm|m4v|wmv)$/i;
const MAX_FILES = 120;

// ------------------------------------------------------------------ helpers
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body && typeof options.body === 'string'
      ? { 'Content-Type': 'application/json' }
      : undefined,
    ...options,
  });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!response.ok) {
    throw new Error((data && data.error) || `Request failed (${response.status})`);
  }
  return data;
}

function toast(message, bad = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('bad', !!bad);
  el.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => el.classList.remove('show'), 4200);
}

function bytes(value) {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = value;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

function seconds(value) {
  if (!value || !Number.isFinite(value)) return null;
  if (value < 90) return `${Math.round(value)}s`;
  const m = value / 60;
  if (m < 60) return `${Math.round(m)} min`;
  return `${(m / 60).toFixed(1)} h`;
}

function titleCase(text) {
  return text
    .replace(/[_\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^\w/, (c) => c.toUpperCase());
}

function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ------------------------------------------------------------------ viewer
function initViewer() {
  try {
    state.viewer = new SplatViewer($('#gl'), $('#viewer'));
  } catch (error) {
    console.warn(error);
    $('#empty-state')?.classList.remove('is-hidden');
    toast('This browser does not support WebGL2. Try Chrome, Edge or Firefox.', true);
    $('#gl').style.display = 'none';
    return;
  }
  bindViewer();
}

function bindViewer() {
  const viewer = state.viewer;
  $('#btn-reset').addEventListener('click', () => viewer.resetView());
  $('#btn-auto').addEventListener('click', () => {
    const on = !viewer.getAutoRotate();
    viewer.setAutoRotate(on);
    $('#btn-auto').classList.toggle('is-active', on);
  });
  $('#btn-speed').addEventListener('click', () => {
    const on = !viewer.getAutoRotate();
    viewer.setAutoRotate(on);
    $('#btn-auto').classList.toggle('is-active', on);
    if (on) toast('Auto-orbit on');
  });
  $('#btn-orbit').addEventListener('click', () => {
    viewer.setAutoRotate(false);
    $('#btn-auto').classList.remove('is-active');
    toast('Drag to orbit · scroll to zoom');
  });
  $('#btn-cams').addEventListener('click', () => {
    const on = !$('#btn-cams').classList.contains('is-active');
    $('#btn-cams').classList.toggle('is-active', on);
    viewer.setCameraHelpers(on);
  });
  $('#btn-density').addEventListener('click', () => {
    const boost = viewer.cycleSplatScale();
    $('#btn-density').classList.toggle('is-active', Math.abs(boost - 1) > 0.01);
    toast(`Splat size ×${boost.toFixed(2).replace(/0$/, '')}`);
  });
  $('#btn-shot').addEventListener('click', () => saveScreenshot());
  $('#btn-full').addEventListener('click', () => toggleFullscreen());
}

async function openScene(item) {
  if (!state.viewer || !item?.scene_url) return;
  const viewer = state.viewer;
  setLoading(true, 'Reading scene…', 0.04);
  $('#empty-state').classList.add('is-hidden');
  state.currentSceneId = item.id;
  renderLibrary();
  try {
    const scene = await viewer.loadScene(item.scene_url, {
      onStage: (label, pct) => setLoading(true, label, pct),
    });
    updateStageHeader(scene, item);
    buildTimeline(scene.cameras || []);
    setLoading(false);
    $('#btn-auto').classList.remove('is-active');
    $('#btn-cams').classList.remove('is-active');
  } catch (error) {
    console.error(error);
    setLoading(false);
    $('#empty-state').classList.remove('is-hidden');
    toast(error.message || 'Could not load the scene', true);
  }
}

function setLoading(on, label = '', progress = 0) {
  const el = $('#viewer-loading');
  el.hidden = !on;
  if (on) {
    $('#loading-label').textContent = label;
    $('#loading-fill').style.width = `${Math.round(progress * 100)}%`;
  }
}

function updateStageHeader(scene, item) {
  const stats = scene.stats || item?.metrics || {};
  const isMesh = (item?.kind || stats.backend) === 'mesh' || scene.kind === 'mesh';
  $('#view-eyebrow').textContent = item?.id === 'demo' ? 'Bundled demo' : 'Reconstruction';
  $('#view-title').textContent = scene.name || item?.name || 'Splat scene';

  const chips = [];
  const count = stats.gaussians || item?.gaussians;
  const faces = stats.faces || item?.faces;
  if (isMesh) {
    if (faces) chips.push(`<span class="chip"><b>${formatCount(faces)}</b> faces</span>`);
    chips.push('<span class="chip accent">Single photo</span>');
  } else if (count) {
    chips.push(`<span class="chip"><b>${formatCount(count)}</b> splats</span>`);
  }
  if (!isMesh && Number.isFinite(Number(stats.psnr)) && Number(stats.psnr) > 0) {
    chips.push(`<span class="chip"><b>${Number(stats.psnr).toFixed(1)}</b> dB PSNR</span>`);
  }
  const cams = stats.cameras || item?.cameras || (scene.cameras || []).length;
  if (cams) chips.push(`<span class="chip"><b>${cams}</b> views</span>`);
  const secs = seconds(stats.total_seconds || stats.seconds || item?.seconds);
  if (secs) chips.push(`<span class="chip">${secs} build</span>`);
  const preset = item?.preset || stats.preset;
  if (preset && preset !== 'bundled') {
    chips.push(`<span class="chip accent">${escapeHtml(titleCase(preset))}</span>`);
  }
  $('#view-chips').innerHTML = chips.join('');
}

function buildTimeline(cameras) {
  const el = $('#timeline');
  if (!cameras || cameras.length < 2) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  const limited = cameras.slice(0, 64);
  el.innerHTML = limited.map((cam, i) =>
    `<button class="cam-chip" type="button" data-index="${i}" title="${escapeHtml(cam.name || 'View ' + (i + 1))}">${String(i + 1).padStart(2, '0')}</button>`
  ).join('');
  el.hidden = false;
  el.scrollLeft = 0;
  $$('.cam-chip', el).forEach((chip) => {
    chip.addEventListener('click', () => {
      const index = Number(chip.dataset.index);
      state.viewer.setCameraPose(index);
      $$('.cam-chip', el).forEach((c) => c.classList.toggle('is-active', c === chip));
      $('#btn-auto').classList.remove('is-active');
    });
  });
}

function toggleFullscreen() {
  const target = $('#viewer');
  if (!document.fullscreenElement) {
    target.requestFullscreen?.().catch(() => toast('Fullscreen is unavailable'));
  } else {
    document.exitFullscreen?.();
  }
}

function saveScreenshot() {
  if (!state.viewer?.ready) { toast('Load a scene first'); return; }
  const title = $('#view-title').textContent || 'kinesplat';
  try {
    const data = state.viewer.snapshot();
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const pad = Math.round(Math.max(18, img.width * 0.018));
      ctx.font = `${Math.round(pad * 0.9)}px "Segoe UI", system-ui, sans-serif`;
      ctx.fillStyle = 'rgba(0,0,0,0.42)';
      const text = `kinesplat  ·  ${title}`;
      const w = ctx.measureText(text).width + pad * 2;
      const h = pad * 2.4;
      const x = pad;
      const y = img.height - pad - h;
      const r = h / 2;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
      ctx.fill();
      ctx.fillStyle = '#eef3f4';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x + pad, y + h / 2 + 1);
      const a = document.createElement('a');
      a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'splatlab'}.png`;
      a.href = canvas.toDataURL('image/png');
      a.click();
    };
    img.src = data;
    toast('Screenshot saved');
  } catch (error) {
    toast('Screenshot failed', true);
  }
}

// ------------------------------------------------------------------ files
function fileName(file) {
  return file.webkitRelativePath || file.name;
}

function addFiles(list) {
  if (!list) return;
  const incoming = [...list].filter((f) => IMAGE_RE.test(f.name) || VIDEO_RE.test(f.name) || /^(image|video)\//.test(f.type));
  if (!incoming.length) {
    toast('Add photos or a video (JPG, PNG, WEBP, MP4, MOV)', true);
    return;
  }
  const existing = new Set(state.files.map((f) => `${f.name}:${f.size}`));
  for (const file of incoming) {
    const key = `${file.name}:${file.size}`;
    if (existing.has(key)) continue;
    if (state.files.length >= MAX_FILES) {
      toast(`At most ${MAX_FILES} files per scene`);
      break;
    }
    state.files.push(file);
    existing.add(key);
  }
  if (!$('#scene-name').value.trim()) {
    const base = titleCase(state.files[0].name.replace(/\.[^.]+$/, '').replace(/[\d_\-]+$/g, '').trim());
    if (base && base.length > 2) $('#scene-name').value = base;
  }
  renderFiles();
  $('#panel-create').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderFiles() {
  const wrap = $('#filelist');
  const files = state.files;
  wrap.hidden = files.length === 0;
  updateBackendNote();
  if (!files.length) { wrap.innerHTML = ''; return; }
  const total = files.reduce((s, f) => s + f.size, 0);
  const shown = files.slice(0, 10).map((file, i) => `
    <div class="file-chip" title="${escapeHtml(fileName(file))}">
      <span>${escapeHtml(file.name)}</span>
      <em>${bytes(file.size)}</em>
      <button type="button" data-index="${i}" aria-label="Remove file">×</button>
    </div>`).join('');
  const more = files.length > 10
    ? `<span class="file-more">+${files.length - 10} more · ${bytes(total)}</span>` : '';
  wrap.innerHTML = shown + more + (files.length > 1
    ? `<span class="file-more">${files.length} files · ${bytes(total)}</span>` : '');
  $$('button[data-index]', wrap).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.files.splice(Number(btn.dataset.index), 1);
      renderFiles();
    });
  });
  updateStartButton();
}

function updateBackendNote() {
  const note = $('#backend-note');
  if (!note) return;
  const count = state.files.length;
  const mesh = state.meshBackend;
  if (count === 1) {
    note.hidden = false;
    note.textContent = mesh?.available
      ? 'One photo → single-image mesh (TripoSR). Add two or more photos to use Gaussian splatting instead.'
      : 'One photo → single-image mesh. Add two or more photos to use Gaussian splatting instead.';
  } else if (count > 1 && mesh && mesh.available === false) {
    note.hidden = false;
    note.textContent = 'Multi-view reconstruction needs the Python packages listed in requirements.txt.';
  } else {
    note.hidden = true;
    note.textContent = '';
  }
}

function updateStartButton() {
  const btn = $('#start-btn');
  btn.disabled = state.busy || state.files.length === 0;
  const label = $('.primary-label', btn);
  if (state.busy) label.textContent = 'Starting…';
  else if (state.files.length) label.textContent = `Reconstruct ${state.files.length} ${state.files.length === 1 ? 'file' : 'files'}`;
  else label.textContent = 'Reconstruct in 3D';
}

// ------------------------------------------------------------------ presets
const PRESET_TIMES = {
  cpu: { draft: '≈ 40 s', balanced: '≈ 12 min', quality: '≈ 40 min' },
  gpu: { draft: '≈ 15 s', balanced: '≈ 2 min', quality: '≈ 6 min' },
};

function applyDevice(config) {
  const info = config.device_info || { type: 'cpu', name: 'CPU', vram_gb: null };
  const gpu = info.type === 'cuda' || info.type === 'mps';
  state.device = info;
  state.gpu = gpu;

  const label = $('#device-pill-text');
  if (label) {
    if (info.type === 'cuda') label.textContent = `GPU · ${info.vram_gb} GB`;
    else if (info.type === 'mps') label.textContent = 'Apple GPU';
    else if (info.type === 'unavailable') label.textContent = 'GPU unavailable';
    else label.textContent = 'CPU';
  }
  const pill = $('#device-pill');
  if (pill) {
    pill.title = info.type === 'cuda'
      ? `${info.name} — reconstruction runs locally on your GPU`
      : info.type === 'mps'
        ? 'Reconstruction runs locally on the Apple GPU'
        : info.type === 'unavailable'
          ? (config.device_error || 'CUDA is unavailable in this Python environment')
          : 'Reconstruction runs locally on your CPU';
  }

  const stat = $('#stat-device');
  if (stat) stat.textContent = info.type === 'cuda' ? `${info.vram_gb} GB`
    : info.type === 'unavailable' ? 'Unavailable' : 'CPU';
  const note = $('#stat-device-note');
  if (note) note.textContent = info.type === 'cuda' ? 'GPU VRAM'
    : info.type === 'unavailable' ? 'fix CUDA before reconstructing' : 'no GPU needed';
}

function renderPresets() {
  const grid = $('#preset-grid');
  const times = state.gpu ? PRESET_TIMES.gpu : PRESET_TIMES.cpu;
  grid.innerHTML = (state.presets || []).map((p) => `
    <button class="preset ${p.key === state.preset ? 'is-active' : ''}" type="button" role="radio"
      aria-checked="${p.key === state.preset}" data-key="${p.key}" title="${escapeHtml(p.description)}">
      <strong>${escapeHtml(p.label)}</strong>
      <small>${times[p.key] || `${p.iterations} its`}</small>
    </button>`).join('');
  $$('.preset', grid).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.preset = btn.dataset.key;
      $$('.preset', grid).forEach((b) => {
        const active = b === btn;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-checked', String(active));
      });
      const preset = state.presets.find((p) => p.key === state.preset);
      if (preset) $('#preset-time').textContent = `${formatCount(preset.max_gaussians)} splats · ${preset.iterations} steps`;
    });
  });
  const preset = state.presets.find((p) => p.key === state.preset);
  if (preset) $('#preset-time').textContent = `${formatCount(preset.max_gaussians)} splats · ${preset.iterations} steps`;
}

// ------------------------------------------------------------------- upload
function uploadFile(jobId, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/jobs/${jobId}/upload`);
    xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else {
        let message = `Upload failed (${xhr.status})`;
        try { message = JSON.parse(xhr.responseText).error || message; } catch { /* ignore */ }
        reject(new Error(message));
      }
    };
    xhr.onerror = () => reject(new Error('Upload failed — is the local server running?'));
    xhr.send(file);
  });
}

async function startReconstruction() {
  if (state.busy || !state.files.length) return;
  state.busy = true;
  state.completedJobId = null;
  updateStartButton();
  $('#jobcard').hidden = false;
  setJobCard({
    name: $('#scene-name').value.trim() || 'Untitled scene',
    status: 'uploading', message: 'Creating job…', progress: 0,
  });
  try {
    const name = $('#scene-name').value.trim() || 'Untitled scene';
    const job = await api('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ name, preset: state.preset }),
    });
    state.activeJobId = job.id;
    setJobCard({ ...job, status: 'uploading', message: 'Uploading files…' });

    const files = state.files.slice();
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      $('#job-status-label').textContent = 'Uploading';
      $('#job-message').textContent = `${i + 1}/${files.length} · ${file.name}`;
      $('#job-progress').style.width = `${Math.round(((i + 0.15) / files.length) * 4)}%`;
      await uploadFile(job.id, file, (p) => {
        $('#job-progress').style.width = `${Math.round(((i + p) / files.length) * 4)}%`;
      });
    }

    state.files = [];
    renderFiles();
    $('#scene-name').value = '';
    await api(`/api/jobs/${job.id}/start`, { method: 'POST' });
    $('#job-status-label').textContent = 'Queued';
    $('#job-progress').style.width = '1%';
    pollJob(job.id);
    refreshLibrary();
  } catch (error) {
    console.error(error);
    toast(error.message || 'Could not start reconstruction', true);
    setJobCard({ name: 'Reconstruction', status: 'failed', message: error.message, progress: 0 });
  } finally {
    state.busy = false;
    updateStartButton();
  }
}

// ------------------------------------------------------------------- job UI
const STAGES = ['frames', 'poses', 'depth', 'train', 'export'];

function setJobCard(job) {
  const card = $('#jobcard');
  card.hidden = false;
  $('#job-title').textContent = job.name || 'Reconstruction';
  const label = job.status === 'running' ? 'Working'
    : job.status === 'queued' ? 'Queued'
    : job.status === 'uploading' ? 'Uploading'
    : job.status === 'done' ? 'Complete'
    : job.status === 'failed' ? 'Failed'
    : job.status === 'cancelled' ? 'Cancelled' : 'Ready';
  $('#job-status-label').textContent = label;
  const dot = $('#job-dot');
  dot.className = 'status-dot' + (
    job.status === 'running' || job.status === 'queued' || job.status === 'uploading' ? ' pulse'
    : job.status === 'failed' ? ' bad' : job.status === 'cancelled' ? ' warn' : '');
  const pct = Math.max(0, Math.min(1, Number(job.progress) || 0));
  $('#job-progress').style.width = `${Math.max(2, Math.round(pct * 100))}%`;
  $('#job-percent').textContent = `${Math.round(pct * 100)}%`;
  $('#job-message').textContent = job.message || '';
  $('#job-time').textContent = job.seconds ? seconds(job.seconds) : '';

  const stage = job.stage || '';
  const activeIdx = STAGES.indexOf(stage);
  $$('#job-steps span').forEach((el, i) => {
    el.classList.toggle('is-active', el.dataset.stage === stage || (job.status === 'running' && i === activeIdx));
    el.classList.toggle('is-done', activeIdx >= 0 ? i < activeIdx : job.status === 'done');
  });

  const cancelBtn = $('#cancel-job-btn');
  cancelBtn.hidden = !(job.status === 'running' || job.status === 'queued');
  const openBtn = $('#open-job-btn');
  openBtn.hidden = job.status !== 'done';
  const preview = $('#job-preview');
  if (job.preview_url) {
    if (preview.dataset.src !== job.preview_url) {
      preview.dataset.src = job.preview_url;
      preview.src = job.preview_url;
    }
    preview.hidden = false;
  } else {
    preview.hidden = true;
  }
  if (job.status === 'failed' && job.error) {
    $('#job-message').textContent = job.error.split('\n')[0].slice(0, 180);
  }
}

function setJobLog(lines, scroll = true) {
  const log = $('#job-log');
  if (!log) return;
  log.textContent = (lines || []).join('\n');
  if (scroll) log.scrollTop = log.scrollHeight;
}

function pollJob(jobId) {
  clearInterval(state.pollTimer);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setJobCard(job);
      setJobLog(job.log || []);
      if (job.status === 'done') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.activeJobId = null;
        state.completedJobId = job.id;
        $('#job-status-label').textContent = 'Complete';
        await refreshLibrary();
        const item = state.jobs.find((j) => j.id === job.id) || jobToLibrary(job);
        if (item) openScene(item);
        const readyKind = job.kind || job.result?.kind;
        toast(readyKind === 'mesh'
          ? `Scene ready — mesh with ${formatCount(job.faces || job.metrics?.faces || 0)} faces`
          : `Scene ready — ${formatCount(job.gaussians)} splats`);
      } else if (job.status === 'failed' || job.status === 'cancelled') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.activeJobId = null;
        refreshLibrary();
        if (job.status === 'failed') toast(job.error || 'Reconstruction failed', true);
      }
    } catch (error) {
      console.warn(error);
    }
  };
  tick();
  state.pollTimer = setInterval(tick, 1300);
}

function jobToLibrary(job) {
  if (!job?.scene_url) return null;
  return {
    ...job,
    id: job.id,
    metrics: job.metrics || {},
    poster_url: job.poster_url,
    preview_url: job.preview_url,
    scene_url: job.scene_url,
  };
}

// ------------------------------------------------------------------ library
async function refreshLibrary() {
  try {
    const data = await api('/api/jobs');
    state.jobs = data.jobs || [];
    state.demo = data.demo || null;
    renderLibrary();
  } catch (error) {
    console.warn('Library refresh failed', error);
  }
}

function renderLibrary() {
  const list = $('#library-list');
  const items = [];
  if (state.demo) items.push({ ...state.demo, isDemo: true });
  items.push(...state.jobs.map((j) => ({ ...j, isDemo: false })));

  $('#library-count').textContent = String(items.length);

  if (!items.length) {
    list.innerHTML = '<div class="library-empty">Nothing here yet. Reconstruct your first scene from the Reconstruct tab.</div>';
    return;
  }

  list.innerHTML = items.map((item) => {
    const metrics = item.metrics || {};
    const isMesh = (item.kind || metrics.backend) === 'mesh';
    const count = item.gaussians || metrics.gaussians;
    const faces = item.faces || metrics.faces;
    const cams = item.cameras || metrics.cameras;
    const psnr = Number(metrics.psnr);
    const image = item.preview_url || item.poster_url;
    const statusBadge = item.status && item.status !== 'done' ? `<span class="scene-badge">${escapeHtml(item.status)}</span>` : '';
    const demoBadge = item.isDemo ? '<span class="scene-badge demo">Demo</span>' : statusBadge;
    const disabled = item.status !== 'done' && !item.isDemo;
    return `
      <article class="scene-card ${state.currentSceneId === item.id ? 'is-current' : ''}" data-id="${escapeHtml(item.id)}">
        <div class="scene-thumb" style="${image ? `background-image:url('${escapeHtml(image)}')` : ''}">
          ${image ? `<img src="${escapeHtml(image)}" alt="${escapeHtml(item.name || 'Scene')} preview" loading="lazy" />` : ''}
          ${demoBadge}
          <div class="scene-thumb-meta">
            <strong>${escapeHtml(item.name || 'Untitled scene')}</strong>
            <em>${item.isDemo ? 'bundled' : escapeHtml(titleCase(item.preset || ''))}</em>
          </div>
        </div>
        <div class="scene-meta">
          <div class="mstats">
            ${isMesh && faces ? `<span><b>${formatCount(faces)}</b> faces</span>` : ''}
            ${!isMesh && count ? `<span><b>${formatCount(count)}</b> splats</span>` : ''}
            ${cams ? `<span><b>${cams}</b> views</span>` : ''}
            ${!isMesh && Number.isFinite(psnr) && psnr > 0 ? `<span><b>${psnr.toFixed(1)}</b> dB</span>` : ''}
          </div>
          <div class="scene-actions">
            ${item.isDemo ? '' : `
              <button class="scene-delete" type="button" data-delete="${escapeHtml(item.id)}" title="Delete scene">
                <svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V5h4v2m-7 0 1 12h8l1-12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </button>`}
            <button class="scene-open" type="button" data-open="${escapeHtml(item.id)}" ${disabled ? 'disabled' : ''}>
              ${state.currentSceneId === item.id ? 'Viewing' : 'Open'}
            </button>
          </div>
        </div>
      </article>`;
  }).join('');

  $$('[data-open]', list).forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.open;
      const item = id === 'demo' ? state.demo : state.jobs.find((j) => j.id === id);
      if (item) openScene(item);
    });
  });
  $$('[data-delete]', list).forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.delete;
      const item = state.jobs.find((j) => j.id === id);
      if (!item) return;
      showConfirm(`Delete “${item.name || 'this scene'}”?`, async () => {
        try {
          await api(`/api/jobs/${id}`, { method: 'DELETE' });
          if (state.currentSceneId === id) {
            state.currentSceneId = null;
            $('#empty-state').classList.remove('is-hidden');
            $('#gl').classList.remove('is-ready');
          }
          await refreshLibrary();
          toast('Scene deleted');
        } catch (error) {
          toast(error.message || 'Could not delete scene', true);
        }
      });
    });
  });
}

// ------------------------------------------------------------------- modals
let confirmAction = null;
function showConfirm(title, action) {
  $('#confirm-title').textContent = title;
  confirmAction = action;
  $('#confirm-modal').hidden = false;
}
function closeModals() {
  $('#help-modal').hidden = true;
  $('#confirm-modal').hidden = true;
  confirmAction = null;
}

// -------------------------------------------------------------------- boot
function bindUI() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.toggle('is-active', t === tab));
      $$('.panel-view').forEach((p) => p.classList.toggle('is-active', p.id === `panel-${tab.dataset.tab}`));
      if (tab.dataset.tab === 'library') refreshLibrary();
    });
  });

  const dropzone = $('#dropzone');
  const input = $('#file-input');
  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });

  let dragDepth = 0;
  const overlay = $('#drop-overlay');
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    overlay.hidden = false;
  });
  window.addEventListener('dragover', (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener('dragleave', (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.hidden = true;
  });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    overlay.hidden = true;
    addFiles(e.dataTransfer.files);
    $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === 'create'));
    $$('.panel-view').forEach((p) => p.classList.toggle('is-active', p.id === 'panel-create'));
  });

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('is-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('is-over'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('is-over');
    addFiles(e.dataTransfer.files);
  });

  $('#start-btn').addEventListener('click', startReconstruction);
  $('#cancel-job-btn').addEventListener('click', async () => {
    if (!state.activeJobId) return;
    try {
      await api(`/api/jobs/${state.activeJobId}/cancel`, { method: 'POST' });
      toast('Cancelling…');
    } catch (error) {
      toast(error.message || 'Could not cancel', true);
    }
  });
  $('#open-job-btn').addEventListener('click', () => {
    const id = state.activeJobId || state.completedJobId;
    const item = state.jobs.find((j) => j.id === id);
    if (item) openScene(item);
    else toast('The scene is still being written');
  });
  $('#log-toggle').addEventListener('click', () => {
    const log = $('#job-log');
    log.hidden = !log.hidden;
  });

  $('#library-refresh').addEventListener('click', async () => {
    await refreshLibrary();
    toast('Library refreshed');
  });

  $('#btn-demo').addEventListener('click', () => {
    if (state.demo) openScene(state.demo);
    else toast('The bundled demo is not available', true);
  });
  $('#btn-upload-empty').addEventListener('click', () => {
    $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === 'create'));
    $$('.panel-view').forEach((p) => p.classList.toggle('is-active', p.id === 'panel-create'));
    $('#file-input').click();
  });

  $('#help-btn').addEventListener('click', () => { $('#help-modal').hidden = false; });
  $('#help-close').addEventListener('click', closeModals);
  $('#help-modal').addEventListener('click', (e) => {
    if (e.target.id === 'help-modal') closeModals();
  });
  $('#confirm-cancel').addEventListener('click', closeModals);
  $('#confirm-ok').addEventListener('click', async () => {
    const action = confirmAction;
    closeModals();
    if (action) await action();
  });
  $('#confirm-modal').addEventListener('click', (e) => {
    if (e.target.id === 'confirm-modal') closeModals();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModals();
    if (e.key === 'f' && document.activeElement?.tagName !== 'INPUT') toggleFullscreen();
  });

  document.addEventListener('fullscreenchange', () => {
    $('#btn-full').classList.toggle('is-active', !!document.fullscreenElement);
    state.viewer?.resize();
  });

  const openBtn = $('#open-job-btn');
  if (openBtn) openBtn.hidden = true;
}

async function boot() {
  initViewer();
  bindUI();
  try {
    const config = await api('/api/config');
    state.config = config;
    state.presets = config.presets || [];
    state.demo = config.demo || null;
    state.meshBackend = config.mesh_backend || { available: false };
    applyDevice(config);
    renderPresets();
    updateStartButton();
    refreshLibrary();
    if (state.demo) {
      setTimeout(() => openScene(state.demo), 240);
    }
  } catch (error) {
    console.error(error);
    toast('Could not reach the kinesplat server.', true);
    $('#empty-state').classList.remove('is-hidden');
  }
}

boot();
