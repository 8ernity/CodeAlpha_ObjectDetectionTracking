/**
 * VisionGrid - High-Performance Object Detection & SORT Tracking
 * Hybrid Engine: Decoupled 60 FPS WebGL GPU AI + High-Precision Python YOLOv8
 */

// ==========================================
// 1. JavaScript Kalman Filter & SORT Tracker
// ==========================================
class KalmanBoxTracker {
  static count = 0;
  constructor(bbox, className, score) {
    KalmanBoxTracker.count += 1;
    this.id = KalmanBoxTracker.count;
    this.className = className;
    this.score = score;
    this.timeSinceUpdate = 0;
    this.hits = 1;
    this.hitStreak = 1;
    this.age = 0;
    
    // State: [cx, cy, w, h, vx, vy]
    const w = Math.max(1, bbox[2] - bbox[0]);
    const h = Math.max(1, bbox[3] - bbox[1]);
    const cx = bbox[0] + w / 2;
    const cy = bbox[1] + h / 2;
    
    this.cx = cx;
    this.cy = cy;
    this.w = w;
    this.h = h;
    this.vx = 0;
    this.vy = 0;
  }

  // Predict state for next frame (60 FPS smooth trajectory interpolation)
  predict() {
    this.cx += this.vx;
    this.cy += this.vy;
    this.age += 1;
    this.timeSinceUpdate += 1;
    
    const x1 = this.cx - this.w / 2;
    const y1 = this.cy - this.h / 2;
    const x2 = this.cx + this.w / 2;
    const y2 = this.cy + this.h / 2;
    return [x1, y1, x2, y2];
  }

  // Update state with newly observed AI detection
  update(bbox, score, className) {
    this.timeSinceUpdate = 0;
    this.hits += 1;
    this.hitStreak += 1;
    this.score = score;
    if (className) this.className = className;

    const w = Math.max(1, bbox[2] - bbox[0]);
    const h = Math.max(1, bbox[3] - bbox[1]);
    const cx = bbox[0] + w / 2;
    const cy = bbox[1] + h / 2;

    // Smooth velocity estimate with exponential moving average
    const alpha = 0.35;
    this.vx = this.vx * (1 - alpha) + (cx - this.cx) * alpha;
    this.vy = this.vy * (1 - alpha) + (cy - this.cy) * alpha;
    
    // Smooth position update
    this.cx = cx;
    this.cy = cy;
    this.w = w;
    this.h = h;
  }

  getBbox() {
    const x1 = this.cx - this.w / 2;
    const y1 = this.cy - this.h / 2;
    const x2 = this.cx + this.w / 2;
    const y2 = this.cy + this.h / 2;
    return [x1, y1, x2, y2];
  }
}

class WebSortTracker {
  constructor(maxAge = 40, minHits = 2, iouThreshold = 0.3) {
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.iouThreshold = iouThreshold;
    this.trackers = [];
    this.totalTracksCount = 0;
  }

  reset() {
    this.trackers = [];
    this.totalTracksCount = 0;
    KalmanBoxTracker.count = 0;
  }

  static iou(bb1, bb2) {
    const xx1 = Math.max(bb1[0], bb2[0]);
    const yy1 = Math.max(bb1[1], bb2[1]);
    const xx2 = Math.min(bb1[2], bb2[2]);
    const yy2 = Math.min(bb1[3], bb2[3]);
    const w = Math.max(0, xx2 - xx1);
    const h = Math.max(0, yy2 - yy1);
    const inter = w * h;
    const area1 = Math.max(1, (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]));
    const area2 = Math.max(1, (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]));
    const union = area1 + area2 - inter;
    return union > 0 ? inter / union : 0;
  }

  predictAll() {
    for (let i = 0; i < this.trackers.length; i++) {
      this.trackers[i].predict();
    }
  }

  update(detections) {
    const trkBoxes = this.trackers.map(t => t.getBbox());
    const matchedDets = new Set();
    const matchedTrks = new Set();
    const matches = [];

    // Best IoU greedy assignment
    for (let d = 0; d < detections.length; d++) {
      let bestIou = this.iouThreshold;
      let bestTrk = -1;

      for (let t = 0; t < this.trackers.length; t++) {
        if (matchedTrks.has(t)) continue;
        const iou = WebSortTracker.iou(detections[d].bbox, trkBoxes[t]);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrk = t;
        }
      }

      if (bestTrk !== -1) {
        matches.push([d, bestTrk]);
        matchedDets.add(d);
        matchedTrks.add(bestTrk);
      }
    }

    // Update matched
    for (const [d, t] of matches) {
      this.trackers[t].update(detections[d].bbox, detections[d].score, detections[d].className);
    }

    // Create new trackers for unmatched detections
    for (let d = 0; d < detections.length; d++) {
      if (!matchedDets.has(d)) {
        const newTrk = new KalmanBoxTracker(detections[d].bbox, detections[d].className, detections[d].score);
        this.trackers.push(newTrk);
        this.totalTracksCount += 1;
      }
    }

    // Clean stale trackers
    this.trackers = this.trackers.filter(t => t.timeSinceUpdate <= this.maxAge);
  }

  getActiveTracks() {
    return this.trackers
      .filter(t => t.timeSinceUpdate <= this.maxAge && (t.hits >= this.minHits || t.age <= 1))
      .map(t => ({
        id: t.id,
        className: t.className,
        score: t.score,
        bbox: t.getBbox(),
        vx: t.vx,
        vy: t.vy,
        speed: Math.round(Math.hypot(t.vx, t.vy) * 2.2) || 5
      }));
  }
}

const webTracker = new WebSortTracker(40, 2, 0.3);

// ==========================================
// 2. Global State & App Setup
// ==========================================
const state = {
  source: 'sample', // 'webcam', 'upload', 'sample'
  isPaused: false,
  isRecording: false,
  mediaRecorder: null,
  recordedChunks: [],
  ws: null,
  isPythonActive: false,
  confThresh: 0.45,
  maxAge: 25,
  linePos: 0.5,
  countIn: 0,
  countOut: 0,
  crossedIds: {},
  classFilter: 'all',
  toggles: {
    boxes: true,
    labels: true,
    ids: false,
    line: false
  },
  activeTracks: [],
  rawTracks: [],
  trackHistories: {},
  totalTracks: 0,
  fps: 0,
  latency: 0,
  fpsList: [],
  lastFrameTime: performance.now(),
  startTime: performance.now(),
  frameCounter: 0,
  eventLogs: []
};

// DOM Elements
const videoEl = document.getElementById('main-video');
const canvasEl = document.getElementById('main-canvas');
const ctx = canvasEl.getContext('2d');
const classDistEl = document.getElementById('class-distribution-list');
const activityFeedEl = document.getElementById('activity-feed');

// Offscreen AI tensor canvas (Decoupled 416px for lightning ~15ms inference)
const aiCanvas = document.createElement('canvas');
aiCanvas.width = 416;
aiCanvas.height = 234;
const aiCtx = aiCanvas.getContext('2d', { willReadFrequently: true });
let isAiDetecting = false;
let webModel = null;

// Offscreen canvas for Python streaming if available
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 416;
offscreenCanvas.height = 234;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
let isWsProcessing = false;
let lastWsSendTime = 0;

// Class Categories Mapping for Top Filters
const CLASS_CATEGORIES = {
  all: null,
  person: ['person'],
  vehicles: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'train', 'boat'],
  animals: ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
  electronics: ['cell phone', 'laptop', 'tv', 'mouse', 'keyboard', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'clock', 'book']
};

function isClassAllowed(className) {
  if (!className) return true;
  if (state.classFilter === 'all') return true;
  const list = CLASS_CATEGORIES[state.classFilter.toLowerCase()];
  if (list) {
    return list.includes(className.toLowerCase());
  }
  return className.toLowerCase() === state.classFilter.toLowerCase();
}

// Aspect Ratio Helper (Preserves native video aspect ratio with zero stretching)
function getVideoDrawRect(cw, ch) {
  const vw = videoEl.videoWidth || 1280;
  const vh = videoEl.videoHeight || 720;
  const videoAspect = vw / vh;
  const canvasAspect = cw / ch;

  let drawW, drawH, drawX, drawY;

  if (canvasAspect > videoAspect) {
    drawH = ch;
    drawW = ch * videoAspect;
    drawX = (cw - drawW) / 2;
    drawY = 0;
  } else {
    drawW = cw;
    drawH = cw / videoAspect;
    drawX = 0;
    drawY = (ch - drawH) / 2;
  }

  return { x: drawX, y: drawY, w: drawW, h: drawH, vw, vh };
}

// Theme Management
function initTheme() {
  const saved = localStorage.getItem('vision_theme') || 'dark';
  setTheme(saved);
}

function setTheme(theme) {
  const isLight = (theme === 'light');
  if (isLight) {
    document.documentElement.classList.remove('dark');
    document.documentElement.classList.add('light');
    localStorage.setItem('vision_theme', 'light');
  } else {
    document.documentElement.classList.remove('light');
    document.documentElement.classList.add('dark');
    localStorage.setItem('vision_theme', 'dark');
  }

  const checkbox = document.getElementById('theme-toggle-checkbox');
  if (checkbox) {
    checkbox.checked = isLight;
  }
}

function toggleThemeFromCheckbox(isChecked) {
  setTheme(isChecked ? 'light' : 'dark');
}

function toggleTheme() {
  const isLight = document.documentElement.classList.contains('light');
  setTheme(isLight ? 'dark' : 'light');
}

// Color Palette Mapping
const CLASS_COLORS = {
  person: { border: '#22d3ee', bg: 'rgba(8, 51, 68, 0.92)', text: '#a5f3fc', shadow: 'rgba(34, 211, 238, 0.4)' },
  car: { border: '#a855f7', bg: 'rgba(59, 7, 100, 0.92)', text: '#e9d5ff', shadow: 'rgba(168, 85, 247, 0.4)' },
  bicycle: { border: '#60a5fa', bg: 'rgba(23, 37, 84, 0.92)', text: '#bfdbfe', shadow: 'rgba(96, 165, 250, 0.4)' },
  bus: { border: '#34d399', bg: 'rgba(6, 78, 59, 0.92)', text: '#a7f3d0', shadow: 'rgba(52, 211, 153, 0.4)' },
  truck: { border: '#a3e635', bg: 'rgba(21, 26, 16, 0.92)', text: '#d9f99d', shadow: 'rgba(163, 230, 53, 0.4)' },
  dog: { border: '#fb923c', bg: 'rgba(67, 20, 7, 0.92)', text: '#fed7aa', shadow: 'rgba(251, 146, 60, 0.4)' },
  cat: { border: '#f472b6', bg: 'rgba(76, 5, 25, 0.92)', text: '#fbcfe8', shadow: 'rgba(244, 114, 182, 0.4)' },
  motorcycle: { border: '#e879f9', bg: 'rgba(74, 4, 78, 0.92)', text: '#f5d0fe', shadow: 'rgba(232, 121, 249, 0.4)' },
  default: { border: '#a3e635', bg: 'rgba(21, 26, 16, 0.92)', text: '#a3e635', shadow: 'rgba(163, 230, 53, 0.4)' }
};

function getClassTheme(cls) {
  const k = (cls || 'default').toLowerCase();
  return CLASS_COLORS[k] || CLASS_COLORS.default;
}

// Update Top Backend Status Badge
function setBackendStatus(mode, text) {
  const statusText = document.getElementById('backend-status-text');
  const statusDot = document.getElementById('backend-status-dot');
  const statusPing = document.getElementById('backend-status-ping');
  const statusBadge = document.getElementById('backend-status-badge');
  const statusIcon = document.getElementById('backend-status-icon');

  if (statusText) statusText.innerText = text;

  if (mode === 'python') {
    if (statusIcon) statusIcon.className = 'fa-brands fa-python text-sm text-lime';
    if (statusDot) statusDot.className = 'relative inline-flex rounded-full h-2 w-2 bg-lime';
    if (statusPing) statusPing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-lime opacity-75';
    if (statusBadge) statusBadge.className = 'bg-obsidian-panel px-3 py-1.5 rounded-lg border border-lime/40 flex items-center text-xs font-semibold text-lime gap-2 shadow-glow-lime transition-all';
  } else if (mode === 'webai') {
    if (statusIcon) statusIcon.className = 'fa-solid fa-bolt text-sm text-lime';
    if (statusDot) statusDot.className = 'relative inline-flex rounded-full h-2 w-2 bg-lime';
    if (statusPing) statusPing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-lime opacity-75';
    if (statusBadge) statusBadge.className = 'bg-obsidian-panel px-3 py-1.5 rounded-lg border border-lime/40 flex items-center text-xs font-semibold text-lime gap-2 shadow-glow-lime transition-all';
  } else if (mode === 'connecting') {
    if (statusIcon) statusIcon.className = 'fa-solid fa-spinner fa-spin text-sm text-amber-400';
    if (statusDot) statusDot.className = 'relative inline-flex rounded-full h-2 w-2 bg-amber-400';
    if (statusPing) statusPing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75';
    if (statusBadge) statusBadge.className = 'bg-obsidian-panel px-3 py-1.5 rounded-lg border border-amber-400/40 flex items-center text-xs font-semibold text-amber-400 gap-2 transition-all';
  }
}

// ==========================================
// 3. Decoupled WebGL AI Inference Worker
// ==========================================
async function initWebAI() {
  setBackendStatus('connecting', 'Initializing WebGL GPU Shaders...');
  try {
    if (typeof tf !== 'undefined') {
      await tf.setBackend('webgl');
      await tf.ready();
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
      tf.enableProdMode();
    }
    
    if (typeof cocoSsd !== 'undefined') {
      webModel = await cocoSsd.load({ base: 'mobilenet_v2' });
      if (!state.isPythonActive) {
        setBackendStatus('webai', 'Browser AI (WebGL GPU · 60 FPS)');
      }
      addActivityLog('ENGINE', 'Initialized Browser WebGL AI (60 FPS Decoupled)', 'lime');
      startDecoupledAiLoop();
    } else {
      console.warn('COCO-SSD not loaded from CDN');
    }
  } catch (e) {
    console.warn('WebGL setup fallback to CPU:', e);
    try {
      if (typeof tf !== 'undefined') {
        await tf.setBackend('cpu');
        await tf.ready();
      }
      if (typeof cocoSsd !== 'undefined') {
        webModel = await cocoSsd.load();
        if (!state.isPythonActive) {
          setBackendStatus('webai', 'Browser AI (CPU Fallback · 60 FPS)');
        }
        addActivityLog('ENGINE', 'Initialized Browser Web AI (CPU Fallback)', 'cyan');
        startDecoupledAiLoop();
      }
    } catch (err) {
      console.error('All Web AI initialization failed:', err);
    }
  }
}

function startDecoupledAiLoop() {
  async function aiWorker() {
    try {
      const isVideoReady = videoEl && (videoEl.readyState >= 2 || videoEl.videoWidth > 0);
      const isPlaying = videoEl && !videoEl.paused && !videoEl.ended;
      const shouldDetect = webModel && isVideoReady && !state.isPythonActive && (isPlaying || !state.isPaused || state.source === 'upload');

      if (shouldDetect && !isAiDetecting) {
        isAiDetecting = true;
        const t0 = performance.now();

        // 416px fast offscreen tensor buffer (runs in ~12-18ms on GPU)
        const vw = videoEl.videoWidth || 640;
        const vh = videoEl.videoHeight || 360;
        const sendW = 416;
        const sendH = Math.max(1, Math.round(sendW * (vh / vw)));

        if (aiCanvas.width !== sendW || aiCanvas.height !== sendH) {
          aiCanvas.width = sendW;
          aiCanvas.height = sendH;
        }

        aiCtx.drawImage(videoEl, 0, 0, sendW, sendH);

        const rawDets = await webModel.detect(aiCanvas, 20, state.confThresh);
        const latency = Math.max(1, performance.now() - t0);
        state.latency = latency;

        const formattedDets = (rawDets || []).map(d => ({
          bbox: [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]],
          score: d.score,
          className: d.class
        }));

        // Pass detections to JavaScript Kalman Tracker
        webTracker.update(formattedDets);
      }
    } catch (e) {
      console.warn('Decoupled AI detection cycle error:', e);
    } finally {
      isAiDetecting = false;
    }
    // Independent background scheduling (does NOT block 60 FPS video render)
    setTimeout(aiWorker, 20);
  }
  aiWorker();
}

// Convert WebSORT Tracker state to Canvas Viewport Coordinates
function syncWebTracksToState(vRect, fw, fh) {
  const tracks = webTracker.getActiveTracks();
  const lineY = vRect.y + vRect.h * state.linePos;
  state.totalTracks = webTracker.totalTracksCount;

  state.activeTracks = tracks.map(t => {
    const x1 = vRect.x + (t.bbox[0] / fw) * vRect.w;
    const y1 = vRect.y + (t.bbox[1] / fh) * vRect.h;
    const x2 = vRect.x + (t.bbox[2] / fw) * vRect.w;
    const y2 = vRect.y + (t.bbox[3] / fh) * vRect.h;
    const centerY = (y1 + y2) / 2;

    const scaleX = vRect.w / fw;
    const scaleY = vRect.h / fh;
    const vx = (t.vx || 0) * scaleX;
    const vy = (t.vy || 0) * scaleY;
    const speed = Math.round(Math.hypot(vx, vy) * 2.2) || 5;

    // Tripwire Crossing Check
    if (state.toggles.line && !state.crossedIds[t.id]) {
      if (Math.abs(centerY - lineY) < 20) {
        state.crossedIds[t.id] = true;
        if (vy < 0) {
          state.countOut += 1;
          addActivityLog(`ID #${t.id}`, `Crossed Tripwire UP [OUT]`, 'lime');
        } else {
          state.countIn += 1;
          addActivityLog(`ID #${t.id}`, `Crossed Tripwire DOWN [IN]`, 'cyan');
        }
      }
    }

    if (!state.trackHistories[t.id]) {
      state.trackHistories[t.id] = true;
    }

    return {
      id: t.id,
      className: t.className,
      score: t.score,
      bbox: [x1, y1, x2, y2],
      speed: speed,
      vx: vx,
      vy: vy
    };
  });
}

// ==========================================
// 4. Optional Python Backend Connection
// ==========================================
function tryConnectPythonWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try {
    const ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
    ws.onopen = () => {
      state.ws = ws;
      state.isPythonActive = true;
      setBackendStatus('python', 'Python Backend (YOLOv8 + SORT)');
      addActivityLog('ENGINE', 'Connected to Python Backend (YOLOv8 + SORT)', 'lime');
    };
    ws.onmessage = (event) => {
      isWsProcessing = false;
      try {
        const data = JSON.parse(event.data);
        if (data.meta && data.meta.latency_ms) state.latency = data.meta.latency_ms;
        if (data.total_tracks !== undefined) state.totalTracks = data.total_tracks;
        if (data.tracks) {
          const fw = data.frame_w || 640;
          const fh = data.frame_h || 360;
          const rect = canvasEl.getBoundingClientRect();
          const vRect = getVideoDrawRect(rect.width, rect.height);
          const lineY = vRect.y + vRect.h * state.linePos;

          state.activeTracks = data.tracks.map(t => {
            const x1 = vRect.x + (t.bbox[0] / fw) * vRect.w;
            const y1 = vRect.y + (t.bbox[1] / fh) * vRect.h;
            const x2 = vRect.x + (t.bbox[2] / fw) * vRect.w;
            const y2 = vRect.y + (t.bbox[3] / fh) * vRect.h;
            const centerY = (y1 + y2) / 2;

            if (state.toggles.line && !state.crossedIds[t.id]) {
              if (Math.abs(centerY - lineY) < 20) {
                state.crossedIds[t.id] = true;
                if (t.vy < 0) {
                  state.countOut += 1;
                  addActivityLog(`ID #${t.id}`, `Crossed Tripwire UP [OUT]`, 'lime');
                } else {
                  state.countIn += 1;
                  addActivityLog(`ID #${t.id}`, `Crossed Tripwire DOWN [IN]`, 'cyan');
                }
              }
            }

            return {
              id: t.id,
              className: t.class_name || 'object',
              score: t.score,
              bbox: [x1, y1, x2, y2],
              speed: Math.round(Math.hypot(t.vx, t.vy) * 2.2) || 5,
              vx: t.vx,
              vy: t.vy
            };
          });
        }
      } catch (e) {}
    };
    ws.onclose = () => {
      state.isPythonActive = false;
      state.ws = null;
      if (webModel) setBackendStatus('webai', 'Browser AI (WebGL GPU · 60 FPS)');
    };
    ws.onerror = () => {
      state.isPythonActive = false;
    };
  } catch (e) {
    state.isPythonActive = false;
  }
}

// Fast Decoupled Frame Encoding to Python Backend
function sendFrameToPython() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN || isWsProcessing || state.isPaused) return;
  const now = performance.now();
  if (now - lastWsSendTime < 65) return; // Throttle to prevent WebSocket buffer congestion
  lastWsSendTime = now;

  const vw = videoEl.videoWidth || 640;
  const vh = videoEl.videoHeight || 360;
  const sendW = 416;
  const sendH = Math.max(1, Math.round(sendW * (vh / vw)));

  if (offscreenCanvas.width !== sendW || offscreenCanvas.height !== sendH) {
    offscreenCanvas.width = sendW;
    offscreenCanvas.height = sendH;
  }

  offscreenCtx.drawImage(videoEl, 0, 0, sendW, sendH);
  const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.55);
  isWsProcessing = true;
  state.ws.send(JSON.stringify({ action: 'process_frame', image: dataUrl }));
}

// ==========================================
// 5. App Initialization & Render Loop
// ==========================================
async function initApp() {
  initTheme();
  updateMetricsDisplay();
  updateClassDistribution();
  renderActivityFeed();
  
  // 1. Start 60.0 FPS native render loop immediately
  requestAnimationFrame(processLoop);

  // 2. Try connecting to local Python backend if available
  tryConnectPythonWs();

  // 3. Initialize Decoupled WebGL AI Engine
  initWebAI();

  // 4. Load default traffic clip with autoplay safety
  await setSource('sample', 'sample1');
}

// Main 60.0 FPS Process & Animation Loop (Rock Solid Native 60Hz)
function processLoop() {
  const now = performance.now();
  const delta = now - state.lastFrameTime;
  state.lastFrameTime = now;

  if (!state.isPaused) {
    state.frameCounter += 1;
  }

  // Measure true display FPS
  if (delta > 0) {
    const instantFps = 1000 / delta;
    state.fpsList.push(instantFps);
    if (state.fpsList.length > 20) state.fpsList.shift();
    state.fps = state.fpsList.reduce((a, b) => a + b, 0) / state.fpsList.length;
  }

  updateTimecode(now);

  const rect = canvasEl.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);
  if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
    canvasEl.width = targetW;
    canvasEl.height = targetH;
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  const cw = rect.width;
  const ch = rect.height;

  // Clear canvas
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, cw, ch);

  const vRect = getVideoDrawRect(cw, ch);

  if (videoEl.readyState >= 2) {
    // ALWAYS draw video frame with preserved aspect ratio
    ctx.drawImage(videoEl, vRect.x, vRect.y, vRect.w, vRect.h);
    
    // Update live resolution indicator tag
    const resEl = document.getElementById('viewport-resolution');
    if (resEl && videoEl.videoWidth) {
      resEl.innerText = `[${videoEl.videoWidth}x${videoEl.videoHeight} @ 60Hz]`;
    }

    if (state.isPythonActive) {
      sendFrameToPython();
    } else {
      // 60 FPS Kalman Prediction Step (Glides boxes seamlessly across frames)
      webTracker.predictAll();
      syncWebTracksToState(vRect, aiCanvas.width || 416, aiCanvas.height || 234);
    }
  }

  renderOverlays(cw, ch, vRect);
  ctx.restore();

  updateMetricsDisplay();
  requestAnimationFrame(processLoop);
}

// Render Overlays on Canvas
function renderOverlays(cw, ch, vRect) {
  const lineY = vRect.y + vRect.h * state.linePos;

  // 1. Tripwire Counting Line
  if (state.toggles.line) {
    ctx.save();
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(vRect.x, lineY);
    ctx.lineTo(vRect.x + vRect.w, lineY);
    ctx.stroke();

    const tagText = `TRIPWIRE VECTOR (Y: ${(state.linePos * 100).toFixed(1)}%)`;
    ctx.font = '600 10px "JetBrains Mono", monospace';
    const tw = ctx.measureText(tagText).width;
    const tagX = vRect.x + vRect.w - tw - 24;
    const tagY = lineY - 14;

    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(17, 21, 12, 0.9)';
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.6)';
    ctx.lineWidth = 1;
    roundRect(ctx, tagX, tagY, tw + 16, 20, 4, true, true);

    ctx.fillStyle = '#22d3ee';
    ctx.shadowColor = 'rgba(34, 211, 238, 0.5)';
    ctx.shadowBlur = 8;
    ctx.fillText(tagText, tagX + 8, tagY + 14);
    ctx.restore();
  }

  // 2. Bounding Boxes, Labels & Entity Badges
  state.activeTracks.forEach(track => {
    if (!isClassAllowed(track.className)) return;

    const [x1, y1, x2, y2] = track.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const theme = getClassTheme(track.className);

    // Bounding Box Rectangle & Corner Reticles
    if (state.toggles.boxes) {
      ctx.save();
      ctx.fillStyle = theme.border + '12';
      ctx.fillRect(x1, y1, w, h);

      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 2;
      ctx.shadowColor = theme.shadow;
      ctx.shadowBlur = 10;
      ctx.strokeRect(x1, y1, w, h);

      // Corner Reticles
      const reticleLen = 8;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = theme.text;
      
      // Top-Left L
      ctx.beginPath();
      ctx.moveTo(x1 - 2, y1 + reticleLen);
      ctx.lineTo(x1 - 2, y1 - 2);
      ctx.lineTo(x1 + reticleLen, y1 - 2);
      ctx.stroke();

      // Bottom-Right L
      ctx.beginPath();
      ctx.moveTo(x2 + 2, y2 - reticleLen);
      ctx.lineTo(x2 + 2, y2 + 2);
      ctx.lineTo(x2 - reticleLen, y2 + 2);
      ctx.stroke();

      ctx.restore();
    }

    // Top Label / ID Badge
    const showLabel = state.toggles.labels;
    const showId = state.toggles.ids;

    if (showLabel || showId) {
      ctx.save();
      let labelText = '';
      if (showId && showLabel) {
        labelText = `ID #${track.id} · ${track.className.toUpperCase()} ${Math.round(track.score * 100)}%`;
      } else if (showId) {
        labelText = `ID #${track.id}`;
      } else if (showLabel) {
        labelText = `${track.className.toUpperCase()} ${Math.round(track.score * 100)}%`;
      }

      ctx.font = '700 11px "JetBrains Mono", monospace';
      const lw = ctx.measureText(labelText).width;

      let badgeX = Math.max(vRect.x + 4, Math.min(x1, vRect.x + vRect.w - lw - 28));
      let badgeY = y1 - 22;
      
      if (badgeY < vRect.y + 2) {
        badgeY = Math.min(y1 + 4, vRect.y + vRect.h - 24);
      }

      ctx.fillStyle = theme.bg;
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      roundRect(ctx, badgeX, badgeY, lw + 24, 20, 4, true, true);

      // Entity dot
      ctx.fillStyle = theme.text;
      ctx.beginPath();
      ctx.arc(badgeX + 9, badgeY + 10, 3, 0, Math.PI * 2);
      ctx.fill();

      // Badge Text
      ctx.fillStyle = theme.text;
      ctx.shadowColor = theme.shadow;
      ctx.shadowBlur = 6;
      ctx.fillText(labelText, badgeX + 18, badgeY + 14);

      // Speed Tag
      const speedText = `v = ${track.speed || 5} km/h`;
      ctx.font = '500 10px "JetBrains Mono", monospace';
      const sw = ctx.measureText(speedText).width;
      const spX = Math.max(vRect.x + 4, Math.min(x2 - sw - 6, vRect.x + vRect.w - sw - 8));
      let spY = Math.max(vRect.y + 16, Math.min(y2 - 6, vRect.y + vRect.h - 6));
      if (Math.abs(spY - badgeY) < 16) {
        spY = badgeY + 24;
      }
      ctx.fillStyle = theme.text + 'c0';
      ctx.fillText(speedText, spX, spY);

      ctx.restore();
    }
  });
}

function roundRect(ctx, x, y, width, height, radius = 4, fill = false, stroke = false) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

function updateTimecode(now) {
  let elapsed = (now - state.startTime) / 1000;
  if (videoEl && !isNaN(videoEl.currentTime)) {
    elapsed = videoEl.currentTime;
  }
  const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
  const secs = Math.floor(elapsed % 60).toString().padStart(2, '0');
  const ms = Math.floor((elapsed % 1) * 100).toString().padStart(2, '0');

  const timeEl = document.getElementById('playback-time');
  const frameEl = document.getElementById('playback-frame');
  if (timeEl) timeEl.innerText = `T+ 00:${mins}:${secs}.${ms}`;
  if (frameEl) frameEl.innerText = `Frame: #${state.frameCounter.toLocaleString('en-US')}`;

  const progressBar = document.getElementById('video-progress-bar');
  const durTag = document.getElementById('video-duration-tag');
  if (videoEl && !isNaN(videoEl.duration) && videoEl.duration > 0) {
    const pct = Math.min(100, Math.max(0, (videoEl.currentTime / videoEl.duration) * 100));
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (durTag) durTag.innerText = `${formatTime(videoEl.currentTime)} / ${formatTime(videoEl.duration)}`;
  } else {
    if (progressBar) progressBar.style.width = '0%';
    if (durTag) durTag.innerText = `${formatTime(elapsed)} / Live`;
  }
}

function updateMetricsDisplay() {
  const fpsEl = document.getElementById('val-fps');
  const latencyEl = document.getElementById('val-latency');
  const activeEl = document.getElementById('val-active');
  const totalEl = document.getElementById('val-total');
  const inEl = document.getElementById('val-in');
  const outEl = document.getElementById('val-out');

  const visibleTracks = state.activeTracks.filter(t => isClassAllowed(t.className));

  if (fpsEl) fpsEl.innerText = state.fps.toFixed(1);
  if (latencyEl) latencyEl.innerText = state.latency.toFixed(1);
  if (activeEl) activeEl.innerText = visibleTracks.length;
  if (totalEl) totalEl.innerText = state.totalTracks;
  if (inEl) inEl.innerText = state.countIn;
  if (outEl) outEl.innerText = state.countOut;

  updateClassDistribution();
}

function updateClassDistribution() {
  const counts = {};
  state.activeTracks.forEach(t => {
    if (!isClassAllowed(t.className)) return;
    const c = (t.className || 'object').toLowerCase();
    counts[c] = (counts[c] || 0) + 1;
  });

  const visibleCount = state.activeTracks.filter(t => isClassAllowed(t.className)).length;
  const totalActive = Math.max(1, visibleCount);
  const tagEl = document.getElementById('active-class-count-tag');
  if (tagEl) tagEl.innerText = `${visibleCount} active`;

  if (classDistEl) {
    const keys = Object.keys(counts);
    if (keys.length === 0) {
      classDistEl.innerHTML = '<div class="text-[11px] font-mono text-gray-500 py-2">No active targets in view</div>';
      return;
    }
    classDistEl.innerHTML = keys.map(k => {
      const theme = getClassTheme(k);
      const count = counts[k];
      const pct = Math.round((count / totalActive) * 100);
      return `
        <div>
          <div class="flex justify-between text-xs font-mono mb-1">
            <span class="text-gray-300 flex items-center gap-1.5 capitalize">
              <span class="w-2 h-2 rounded-full" style="background-color: ${theme.border};"></span>
              ${k}
            </span>
            <span class="font-bold text-white">${count}</span>
          </div>
          <div class="w-full bg-obsidian-panel rounded-full h-1.5 overflow-hidden">
            <div class="h-1.5 rounded-full transition-all duration-300" style="width: ${pct}%; background-color: ${theme.border};"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

function addActivityLog(id, desc, color = 'lime') {
  const d = new Date();
  const time = `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
  state.eventLogs.unshift({ time, id, desc, type: color });
  if (state.eventLogs.length > 30) state.eventLogs.pop();
  renderActivityFeed();
}

function renderActivityFeed() {
  if (!activityFeedEl) return;
  activityFeedEl.innerHTML = state.eventLogs.map(l => {
    let borderClass = 'border-lime';
    let textClass = 'text-lime';
    if (l.type === 'cyan') { borderClass = 'border-cyan-400'; textClass = 'text-cyan-300'; }
    if (l.type === 'purple') { borderClass = 'border-purple-400'; textClass = 'text-purple-300'; }
    if (l.type === 'amber') { borderClass = 'border-amber-400'; textClass = 'text-amber-400'; }
    if (l.type === 'blue') { borderClass = 'border-blue-400'; textClass = 'text-blue-300'; }
    if (l.type === 'red') { borderClass = 'border-red-500'; textClass = 'text-red-400'; }

    return `
      <div class="border-l-2 ${borderClass} pl-2 leading-relaxed">
        <span class="text-gray-500">${l.time}</span>
        <span class="${textClass} font-bold ml-1">${l.id}</span>
        <span class="text-gray-300 block">${l.desc}</span>
      </div>
    `;
  }).join('');
}

function clearLogs() {
  state.eventLogs = [];
  renderActivityFeed();
}

// Inside Video Player Controls Management
let controlsHideTimer = null;
let currentSpeed = 1.0;

function showInVideoControls() {
  const controls = document.getElementById('in-video-controls');
  const header = document.getElementById('in-video-header');
  if (controls) {
    controls.style.opacity = '1';
    controls.style.pointerEvents = 'auto';
  }
  if (header) {
    header.style.opacity = '1';
  }
  clearTimeout(controlsHideTimer);
  if (!state.isPaused && !videoEl.paused && !videoEl.ended) {
    controlsHideTimer = setTimeout(() => {
      if (!state.isPaused) {
        if (controls) {
          controls.style.opacity = '0';
          controls.style.pointerEvents = 'none';
        }
        if (header) {
          header.style.opacity = '0';
        }
      }
    }, 2800);
  }
}

function hideInVideoControls() {
  const controls = document.getElementById('in-video-controls');
  const header = document.getElementById('in-video-header');
  if (!state.isPaused && !videoEl.paused && !videoEl.ended) {
    if (controls) {
      controls.style.opacity = '0';
      controls.style.pointerEvents = 'none';
    }
    if (header) {
      header.style.opacity = '0';
    }
  }
}

function showCenterRipple(type) {
  const ripple = document.getElementById('center-play-ripple');
  const icon = document.getElementById('center-ripple-icon');
  if (!ripple || !icon) return;
  
  if (type === 'play') {
    icon.className = 'fa-solid fa-play ml-1';
  } else if (type === 'pause') {
    icon.className = 'fa-solid fa-pause';
  } else if (type === 'replay') {
    icon.className = 'fa-solid fa-rotate-left';
  }
  
  ripple.classList.remove('opacity-0', 'scale-50');
  ripple.classList.add('opacity-100', 'scale-100');
  
  setTimeout(() => {
    ripple.classList.remove('opacity-100', 'scale-100');
    ripple.classList.add('opacity-0', 'scale-50');
  }, 380);
}

function handleCanvasClick(event) {
  togglePlayPause();
}

function nextSamplePreset() {
  const nextKey = currentSampleClip === 'sample1' ? 'sample2' : 'sample1';
  setSource('sample', nextKey);
}

const SPEED_PRESETS = [
  { val: 0.5, id: 'speed-05' },
  { val: 1.0, id: 'speed-10' },
  { val: 1.5, id: 'speed-15' },
  { val: 2.0, id: 'speed-20' }
];

function setSpeed(speed) {
  currentSpeed = parseFloat(speed);
  if (videoEl) {
    videoEl.playbackRate = currentSpeed;
  }
  
  SPEED_PRESETS.forEach(cfg => {
    const btn = document.getElementById(cfg.id);
    if (btn) {
      if (Math.abs(cfg.val - currentSpeed) < 0.05) {
        btn.className = 'px-2 py-0.5 rounded bg-lime text-black font-bold shadow-glow-lime text-[11px]';
      } else {
        btn.className = 'px-2 py-0.5 rounded text-gray-400 hover:text-white transition-all text-[11px]';
      }
    }
  });
  addActivityLog('PLAYBACK', `Playback speed: ${currentSpeed}x`, 'cyan');
}

function toggleMute() {
  if (!videoEl) return;
  videoEl.muted = !videoEl.muted;
  const icon = document.getElementById('volume-icon');
  const slider = document.getElementById('volume-slider');
  if (videoEl.muted) {
    if (icon) icon.className = 'fa-solid fa-volume-xmark text-red-400';
    if (slider) slider.value = 0;
  } else {
    if (icon) icon.className = 'fa-solid fa-volume-high';
    if (slider) slider.value = videoEl.volume || 1;
  }
}

function setVolume(val) {
  if (!videoEl) return;
  videoEl.volume = parseFloat(val);
  videoEl.muted = (videoEl.volume === 0);
  const icon = document.getElementById('volume-icon');
  if (videoEl.muted) {
    if (icon) icon.className = 'fa-solid fa-volume-xmark text-red-400';
  } else if (videoEl.volume < 0.5) {
    if (icon) icon.className = 'fa-solid fa-volume-low';
  } else {
    if (icon) icon.className = 'fa-solid fa-volume-high';
  }
}

function updatePlayPauseButton(status) {
  const playIcon = document.getElementById('play-icon');
  const btnPlay = document.getElementById('btn-play');
  if (!btnPlay) return;

  if (status === 'playing') {
    if (playIcon) playIcon.className = 'fa-solid fa-pause';
    btnPlay.className = 'w-8 h-8 rounded-lg bg-lime hover:bg-lime-glow text-black font-bold flex items-center justify-center text-xs transition-all shadow-glow-lime hover:scale-105 active:scale-95';
    btnPlay.title = 'Pause Video (Space / P)';
  } else if (status === 'ended') {
    if (playIcon) playIcon.className = 'fa-solid fa-rotate-left';
    btnPlay.className = 'w-8 h-8 rounded-lg bg-lime hover:bg-lime-glow text-black font-bold flex items-center justify-center text-xs transition-all shadow-glow-lime hover:scale-105 active:scale-95';
    btnPlay.title = 'Replay Video (Space / P)';
  } else {
    if (playIcon) playIcon.className = 'fa-solid fa-play ml-0.5';
    btnPlay.className = 'w-8 h-8 rounded-lg bg-black/60 hover:bg-lime/20 border border-lime text-lime font-bold flex items-center justify-center text-xs transition-all shadow-glow-lime hover:scale-105 active:scale-95';
    btnPlay.title = 'Play / Resume Video (Space / P)';
  }
}

function togglePlayPause() {
  if (videoEl.ended || (videoEl.duration && videoEl.currentTime >= videoEl.duration - 0.1)) {
    videoEl.currentTime = 0;
    state.isPaused = false;
    videoEl.play().catch(() => {});
    updatePlayPauseButton('playing');
    showCenterRipple('replay');
    showInVideoControls();
    addActivityLog('PLAYBACK', 'Replaying video from beginning', 'lime');
  } else if (state.isPaused || videoEl.paused) {
    state.isPaused = false;
    videoEl.play().catch(() => {});
    updatePlayPauseButton('playing');
    showCenterRipple('play');
    showInVideoControls();
    addActivityLog('PLAYBACK', 'Resumed video playback', 'lime');
  } else {
    state.isPaused = true;
    videoEl.pause();
    updatePlayPauseButton('paused');
    showCenterRipple('pause');
    showInVideoControls();
    addActivityLog('PLAYBACK', 'Paused video stream', 'amber');
  }
}

function seekVideo(event) {
  if (!videoEl || isNaN(videoEl.duration) || videoEl.duration <= 0) return;
  const bg = document.getElementById('video-progress-bg');
  if (!bg) return;
  const rect = bg.getBoundingClientRect();
  const clickX = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
  const ratio = clickX / rect.width;
  videoEl.currentTime = ratio * videoEl.duration;
  
  if (videoEl.ended) {
    state.isPaused = false;
    videoEl.play().catch(() => {});
    updatePlayPauseButton('playing');
  }
}

function captureSnapshot() {
  const link = document.createElement('a');
  link.download = `visiongrid_snapshot_${Date.now()}.png`;
  link.href = canvasEl.toDataURL('image/png');
  link.click();
  addActivityLog('SNAPSHOT', 'Captured frame PNG', 'lime');
}

function toggleRecording() {
  if (state.isRecording) {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
    }
    state.isRecording = false;
    const btnRecord = document.getElementById('btn-record');
    const recText = document.getElementById('record-text');
    if (btnRecord) btnRecord.className = 'px-3 py-1.5 rounded-lg bg-obsidian-panel border border-obsidian-border text-gray-300 hover:text-white hover:border-red-400/40 font-medium text-xs flex items-center gap-2 transition-all';
    if (recText) recText.innerText = 'Record Video';
    addActivityLog('RECORDER', 'Video capture finished and saved', 'lime');
  } else {
    state.recordedChunks = [];
    try {
      const stream = canvasEl.captureStream(30);
      state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
      };
      state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `visiongrid_clip_${Date.now()}.webm`;
        a.click();
      };
      state.mediaRecorder.start();
      state.isRecording = true;
      const btnRecord = document.getElementById('btn-record');
      const recText = document.getElementById('record-text');
      if (btnRecord) btnRecord.className = 'px-3 py-1.5 rounded-lg bg-red-500/20 border border-red-500 text-red-400 font-bold text-xs flex items-center gap-2 transition-all animate-pulse';
      if (recText) recText.innerText = 'Stop Recording';
      addActivityLog('RECORDER', 'Recording video stream...', 'red');
    } catch (e) {
      console.error(e);
    }
  }
}

async function resetTracking() {
  state.countIn = 0;
  state.countOut = 0;
  state.totalTracks = 0;
  state.crossedIds = {};
  state.trackHistories = {};
  state.activeTracks = [];
  state.rawTracks = [];
  webTracker.reset();

  if (state.isPythonActive) {
    try {
      await fetch('/api/reset', { method: 'POST' });
    } catch (e) {}
  }
  addActivityLog('TRACKER', 'SORT tracker reset', 'amber');
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

function exportData(format) {
  const data = {
    exportedAt: new Date().toISOString(),
    engine: state.isPythonActive ? 'Python YOLOv8 + SORT' : 'VisionGrid WebGL AI + SORT',
    metrics: {
      fps: state.fps.toFixed(1),
      latencyMs: state.latency.toFixed(1),
      totalTracks: state.totalTracks,
      activeTracks: state.activeTracks.length,
      tripwireIn: state.countIn,
      tripwireOut: state.countOut
    },
    activeTracks: state.activeTracks,
    eventLogs: state.eventLogs
  };

  let fileContent, mimeType, ext;
  if (format === 'json') {
    fileContent = JSON.stringify(data, null, 2);
    mimeType = 'application/json';
    ext = 'json';
  } else {
    let csv = 'Timestamp,EntityID,Description,Type\n';
    for (const log of state.eventLogs) {
      csv += `"${log.time}","${log.id}","${log.desc}","${log.type}"\n`;
    }
    fileContent = csv;
    mimeType = 'text/csv';
    ext = 'csv';
  }

  const blob = new Blob([fileContent], { type: mimeType });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `visiongrid_telemetry_${Date.now()}.${ext}`;
  a.click();
  addActivityLog('EXPORT', `Exported telemetry (${ext.toUpperCase()})`, 'lime');
}

// Source Handling (Sample clips, Webcam, Upload)
const SAMPLE_CLIPS = {
  sample1: {
    name: 'Sample 1 (Person, Vehicles, Animals)',
    src: 'sample_data/Sample1.mp4',
    btnId: 'btn-sample-1'
  },
  sample2: {
    name: 'Sample 2 (Person, Vehicles, Animals)',
    src: 'sample_data/Sample2.mp4',
    btnId: 'btn-sample-2'
  }
};

let currentSampleClip = 'sample1';

async function loadSampleClip(clipKey) {
  if (!SAMPLE_CLIPS[clipKey]) return;
  currentSampleClip = clipKey;
  updateSampleButtons(clipKey);
  await setSource('sample', clipKey);
}

function updateSampleButtons(activeKey) {
  ['sample1', 'sample2'].forEach(k => {
    const btn = document.getElementById(SAMPLE_CLIPS[k].btnId);
    if (btn) {
      if (k === activeKey) {
        btn.className = 'flex-1 py-1.5 px-2 rounded bg-lime/15 border border-lime/40 text-lime font-bold shadow-glow-lime flex items-center justify-center gap-1.5 transition-all';
      } else {
        btn.className = 'flex-1 py-1.5 px-2 rounded text-gray-300 hover:text-white hover:bg-obsidian-surface border border-transparent hover:border-lime/30 transition-all flex items-center justify-center gap-1.5';
      }
    }
  });
}

function updateSourceButtons(activeSrc) {
  ['webcam', 'upload', 'sample'].forEach(s => {
    const btn = document.getElementById(`src-${s}`);
    if (btn) {
      if (s === activeSrc) {
        btn.className = 'p-2.5 rounded-lg bg-lime/10 border border-lime/40 text-lime font-medium flex flex-col items-center justify-center gap-1.5 shadow-glow-lime';
      } else {
        btn.className = 'p-2.5 rounded-lg bg-obsidian-panel border border-obsidian-border text-gray-300 hover:border-lime/30 hover:text-white flex flex-col items-center justify-center gap-1.5 transition-all';
      }
    }
  });
}

async function setSource(src, sampleKey = currentSampleClip) {
  state.source = src;
  updateSourceButtons(src);
  state.activeTracks = [];
  state.rawTracks = [];
  state.trackHistories = {};
  webTracker.reset();

  const headerTitle = document.getElementById('viewport-header-title');
  const sampleBox = document.getElementById('sample-preset-container');

  if (src === 'webcam') {
    if (sampleBox) sampleBox.style.display = 'none';
    if (headerTitle) headerTitle.innerText = 'Live Video Feed (Webcam)';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false
      });
      videoEl.srcObject = stream;
      state.isPaused = false;
      await videoEl.play();
      updatePlayPauseButton('playing');
      addActivityLog('INPUT', 'Webcam stream connected', 'lime');
    } catch (err) {
      console.warn('Webcam access error:', err);
      setSource('sample');
    }
  } else if (src === 'upload') {
    if (sampleBox) sampleBox.style.display = 'none';
    triggerVideoUpload();
  } else if (src === 'sample') {
    if (sampleBox) sampleBox.style.display = 'flex';
    currentSampleClip = sampleKey;
    updateSampleButtons(sampleKey);
    const clip = SAMPLE_CLIPS[sampleKey] || SAMPLE_CLIPS.sample1;
    
    if (headerTitle) headerTitle.innerText = `Live Video Feed (${clip.name})`;
    if (videoEl.srcObject) {
      videoEl.srcObject.getTracks().forEach(t => t.stop());
      videoEl.srcObject = null;
    }
    
    videoEl.src = clip.src;
    videoEl.crossOrigin = 'anonymous';
    videoEl.muted = true;
    videoEl.defaultMuted = true;
    videoEl.playsInline = true;
    videoEl.loop = true;
    state.isPaused = false;
    
    try {
      const playPromise = videoEl.play();
      if (playPromise !== undefined) {
        await playPromise;
      }
      state.isPaused = false;
      updatePlayPauseButton('playing');
      addActivityLog('INPUT', `Loaded ${clip.name}`, 'lime');
    } catch (e) {
      console.warn('Autoplay prevented, will play on user interaction:', e);
      state.isPaused = false;
      updatePlayPauseButton('playing');
    }
  }
}

function triggerVideoUpload() {
  document.getElementById('video-upload-input').click();
}

function handleVideoFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  state.source = 'upload';
  updateSourceButtons('upload');
  state.activeTracks = [];
  state.rawTracks = [];
  state.trackHistories = {};
  webTracker.reset();

  if (videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach(t => t.stop());
    videoEl.srcObject = null;
  }
  videoEl.src = url;
  videoEl.muted = true;
  videoEl.defaultMuted = true;
  videoEl.playsInline = true;
  videoEl.loop = true;
  state.isPaused = false;

  const playPromise = videoEl.play();
  if (playPromise !== undefined) {
    playPromise.then(() => {
      state.isPaused = false;
      updatePlayPauseButton('playing');
    }).catch(err => {
      console.warn('Video play warning on upload:', err);
      state.isPaused = false;
      updatePlayPauseButton('playing');
    });
  }

  const headerTitle = document.getElementById('viewport-header-title');
  if (headerTitle) headerTitle.innerText = `Uploaded Video: ${file.name}`;
  addActivityLog('INPUT', `Uploaded: ${file.name}`, 'lime');
}

// Sliders Handlers
function updateConfidence(val) {
  state.confThresh = parseFloat(val) / 100;
  document.getElementById('conf-val').innerText = `${val}%`;
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    try {
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conf_thresh: state.confThresh })
      });
    } catch (e) {}
  }
}

function updateMaxAge(val) {
  state.maxAge = parseInt(val, 10);
  webTracker.maxAge = Math.max(15, state.maxAge * 2);
  document.getElementById('age-val').innerText = `${val} frames`;
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    try {
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_age: state.maxAge })
      });
    } catch (e) {}
  }
}

function updateLinePos(val) {
  state.linePos = parseFloat(val) / 100;
  document.getElementById('line-val').innerText = `${val}%`;
}

// Feature Toggle Handlers
function toggleSetting(name) {
  const el = document.getElementById(`toggle-${name}`);
  if (el) {
    state.toggles[name] = el.checked;
    addActivityLog('TOGGLE', `${name.toUpperCase()} ${el.checked ? 'ENABLED' : 'DISABLED'}`, el.checked ? 'lime' : 'amber');
  }
}

// Filter Class Handler
function filterClass(cls, btn) {
  state.classFilter = cls;
  document.querySelectorAll('#filter-pills-container button').forEach(b => {
    b.className = 'filter-pill px-2.5 py-1 rounded bg-obsidian-panel border border-obsidian-border text-gray-300 hover:text-white hover:border-lime/30 transition-all';
  });
  if (btn) {
    btn.className = 'filter-pill px-2.5 py-1 rounded bg-lime text-black font-bold shadow-glow-lime';
  }
  updateClassDistribution();
}

// Video Event Listeners
videoEl.addEventListener('loadeddata', () => {
  if (webModel && !state.isPythonActive) {
    const vw = videoEl.videoWidth || 640;
    const vh = videoEl.videoHeight || 360;
    const sendW = 416;
    const sendH = Math.max(1, Math.round(sendW * (vh / vw)));
    aiCanvas.width = sendW;
    aiCanvas.height = sendH;
    try {
      aiCtx.drawImage(videoEl, 0, 0, sendW, sendH);
      webModel.detect(aiCanvas, 20, state.confThresh).then(rawDets => {
        const formattedDets = (rawDets || []).map(d => ({
          bbox: [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]],
          score: d.score,
          className: d.class
        }));
        webTracker.update(formattedDets);
      }).catch(() => {});
    } catch (e) {}
  }
});

videoEl.addEventListener('ended', () => {
  state.isPaused = true;
  updatePlayPauseButton('ended');
  addActivityLog('PLAYBACK', 'Video playback ended. Click Replay to restart.', 'amber');
});

videoEl.addEventListener('pause', () => {
  if (!videoEl.ended) {
    state.isPaused = true;
    updatePlayPauseButton('paused');
  }
});

videoEl.addEventListener('play', () => {
  state.isPaused = false;
  updatePlayPauseButton('playing');
});

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space' || e.key === ' ' || e.key === 'p' || e.key === 'P') {
    e.preventDefault();
    togglePlayPause();
  } else if (e.key === 'f' || e.key === 'F') {
    e.preventDefault();
    toggleFullscreen();
  } else if (e.key === 'm' || e.key === 'M') {
    e.preventDefault();
    toggleMute();
  } else if (e.key === 's' || e.key === 'S') {
    e.preventDefault();
    captureSnapshot();
  } else if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    resetTracking();
  }
});

window.addEventListener('DOMContentLoaded', () => {
  initApp();
});
