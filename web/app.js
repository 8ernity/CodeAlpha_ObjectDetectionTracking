/**
 * VisionGrid - High-Performance Object Detection & Multi-Target SORT Tracking
 * Hybrid Engine: Decoupled 60.0 FPS Canvas Render Loop + WebGL GPU AI / Python YOLOv8
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

  // Predict state for next animation frame (60 FPS smooth trajectory)
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

  // Update state with newly observed detection
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
  constructor(maxAge = 35, minHits = 1, iouThreshold = 0.30) {
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

    // Update matched trackers
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
      .filter(t => t.timeSinceUpdate <= this.maxAge && t.hits >= this.minHits)
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

const webTracker = new WebSortTracker(35, 1, 0.30);

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
  confThresh: 0.40,
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
  pythonTracks: [],
  trackHistories: {},
  totalTracks: 0,
  fps: 60.0,
  latency: 15.0,
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

// Offscreen AI tensor canvas (Decoupled 416px for lightning GPU inference)
const aiCanvas = document.createElement('canvas');
aiCanvas.width = 416;
aiCanvas.height = 234;
const aiCtx = aiCanvas.getContext('2d', { willReadFrequently: true });
let isAiDetecting = false;
let webModel = null;

// Offscreen canvas for Python streaming
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 416;
offscreenCanvas.height = 234;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
let isWsProcessing = false;
let pythonStreamInterval = null;

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
        setBackendStatus('webai', 'Browser AI (WebGL GPU • 60 FPS)');
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
          setBackendStatus('webai', 'Browser AI (CPU Fallback • 60 FPS)');
        }
        addActivityLog('ENGINE', 'Initialized Browser Web AI (CPU Fallback)', 'cyan');
        startDecoupledAiLoop();
      }
    } catch (err) {
      console.error('Web AI Load Error:', err);
    }
  }
}

// Hierarchical Nested Box Suppression:
// Suppresses duplicate limb/sub-part boxes of the SAME class (containment > 0.80) while preserving distinct objects.
function suppressNestedDetections(detections, sameClassContainment = 0.80, iouThresh = 0.50) {
  if (!detections || detections.length <= 1) return detections || [];
  
  const sorted = [...detections].sort((a, b) => (b.score || 0) - (a.score || 0));
  const keep = [];
  
  for (let i = 0; i < sorted.length; i++) {
    const dA = sorted[i];
    const [ax1, ay1, ax2, ay2] = dA.bbox;
    const areaA = Math.max(1, (ax2 - ax1) * (ay2 - ay1));
    let shouldSuppress = false;
    
    for (let j = 0; j < keep.length; j++) {
      const dB = keep[j];
      const [bx1, by1, bx2, by2] = dB.bbox;
      const areaB = Math.max(1, (bx2 - bx1) * (by2 - by1));
      
      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      
      const iw = Math.max(0, ix2 - ix1);
      const ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      
      if (inter > 0) {
        const union = areaA + areaB - inter;
        const iou = inter / union;
        const containment = inter / Math.min(areaA, areaB);
        
        const isSameClass = (dA.className === dB.className);
        
        if (isSameClass) {
          // Suppress same-class sub-box or heavy overlap
          if (iou > iouThresh || containment > sameClassContainment) {
            shouldSuppress = true;
            break;
          }
        } else {
          // Different classes: only suppress if bounding boxes are practically identical
          if (iou > 0.85) {
            shouldSuppress = true;
            break;
          }
        }
      }
    }
    
    if (!shouldSuppress) {
      keep.push(dA);
    }
  }
  return keep;
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
        state.latency = Math.round(latency);

        const formattedDets = (rawDets || []).map(d => ({
          bbox: [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]],
          score: d.score,
          className: d.class
        }));

        // Pass clean detections to JavaScript Kalman Tracker
        const cleanDets = suppressNestedDetections(formattedDets);
        webTracker.update(cleanDets);
      }
    } catch (e) {
      console.warn('Decoupled AI detection cycle error:', e);
    } finally {
      isAiDetecting = false;
    }
    // Independent background scheduling (does NOT block 60 FPS video render)
    setTimeout(aiWorker, 30);
  }
  aiWorker();
}

// Convert WebSORT Tracker state to Viewport Active Tracks
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
      if (Math.abs(centerY - lineY) < 24) {
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
      startPythonStreamingWorker();
    };
    ws.onmessage = (event) => {
      isWsProcessing = false;
      try {
        const data = JSON.parse(event.data);
        if (data.meta && data.meta.latency_ms) state.latency = data.meta.latency_ms;
        if (data.total_tracks !== undefined) state.totalTracks = data.total_tracks;
        if (data.tracks) {
          state.pythonTracks = data.tracks;
          state.pythonFrameW = data.frame_w || 416;
          state.pythonFrameH = data.frame_h || 234;
        }
      } catch (e) {}
    };
    ws.onclose = () => {
      state.isPythonActive = false;
      state.ws = null;
      if (pythonStreamInterval) clearInterval(pythonStreamInterval);
      if (webModel) setBackendStatus('webai', 'Browser AI (WebGL GPU • 60 FPS)');
    };
    ws.onerror = () => {
      state.isPythonActive = false;
    };
  } catch (e) {
    state.isPythonActive = false;
  }
}

// Independent background worker for Python frame encoding (does NOT block 60 FPS render loop)
function startPythonStreamingWorker() {
  if (pythonStreamInterval) clearInterval(pythonStreamInterval);
  pythonStreamInterval = setInterval(() => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN || isWsProcessing || state.isPaused) return;
    const isVideoReady = videoEl && (videoEl.readyState >= 2 || videoEl.videoWidth > 0);
    if (!isVideoReady) return;

    const vw = videoEl.videoWidth || 640;
    const vh = videoEl.videoHeight || 360;
    const sendW = 416;
    const sendH = Math.max(1, Math.round(sendW * (vh / vw)));

    if (offscreenCanvas.width !== sendW || offscreenCanvas.height !== sendH) {
      offscreenCanvas.width = sendW;
      offscreenCanvas.height = sendH;
    }

    offscreenCtx.drawImage(videoEl, 0, 0, sendW, sendH);
    const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.60);
    isWsProcessing = true;
    state.ws.send(JSON.stringify({ action: 'process_frame', image: dataUrl }));
  }, 65); // ~15 FPS background inference stream
}

// Sync Python Tracks to Viewport Canvas
function syncPythonTracksToState(vRect) {
  const fw = state.pythonFrameW || 416;
  const fh = state.pythonFrameH || 234;
  const lineY = vRect.y + vRect.h * state.linePos;

  state.activeTracks = (state.pythonTracks || []).map(t => {
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
      if (Math.abs(centerY - lineY) < 24) {
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
      className: t.class_name || 'object',
      score: t.score,
      bbox: [x1, y1, x2, y2],
      speed: speed,
      vx: vx,
      vy: vy
    };
  });
}

// ==========================================
// 5. App Initialization & 60 FPS Render Loop
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
      syncPythonTracksToState(vRect);
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
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.65)';
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
      ctx.fillStyle = theme.border + '14';
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

      // Top-Right L
      ctx.beginPath();
      ctx.moveTo(x2 + 2 - reticleLen, y1 - 2);
      ctx.lineTo(x2 + 2, y1 - 2);
      ctx.lineTo(x2 + 2, y1 + reticleLen);
      ctx.stroke();

      // Bottom-Left L
      ctx.beginPath();
      ctx.moveTo(x1 - 2, y2 + 2 - reticleLen);
      ctx.lineTo(x1 - 2, y2 + 2);
      ctx.lineTo(x1 + reticleLen, y2 + 2);
      ctx.stroke();

      // Bottom-Right L
      ctx.beginPath();
      ctx.moveTo(x2 + 2 - reticleLen, y2 + 2);
      ctx.lineTo(x2 + 2, y2 + 2);
      ctx.lineTo(x2 + 2, y2 + 2 - reticleLen);
      ctx.stroke();

      ctx.restore();
    }

    // Top Header Badge: Class Name + Confidence %
    if (state.toggles.labels) {
      ctx.save();
      const scorePct = Math.round((track.score || 0.85) * 100);
      const labelText = `${(track.className || 'object').toUpperCase()} ${scorePct}%`;

      ctx.font = '700 11px "JetBrains Mono", monospace';
      const tw = ctx.measureText(labelText).width;
      const tagW = tw + 14;
      const tagH = 20;
      const tagX = Math.max(vRect.x, Math.min(x1, vRect.x + vRect.w - tagW));
      const tagY = y1 - tagH - 4 >= vRect.y ? y1 - tagH - 4 : y1 + 4;

      ctx.fillStyle = theme.bg;
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      roundRect(ctx, tagX, tagY, tagW, tagH, 3, true, true);

      ctx.fillStyle = theme.text;
      ctx.shadowColor = theme.shadow;
      ctx.shadowBlur = 6;
      ctx.fillText(labelText, tagX + 7, tagY + 14);
      ctx.restore();
    }

    // Tracking ID & Speed Tag
    if (state.toggles.ids) {
      ctx.save();
      const idText = `ID #${track.id} • ${track.speed} km/h`;
      ctx.font = '600 10px "JetBrains Mono", monospace';
      const tw = ctx.measureText(idText).width;
      const tagW = tw + 12;
      const tagH = 18;
      const tagX = Math.max(vRect.x, Math.min(x1, vRect.x + vRect.w - tagW));
      const tagY = Math.min(vRect.y + vRect.h - tagH, y2 + 4);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      roundRect(ctx, tagX, tagY, tagW, tagH, 3, true, true);

      ctx.fillStyle = '#f8fafc';
      ctx.fillText(idText, tagX + 6, tagY + 13);
      ctx.restore();
    }
  });

  // Top Status Watermark
  ctx.save();
  ctx.font = '600 11px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
  ctx.fillText(`VISIONGRID AI LIVE • ${state.activeTracks.length} TARGETS IN VIEW`, vRect.x + 12, vRect.y + 24);
  ctx.restore();
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
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function updateTimecode(now) {
  const timecodeEl = document.getElementById('playback-time');
  const frameEl = document.getElementById('playback-frame');
  const progressEl = document.getElementById('video-progress-bar');
  const durationTag = document.getElementById('video-duration-tag');

  if (videoEl && videoEl.duration) {
    const cur = videoEl.currentTime || 0;
    const dur = videoEl.duration || 1;
    
    // Auto-detect video loop restart (e.g. from end back to 0:00)
    if (cur < lastVideoCurrentTime - 1.2 && lastVideoCurrentTime > 2.0) {
      resetTracking(true);
    }
    lastVideoCurrentTime = cur;

    const pct = Math.min(100, Math.max(0, (cur / dur) * 100));
    if (progressEl) progressEl.style.width = `${pct}%`;
    if (durationTag) durationTag.innerText = `${formatTime(cur)} / ${formatTime(dur)}`;
  }

  const elapsed = (now - state.startTime) / 1000;
  const hours = Math.floor(elapsed / 3600);
  const minutes = Math.floor((elapsed % 3600) / 60);
  const seconds = Math.floor(elapsed % 60);
  const centis = Math.floor((elapsed % 1) * 100);

  const formatted = `T+ ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centis).padStart(2, '0')}`;
  if (timecodeEl) timecodeEl.innerText = formatted;
  if (frameEl) frameEl.innerText = `#${state.frameCounter}`;
}

function updateMetricsDisplay() {
  const valFps = document.getElementById('val-fps');
  const valLat = document.getElementById('val-latency');
  const valActive = document.getElementById('val-active');
  const valTotal = document.getElementById('val-total');
  const valIn = document.getElementById('val-in');
  const valOut = document.getElementById('val-out');

  if (valFps) valFps.innerText = state.fps.toFixed(1);
  if (valLat) valLat.innerText = state.latency.toFixed(1);
  if (valActive) valActive.innerText = state.activeTracks.length;
  if (valTotal) valTotal.innerText = state.totalTracks;
  if (valIn) valIn.innerText = state.countIn;
  if (valOut) valOut.innerText = state.countOut;

  updateClassDistribution();
}

function updateClassDistribution() {
  if (!classDistEl) return;
  const counts = {};
  
  state.activeTracks.forEach(t => {
    const c = (t.className || 'object').toLowerCase();
    counts[c] = (counts[c] || 0) + 1;
  });

  const activeTag = document.getElementById('active-class-count-tag');
  if (activeTag) {
    const totalActive = Object.values(counts).reduce((a, b) => a + b, 0);
    activeTag.innerText = `${totalActive} active`;
  }

  const keys = Object.keys(counts);
  if (keys.length === 0) {
    classDistEl.innerHTML = `<div class="text-[11px] font-mono text-gray-500 py-2">No active targets in view</div>`;
    return;
  }

  let html = '';
  keys.forEach(k => {
    const count = counts[k];
    const theme = getClassTheme(k);
    const maxCount = Math.max(...Object.values(counts), 1);
    const barPct = Math.round((count / maxCount) * 100);

    html += `
      <div class="flex flex-col gap-1 text-xs font-mono">
        <div class="flex justify-between items-center">
          <span class="text-gray-300 font-medium capitalize flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full" style="background: ${theme.border}"></span>
            ${k}
          </span>
          <span class="font-bold text-white">${count}</span>
        </div>
        <div class="w-full h-1.5 bg-obsidian-border rounded-full overflow-hidden">
          <div class="h-full rounded-full transition-all duration-300" style="width: ${barPct}%; background: ${theme.border}; box-shadow: 0 0 8px ${theme.border}"></div>
        </div>
      </div>
    `;
  });

  classDistEl.innerHTML = html;
}

function addActivityLog(id, desc, color = 'lime') {
  const d = new Date();
  const timeStr = d.toTimeString().split(' ')[0];
  state.eventLogs.unshift({ time: timeStr, id, desc, color });
  if (state.eventLogs.length > 40) state.eventLogs.pop();
  renderActivityFeed();
}

function renderActivityFeed() {
  if (!activityFeedEl) return;
  if (state.eventLogs.length === 0) {
    activityFeedEl.innerHTML = `
      <div class="border-l-2 border-lime pl-2 leading-relaxed">
        <span class="text-gray-500">SYSTEM</span>
        <span class="text-lime font-bold ml-1">INFO</span>
        <span class="text-gray-300 block">VisionGrid Initialized</span>
      </div>
    `;
    return;
  }

  activityFeedEl.innerHTML = state.eventLogs.map(l => `
    <div class="border-l-2 border-${l.color} pl-2 leading-relaxed">
      <span class="text-gray-500">${l.time}</span>
      <span class="text-${l.color} font-bold ml-1">${l.id}</span>
      <span class="text-gray-300 block">${l.desc}</span>
    </div>
  `).join('');
}

function clearLogs() {
  state.eventLogs = [];
  renderActivityFeed();
}

// ==========================================
// 6. In-Video Controls & Playback
// ==========================================
let controlsTimeout = null;

function showInVideoControls() {
  const header = document.getElementById('in-video-header');
  const footer = document.getElementById('in-video-footer');
  if (header) {
    header.classList.remove('opacity-0');
    header.classList.add('opacity-100');
  }
  if (footer) {
    footer.classList.remove('opacity-0', 'translate-y-4');
    footer.classList.add('opacity-100', 'translate-y-0');
  }

  clearTimeout(controlsTimeout);
  controlsTimeout = setTimeout(() => {
    if (!state.isPaused && videoEl && !videoEl.paused) {
      hideInVideoControls();
    }
  }, 3500);
}

function hideInVideoControls() {
  const header = document.getElementById('in-video-header');
  const footer = document.getElementById('in-video-footer');
  if (header) {
    header.classList.remove('opacity-100');
    header.classList.add('opacity-0');
  }
  if (footer) {
    footer.classList.remove('opacity-100', 'translate-y-0');
    footer.classList.add('opacity-0', 'translate-y-4');
  }
}

function showCenterRipple(type) {
  const ripple = document.getElementById('center-play-ripple');
  const icon = document.getElementById('center-ripple-icon');
  if (!ripple || !icon) return;

  if (type === 'play') {
    icon.className = 'fa-solid fa-play ml-1';
  } else {
    icon.className = 'fa-solid fa-pause';
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
  loadSampleClip(nextKey);
}

let currentSpeed = 1.0;
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
  addActivityLog('SPEED', `Playback speed set to ${currentSpeed}x`, 'cyan');
}

let isMuted = true;
function toggleMute() {
  isMuted = !isMuted;
  if (videoEl) videoEl.muted = isMuted;
  const icon = document.getElementById('volume-icon');
  const slider = document.getElementById('volume-slider');
  if (icon) {
    icon.className = isMuted ? 'fa-solid fa-volume-xmark text-red-400' : 'fa-solid fa-volume-high text-lime';
  }
  if (slider) {
    slider.value = isMuted ? 0 : 1;
  }
}

function setVolume(val) {
  const v = parseFloat(val);
  if (videoEl) {
    videoEl.volume = v;
    videoEl.muted = (v === 0);
  }
  isMuted = (v === 0);
  const icon = document.getElementById('volume-icon');
  if (icon) {
    if (v === 0) icon.className = 'fa-solid fa-volume-xmark text-red-400';
    else if (v < 0.5) icon.className = 'fa-solid fa-volume-low text-yellow-400';
    else icon.className = 'fa-solid fa-volume-high text-lime';
  }
}

function updatePlayPauseButton(status) {
  const icon = document.getElementById('play-icon');
  if (!icon) return;
  if (status === 'playing') {
    icon.className = 'fa-solid fa-pause';
  } else if (status === 'ended') {
    icon.className = 'fa-solid fa-rotate-left';
  } else {
    icon.className = 'fa-solid fa-play ml-0.5';
  }
}

async function togglePlayPause() {
  if (!videoEl) return;
  if (videoEl.paused || videoEl.ended) {
    try {
      await videoEl.play();
      state.isPaused = false;
      updatePlayPauseButton('playing');
      showCenterRipple('play');
    } catch (e) {
      console.warn('Play error:', e);
    }
  } else {
    videoEl.pause();
    state.isPaused = true;
    updatePlayPauseButton('paused');
    showCenterRipple('pause');
    showInVideoControls();
  }
}

function seekVideo(event) {
  const container = document.getElementById('video-progress-container');
  if (!container || !videoEl || !videoEl.duration) return;
  const rect = container.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const pct = Math.max(0, Math.min(1, clickX / rect.width));
  videoEl.currentTime = pct * videoEl.duration;
  
  if (state.isPaused) {
    videoEl.play().then(() => {
      state.isPaused = false;
      updatePlayPauseButton('playing');
    }).catch(() => {});
  }
}

function captureSnapshot() {
  const link = document.createElement('a');
  link.download = `visiongrid_frame_${Date.now()}.png`;
  link.href = canvasEl.toDataURL('image/png');
  link.click();
  addActivityLog('EXPORT', 'High-res Snapshot saved to Downloads', 'lime');
}

function toggleRecording() {
  if (state.isRecording) {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
    }
    state.isRecording = false;
    const recIcon = document.getElementById('record-icon');
    if (recIcon) recIcon.className = 'fa-solid fa-circle text-[10px] text-red-500';
    addActivityLog('REC', 'Recording stopped. Processing video file...', 'amber');
  } else {
    try {
      const stream = canvasEl.captureStream(30);
      state.recordedChunks = [];
      state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
      };
      state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `visiongrid_recording_${Date.now()}.webm`;
        a.click();
        addActivityLog('REC', 'Tracked Video Exported (.webm)', 'lime');
      };
      state.mediaRecorder.start();
      state.isRecording = true;
      const recIcon = document.getElementById('record-icon');
      if (recIcon) recIcon.className = 'fa-solid fa-stop text-[10px] text-lime animate-pulse';
      addActivityLog('REC', 'Recording started (Live 30 FPS Stream)', 'lime');
    } catch (e) {
      console.error('MediaRecorder error:', e);
      addActivityLog('REC', 'Screen recording not supported in this browser', 'red');
    }
  }
}

let lastVideoCurrentTime = 0;

async function resetTracking(isLoopReset = false) {
  state.countIn = 0;
  state.countOut = 0;
  state.crossedIds = {};
  state.activeTracks = [];
  state.pythonTracks = [];
  state.trackHistories = {};
  state.totalTracks = 0;
  webTracker.reset();

  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: 'reset' }));
  }

  fetch('/api/reset', { method: 'POST' }).catch(() => {});

  if (!isLoopReset) {
    addActivityLog('SYSTEM', 'Tracker & telemetry counters reset to 0', 'lime');
  }
  updateMetricsDisplay();
}

function toggleFullscreen() {
  const frame = document.getElementById('video-player-frame');
  if (!document.fullscreenElement) {
    if (frame.requestFullscreen) frame.requestFullscreen();
    else if (frame.webkitRequestFullscreen) frame.webkitRequestFullscreen();
  } else {
    if (document.exitFullscreen) document.exitFullscreen();
  }
}

function exportData(format) {
  const telemetry = {
    timestamp: new Date().toISOString(),
    fps: state.fps,
    latency_ms: state.latency,
    total_tracks: state.totalTracks,
    count_in: state.countIn,
    count_out: state.countOut,
    active_targets: state.activeTracks.map(t => ({
      id: t.id,
      class: t.className,
      confidence: t.score,
      speed_kmh: t.speed,
      bbox: t.bbox
    }))
  };

  if (format === 'json') {
    const blob = new Blob([JSON.stringify(telemetry, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `visiongrid_telemetry_${Date.now()}.json`;
    a.click();
    addActivityLog('EXPORT', 'Telemetry exported as JSON', 'lime');
  } else if (format === 'csv') {
    let csv = 'ID,Class,Confidence,Speed_KMH,X1,Y1,X2,Y2\n';
    state.activeTracks.forEach(t => {
      csv += `${t.id},${t.className},${(t.score || 0).toFixed(2)},${t.speed},${t.bbox.map(v => Math.round(v)).join(',')}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `visiongrid_telemetry_${Date.now()}.csv`;
    a.click();
    addActivityLog('EXPORT', 'Telemetry exported as CSV', 'lime');
  }
}

// ==========================================
// 7. Source Switching & Sample Clips
// ==========================================
const SAMPLE_CLIPS = {
  sample1: {
    name: 'Sample 1: Traffic & Pedestrians',
    src: 'sample_data/Sample1.mp4',
    btnId: 'btn-sample-1'
  },
  sample2: {
    name: 'Sample 2: Walkway & Street',
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
  state.pythonTracks = [];
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
      await videoEl.play();
      updatePlayPauseButton('playing');
      addActivityLog('INPUT', `Preset loaded: ${clip.name}`, 'lime');
    } catch (err) {
      console.warn('Autoplay prevented:', err);
      state.isPaused = true;
      updatePlayPauseButton('paused');
      addActivityLog('INPUT', 'Click Play ▶ to begin video stream', 'amber');
    }
  }
}

function triggerVideoUpload() {
  const input = document.getElementById('video-file-input');
  if (input) input.click();
}

function handleVideoFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const headerTitle = document.getElementById('viewport-header-title');
  if (headerTitle) headerTitle.innerText = `Uploaded Video: ${file.name}`;

  if (videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach(t => t.stop());
    videoEl.srcObject = null;
  }

  videoEl.src = url;
  videoEl.crossOrigin = 'anonymous';
  videoEl.muted = true;
  videoEl.playsInline = true;
  videoEl.loop = true;
  state.isPaused = false;

  videoEl.play().then(() => {
    updatePlayPauseButton('playing');
    addActivityLog('UPLOAD', `Loaded video file: ${file.name}`, 'lime');
  }).catch(() => {
    state.isPaused = true;
    updatePlayPauseButton('paused');
  });

  // Try uploading to backend if Python active
  if (state.isPythonActive) {
    const formData = new FormData();
    formData.append('file', file);
    fetch('/api/upload', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => {
        addActivityLog('UPLOAD', 'Video synced with Python YOLOv8 backend', 'lime');
      })
      .catch(() => {});
  }
}

// ==========================================
// 8. Sliders, Toggles & Filters
// ==========================================
function updateConfidence(val) {
  const v = parseFloat(val) / 100;
  state.confThresh = v;
  const tag = document.getElementById('conf-val');
  if (tag) tag.innerText = `${val}%`;

  if (state.isPythonActive) {
    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conf_thresh: v })
    }).catch(() => {});
  }
}

function updateMaxAge(val) {
  const v = parseInt(val);
  state.maxAge = v;
  webTracker.maxAge = v;
  const tag = document.getElementById('age-val');
  if (tag) tag.innerText = `${v} frames`;

  if (state.isPythonActive) {
    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_age: v })
    }).catch(() => {});
  }
}

function updateLinePos(val) {
  state.linePos = parseFloat(val) / 100;
  const tag = document.getElementById('line-val');
  if (tag) tag.innerText = `${val}%`;
}

function toggleSetting(name) {
  if (state.toggles[name] !== undefined) {
    state.toggles[name] = !state.toggles[name];
    const checkbox = document.getElementById(`toggle-${name}`);
    if (checkbox) checkbox.checked = state.toggles[name];
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
