/**
 * In-Browser SORT (Simple Online and Realtime Tracking) & Kalman Filter Engine
 * Enables 100% client-side 60FPS multi-object tracking in JavaScript.
 */

class KalmanBoxTrackerWeb {
  static count = 0;

  constructor(bbox, classId = 0, className = 'object', score = 1.0, maxHistory = 30) {
    this.id = ++KalmanBoxTrackerWeb.count;
    this.classId = classId;
    this.className = className;
    this.score = score;
    this.maxHistory = maxHistory;

    const w = Math.max(1, bbox[2] - bbox[0]);
    const h = Math.max(1, bbox[3] - bbox[1]);
    const x = bbox[0] + w / 2;
    const y = bbox[1] + h / 2;
    const s = w * h;
    const r = w / h;

    this.x = [x, y, s, r, 0, 0, 0];
    this.timeSinceUpdate = 0;
    this.hits = 1;
    this.hitStreak = 1;
    this.age = 0;

    this.trajectory = [{ x, y }];
  }

  predict() {
    this.x[0] += this.x[4];
    this.x[1] += this.x[5];
    this.x[2] = Math.max(10, this.x[2] + this.x[6]);

    this.age += 1;
    if (this.timeSinceUpdate > 0) {
      this.hitStreak = 0;
    }
    this.timeSinceUpdate += 1;

    this._recordPosition();
    return this.getState();
  }

  update(bbox, score, classId, className) {
    this.timeSinceUpdate = 0;
    this.hits += 1;
    this.hitStreak += 1;
    if (score !== undefined) this.score = score;
    if (classId !== undefined) this.classId = classId;
    if (className !== undefined) this.className = className;

    const w = Math.max(1, bbox[2] - bbox[0]);
    const h = Math.max(1, bbox[3] - bbox[1]);
    const newX = bbox[0] + w / 2;
    const newY = bbox[1] + h / 2;
    const newS = w * h;
    const newR = w / h;

    const alpha = 0.75;
    const beta = 0.25;

    const vx_meas = newX - this.x[0];
    const vy_meas = newY - this.x[1];

    this.x[4] = this.x[4] * (1 - beta) + vx_meas * beta;
    this.x[5] = this.x[5] * (1 - beta) + vy_meas * beta;

    this.x[0] = this.x[0] * (1 - alpha) + newX * alpha;
    this.x[1] = this.x[1] * (1 - alpha) + newY * alpha;
    this.x[2] = this.x[2] * (1 - alpha) + newS * alpha;
    this.x[3] = this.x[3] * (1 - alpha) + newR * alpha;

    this._recordPosition();
  }

  _recordPosition() {
    this.trajectory.push({ x: this.x[0], y: this.x[1] });
    if (this.trajectory.length > this.maxHistory) {
      this.trajectory.shift();
    }
  }

  getState() {
    const s = Math.max(1, this.x[2]);
    const r = Math.max(0.01, this.x[3]);
    const w = Math.sqrt(s * r);
    const h = s / (w || 1);
    const x1 = this.x[0] - w / 2;
    const y1 = this.x[1] - h / 2;
    const x2 = this.x[0] + w / 2;
    const y2 = this.x[1] + h / 2;
    return [x1, y1, x2, y2];
  }

  getVelocity() {
    return { vx: this.x[4], vy: this.x[5] };
  }
}

class SortTrackerWeb {
  constructor(maxAge = 25, minHits = 2, iouThreshold = 0.3) {
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.iouThreshold = iouThreshold;
    this.trackers = [];
    this.frameCount = 0;
    this.totalTracksCount = 0;
  }

  reset() {
    this.trackers = [];
    this.frameCount = 0;
    KalmanBoxTrackerWeb.count = 0;
    this.totalTracksCount = 0;
  }

  static calculateIoU(bb1, bb2) {
    const xx1 = Math.max(bb1[0], bb2[0]);
    const yy1 = Math.max(bb1[1], bb2[1]);
    const xx2 = Math.min(bb1[2], bb2[2]);
    const yy2 = Math.min(bb1[3], bb2[3]);

    const w = Math.max(0, xx2 - xx1);
    const h = Math.max(0, yy2 - yy1);
    const intersection = w * h;

    const area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]);
    const area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]);
    const union = area1 + area2 - intersection;

    return union <= 0 ? 0 : intersection / union;
  }

  update(detections = []) {
    this.frameCount++;

    for (const trk of this.trackers) {
      trk.predict();
    }

    const matches = [];
    const unmatchedDetections = [];
    const unmatchedTrackers = new Set(this.trackers.map((_, i) => i));

    if (detections.length > 0 && this.trackers.length > 0) {
      const costMatrix = [];
      for (let d = 0; d < detections.length; d++) {
        costMatrix[d] = [];
        const detBox = detections[d].bbox;
        for (let t = 0; t < this.trackers.length; t++) {
          const trkBox = this.trackers[t].getState();
          const iou = SortTrackerWeb.calculateIoU(detBox, trkBox);
          costMatrix[d][t] = iou;
        }
      }

      const usedDets = new Set();
      const usedTrks = new Set();

      while (true) {
        let maxIoU = this.iouThreshold;
        let bestD = -1;
        let bestT = -1;

        for (let d = 0; d < detections.length; d++) {
          if (usedDets.has(d)) continue;
          for (let t = 0; t < this.trackers.length; t++) {
            if (usedTrks.has(t)) continue;
            if (costMatrix[d][t] > maxIoU) {
              maxIoU = costMatrix[d][t];
              bestD = d;
              bestT = t;
            }
          }
        }

        if (bestD !== -1 && bestT !== -1) {
          matches.push({ detIdx: bestD, trkIdx: bestT });
          usedDets.add(bestD);
          usedTrks.add(bestT);
          unmatchedTrackers.delete(bestT);
        } else {
          break;
        }
      }

      for (let d = 0; d < detections.length; d++) {
        if (!usedDets.has(d)) unmatchedDetections.push(d);
      }
    } else {
      for (let d = 0; d < detections.length; d++) unmatchedDetections.push(d);
    }

    for (const match of matches) {
      const det = detections[match.detIdx];
      this.trackers[match.trkIdx].update(det.bbox, det.score, det.classId, det.className);
    }

    for (const detIdx of unmatchedDetections) {
      const det = detections[detIdx];
      const newTrk = new KalmanBoxTrackerWeb(det.bbox, det.classId, det.className, det.score);
      this.trackers.push(newTrk);
      this.totalTracksCount++;
    }

    const activeTracks = [];
    const aliveTrackers = [];

    for (const trk of this.trackers) {
      if (trk.timeSinceUpdate <= this.maxAge) {
        aliveTrackers.push(trk);
        if (trk.timeSinceUpdate < 1 && (trk.hitStreak >= this.minHits || this.frameCount <= this.minHits)) {
          const bbox = trk.getState();
          const vel = trk.getVelocity();
          activeTracks.push({
            id: trk.id,
            bbox: bbox,
            classId: trk.classId,
            className: trk.className,
            score: trk.score,
            velocity: vel,
            trajectory: [...trk.trajectory],
            hits: trk.hits,
            age: trk.age
          });
        }
      }
    }

    this.trackers = aliveTrackers;
    return activeTracks;
  }
}
