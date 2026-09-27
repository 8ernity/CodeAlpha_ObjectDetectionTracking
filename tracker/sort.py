"""
SORT: Simple Online and Realtime Tracking with a Kalman Filter
Extended with trajectory history, class-aware tracking, and velocity estimation.
Based on original paper by Alex Bewley et al.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
        + 1e-6
    )
    return o


def convert_bbox_to_z(bbox):
    w = max(1e-4, bbox[2] - bbox[0])
    h = max(1e-4, bbox[3] - bbox[1])
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None, class_id=None):
    s = max(1e-4, float(x[2, 0]))
    r = max(1e-4, float(x[3, 0]))
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = float(x[0, 0]) - w / 2.0
    y1 = float(x[1, 0]) - h / 2.0
    x2 = float(x[0, 0]) + w / 2.0
    y2 = float(x[1, 0]) + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score, class_id]).reshape((1, 6))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, class_id=0, score=1.0, max_history=30):
        self.dim_x = 7
        self.dim_z = 4

        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.R = np.eye(self.dim_z)
        self.R[2:, 2:] *= 10.0

        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = int(class_id)
        self.score = float(score)

        self.trajectory = []
        self.max_history = max_history
        self._record_center()

    def _record_center(self):
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)

    def update(self, bbox, score=None, class_id=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if score is not None:
            self.score = float(score)
        if class_id is not None:
            self.class_id = int(class_id)

        z = convert_bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        self._record_center()

    def predict(self):
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.x))
        self._record_center()
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.x)

    def get_velocity(self):
        return float(self.x[4, 0]), float(self.x[5, 0])


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


class Sort:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.total_tracks_count = 0

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
        self.total_tracks_count = 0

    def update(self, dets=np.empty((0, 6))):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        for m in matched:
            det_idx = m[0]
            trk_idx = m[1]
            det = dets[det_idx]
            self.trackers[trk_idx].update(
                det[:4],
                score=det[4] if len(det) > 4 else 1.0,
                class_id=det[5] if len(det) > 5 else 0,
            )

        for i in unmatched_dets:
            det = dets[i]
            score = det[4] if len(det) > 4 else 1.0
            class_id = det[5] if len(det) > 5 else 0
            trk = KalmanBoxTracker(det[:4], class_id=class_id, score=score)
            self.trackers.append(trk)
            self.total_tracks_count += 1

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            vx, vy = trk.get_velocity()
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.array(
                        [
                            d[0],
                            d[1],
                            d[2],
                            d[3],
                            trk.id + 1,
                            trk.class_id,
                            trk.score,
                            vx,
                            vy,
                        ]
                    )
                )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 9))

    def get_trajectories(self):
        return {
            trk.id + 1: list(trk.trajectory)
            for trk in self.trackers
            if trk.time_since_update < self.max_age
        }
