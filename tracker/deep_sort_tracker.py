"""
Advanced Tracking module: ByteTrack & Feature-based Matching extensions.
Provides two-stage association for high-recall tracking under occlusion.
"""

import numpy as np
from tracker.sort import KalmanBoxTracker, associate_detections_to_trackers, iou_batch


class ByteSortTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.1, max_age=30, min_hits=2, match_thresh=0.7):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_thresh = match_thresh
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.total_tracks_count = 0

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        KalmanBoxTracker.count = 0
        self.total_tracks_count = 0

    def update(self, dets=np.empty((0, 6))):
        self.frame_id += 1
        if len(dets) == 0:
            dets = np.empty((0, 6))

        scores = dets[:, 4] if len(dets) > 0 else np.array([])
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)

        dets_high = dets[high_mask] if len(dets) > 0 else np.empty((0, 6))
        dets_low = dets[low_mask] if len(dets) > 0 else np.empty((0, 6))

        for trk in self.tracked_stracks:
            trk.predict()

        trks = np.array([trk.get_state()[0] for trk in self.tracked_stracks]) if len(self.tracked_stracks) > 0 else np.empty((0, 4))
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_high, trks, self.match_thresh
        )

        for m in matched:
            det = dets_high[m[0]]
            self.tracked_stracks[m[1]].update(det[:4], score=det[4], class_id=det[5])

        rem_trks_indices = unmatched_trks
        if len(rem_trks_indices) > 0 and len(dets_low) > 0:
            rem_trks = np.array([self.tracked_stracks[i].get_state()[0] for i in rem_trks_indices])
            matched_low, unmatched_dets_low, unmatched_trks_low = associate_detections_to_trackers(
                dets_low, rem_trks, 0.4
            )
            for m in matched_low:
                det = dets_low[m[0]]
                orig_trk_idx = rem_trks_indices[m[1]]
                self.tracked_stracks[orig_trk_idx].update(det[:4], score=det[4], class_id=det[5])
            
            final_unmatched_trks = [rem_trks_indices[i] for i in unmatched_trks_low]
        else:
            final_unmatched_trks = list(rem_trks_indices)

        for i in unmatched_dets:
            det = dets_high[i]
            trk = KalmanBoxTracker(det[:4], class_id=det[5], score=det[4], max_history=40)
            self.tracked_stracks.append(trk)
            self.total_tracks_count += 1

        ret = []
        kept_tracks = []
        for trk in self.tracked_stracks:
            if trk.time_since_update > self.max_age:
                self.removed_stracks.append(trk)
            else:
                kept_tracks.append(trk)
                if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_id <= self.min_hits):
                    d = trk.get_state()[0]
                    vx, vy = trk.get_velocity()
                    ret.append(np.array([d[0], d[1], d[2], d[3], trk.id + 1, trk.class_id, trk.score, vx, vy]))

        self.tracked_stracks = kept_tracks

        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 9))

    def get_trajectories(self):
        return {trk.id + 1: list(trk.trajectory) for trk in self.tracked_stracks if trk.time_since_update < self.max_age}
