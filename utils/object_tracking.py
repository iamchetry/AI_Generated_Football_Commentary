import numpy as np
from collections import deque
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

deep_sort = DeepSort(max_age=100, n_init=3)


def track_objects(frame, detections):
    return deep_sort.update_tracks(detections, frame=frame)


def draw_player_tracks(frame, tracks):
    for track in tracks:
        # if not track.is_confirmed() or track.time_since_update > 1:
        #     continue

        # Get bounding box and track ID
        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw ID label
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def draw_ball_tracks(frame, ball_center):
    cv2.circle(frame, ball_center, 6, (0, 0, 255), -1)
    cv2.putText(frame, "BALL", (ball_center[0] + 5, ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# def draw_ball_tracks(frame, ball_tracks):
#     for ball_id, tracker in ball_tracks.items():
#         cx, cy = tracker.get_position()
#         cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
#         cv2.putText(frame, f"BALL {ball_id}", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     return frame


class BallTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # state: x, y, dx, dy | measurement: x, y
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.last_position = None

    def update(self, bbox):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self.kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            self.last_position = (int(cx), int(cy))
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])


# class BallTrackerWithMotion:
#     def __init__(self, maxlen):
#         self.kalman = cv2.KalmanFilter(4, 2)
#         self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
#                                                   [0, 1, 0, 0]], np.float32)
#         self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
#                                                  [0, 1, 0, 1],
#                                                  [0, 0, 1, 0],
#                                                  [0, 0, 0, 1]], np.float32)
#         self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
#         self.motion_history = deque(maxlen=maxlen)
#         self.last_box = None

#     def update(self, bbox):
#         self.last_box = bbox
#         x1, y1, x2, y2 = bbox
#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2
#         self.kalman.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
#         self.motion_history.append((cx, cy))

#     def predict_only(self):
#         self.kalman.predict()

#     def get_position(self):
#         prediction = self.kalman.predict()
#         return int(prediction[0]), int(prediction[1])

#     def get_total_motion(self):
#         if len(self.motion_history) < 2:
#             return 0
#         return sum(
#             np.linalg.norm(np.array(self.motion_history[i]) - np.array(self.motion_history[i - 1]))
#             for i in range(1, len(self.motion_history))
#         )
    

# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# class BallTrackManager:
#     def __init__(self, maxlen=None):
#         self.trackers = dict()
#         self.next_id = 0
#         self.maxlen = maxlen

#     def update(self, ball_detections):
#         updated = dict()
#         for box in ball_detections:
#             matched_id = self._match(box)
#             if matched_id is not None:
#                 tracker = self.trackers[matched_id]
#             else:
#                 tracker = BallTrackerWithMotion(self.maxlen)
#                 matched_id = self.next_id
#                 self.next_id += 1
#             tracker.update(box)
#             updated[matched_id] = tracker
        
#         # Keep unmatched trackers and predict for them
#         for tid, tracker in self.trackers.items():
#             if tid not in updated:
#                 tracker.predict_only()  # Predict ball position even if not matched
#                 updated[tid] = tracker

#         self.trackers = updated

#         return self.get_valid_ball()

#     def _match(self, new_box):
#         for tid, tracker in self.trackers.items():
#             if tracker.last_box is None:
#                 continue
#             iou = compute_iou(tracker.last_box, new_box)
#             if iou >= 0.2:
#                 return tid
#         return None

#     def get_valid_ball(self):
#         best_id = None
#         max_motion = -1
#         best_position = None

#         for tid, tracker in self.trackers.items():
#             motion = tracker.get_total_motion()
#             if motion > max_motion:
#                 max_motion = motion
#                 best_id = tid
#                 best_position = tracker.get_position()

#         if best_position is not None:
#             print(f"✅ Selected Ball ID {best_id} with max motion: {max_motion:.2f}")
#         return best_position
