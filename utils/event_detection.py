import numpy as np
import cv2


class PassDetector:
    def __init__(self, distance_threshold=300):
        self.prev_ball_center = None
        self.prev_possession_id = None
        self.distance_threshold = distance_threshold

    def find_closest_player(self, ball_center, tracks):
        min_dist = float('inf')
        closest_id = None
        for track in tracks:
            if not hasattr(track, 'to_tlbr'):
                continue  # Skip if method not available

            x1, y1, x2, y2 = track.to_tlbr()
            player_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            dist = np.linalg.norm(np.array(ball_center) - np.array(player_center))
            if dist < min_dist and dist < self.distance_threshold:
                min_dist = dist
                closest_id = track.track_id
        return closest_id

    def update(self, ball_center, tracks):
        if ball_center is None:
            return None  # Ball not detected

        curr_possession_id = self.find_closest_player(ball_center, tracks)

        pass_event = None
        if (
            self.prev_possession_id is not None and
            curr_possession_id is not None and
            self.prev_possession_id != curr_possession_id
        ):
            pass_event = {
                "from_id": self.prev_possession_id,
                "to_id": curr_possession_id
            }
            print(f"PASS: Player {self.prev_possession_id} → Player {curr_possession_id}")

        self.prev_possession_id = curr_possession_id
        self.prev_ball_center = ball_center
        return pass_event


class GoalDetector:
    def __init__(self, goal_area_left, goal_area_right, speed_threshold=5):
        self.goal_area_left = goal_area_left
        self.goal_area_right = goal_area_right
        self.prev_center = None
        self.goal_scored = False
        self.speed_threshold = speed_threshold
        self.last_scorer_id = None

    def ball_in_goal_area(self, center):
        x, y = center
        in_left = self.goal_area_left[0] <= x <= self.goal_area_left[2] and \
                  self.goal_area_left[1] <= y <= self.goal_area_left[3]
        in_right = self.goal_area_right[0] <= x <= self.goal_area_right[2] and \
                   self.goal_area_right[1] <= y <= self.goal_area_right[3]
        return in_left or in_right

    def find_closest_player(self, ball_center, tracks):
        min_dist = float('inf')
        closest_id = None

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            player_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            dist = np.linalg.norm(np.array(ball_center) - np.array(player_center))
            if dist < min_dist:
                min_dist = dist
                closest_id = track.track_id

        return closest_id

    def update(self, ball_center, tracks):
        if ball_center is None:
            self.prev_center = None
            return None  # No ball detected

        if self.prev_center is not None:
            dx = ball_center[0] - self.prev_center[0]
            dy = ball_center[1] - self.prev_center[1]
            speed = np.sqrt(dx ** 2 + dy ** 2)

            if self.ball_in_goal_area(ball_center) and speed < self.speed_threshold:
                if not self.goal_scored:
                    self.goal_scored = True
                    self.last_scorer_id = self.find_closest_player(ball_center, tracks)
                    print(f"GOAL by Player {self.last_scorer_id}!")
                    return self.last_scorer_id  # Return scorer ID

        self.prev_center = ball_center
        return None
    

def draw_passes(frame, from_name, to_name):
    cv2.putText(frame, f"PASS: {from_name} → {to_name}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


def draw_goal_areas(frame, goal_area_left, goal_area_right, color=(0, 0, 255)):
    # Draw left goal box
    x1, y1, x2, y2 = goal_area_left
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "LEFT GOAL", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw right goal box
    x1, y1, x2, y2 = goal_area_right
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "RIGHT GOAL", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame
