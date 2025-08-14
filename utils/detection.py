from ultralytics import YOLO
import cv2


def select_ball_closest_to_horizontal_center(ball_detections, frame_height):
    if not ball_detections:
        return None

    center_y = frame_height/2

    sorted_balls = sorted(
        ball_detections,
        key=lambda b: abs(((b[1] + b[3]) / 2) - center_y)  # vertical distance from center line
    )

    return sorted_balls[0]


# def detect_objects_v1(frame, player_class_id, ball_class_id, yolov8_model=None, show_frame=False, conf_threshold_player=None, conf_threshold_ball=None):
#     model = YOLO(yolov8_model)
#     results = model(frame)
    
#     annotated_frame = results[0].plot()

#     if show_frame:
#         cv2.imshow("YOLOv8 Detections", annotated_frame)
    
#     player_detections = list()
#     ball_boxes = list()

#     for box in results[0].boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         conf = float(box.conf[0])
#         cls_id = int(box.cls[0])

#         if cls_id == player_class_id and conf >= conf_threshold_player:
#             player_detections.append([[x1, y1, x2-x1, y2-y1], conf, cls_id])
#         elif cls_id == ball_class_id and conf >= conf_threshold_ball:
#             ball_boxes.append([x1, y1, x2, y2])
#         else:
#             continue

#     return (player_detections, annotated_frame, ball_boxes)


def detect_objects(frame, player_class_id, ball_class_id, yolov8_model=None, show_frame=False, conf_threshold_player=None, conf_threshold_ball=None):
    model = YOLO(yolov8_model)
    results = model(frame)

    annotated_frame = results[0].plot()

    if show_frame:
        cv2.imshow("YOLOv8 Detections", annotated_frame)
    
    player_detections = list()
    ball_candidates = list()  # collect all eligible balls first

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if cls_id == player_class_id and conf >= conf_threshold_player:
            player_detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])
        
        elif cls_id == ball_class_id:
            ball_candidates.append([x1, y1, x2, y2])  # (no conf needed here but can be added)
        
        else:
            continue

    # Choose the best ball (closest to center-y)
    frame_height = frame.shape[0]
    ball_box = select_ball_closest_to_horizontal_center(ball_candidates, frame_height)

    return (player_detections, annotated_frame, ball_box)
