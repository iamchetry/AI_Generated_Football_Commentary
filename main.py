import numpy as np
from collections import deque
import time

from utils.frame_extraction import *
from utils.detection import *
from utils.object_tracking import *
from utils.jersey_detection import *
from utils.event_detection import *
from utils.commentary_generation import *
from utils.text_to_speech_conversion import *
from utils.video_generation import *


def call_all(video_path, yolov8_model, player_class_id, ball_class_id, annonated_folder=None, show_frame=False, conf_threshold_player=None, conf_threshold_ball=None, ocr_conf_threshold=None, tracked_folder=None, overlay_jersey_info=False, debug_dir=None, maxlen=None, output_audio_path=None, speedx=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")
    ball_manager = BallTracker()
#     goal_detector = GoalDetector(
#     goal_area_left=(300, 600, 350, 1300),           
#     goal_area_right=(1250, 600, 1300, 1300)
# )
    pass_detector = PassDetector(distance_threshold=300)

    c = 0
    jersey_info = dict()
    pass_events = list()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unexpected end of video.")
            break

        player_detections, annotated_frame, ball_box = detect_objects(frame, player_class_id, ball_class_id, yolov8_model=yolov8_model, show_frame=show_frame, conf_threshold_player=conf_threshold_player, conf_threshold_ball=conf_threshold_ball)
        t1 = time.time()

        if annonated_folder:
            print('annotated')
            cv2.imwrite(annonated_folder+'frame_{}.jpg'.format(c), annotated_frame)
        t2 = time.time()
        print('t2 - t1 : {}'.format(int(t2-t1)))

        tracked_players = track_objects(frame, player_detections)
        t3 = time.time()
        print('t3 - t2 : {}'.format(int(t3-t2)))

        jersey_info = extract_jersey_numbers(frame, tracked_players, jersey_info, ocr_conf_threshold, frame_id=c, debug_dir=debug_dir)
        t4 = time.time()
        print('t4 - t3 : {}'.format(int(t4-t3)))
        
        ball_center = ball_manager.update(ball_box)
        t5 = time.time()

        # === NEW: Detect pass event ===
        pass_event = pass_detector.update(ball_center, tracked_players)

        if pass_event:
            from_id = pass_event['from_id']
            to_id = pass_event['to_id']
            from_name = jersey_info.get(from_id, f'Player {from_id}')
            to_name = jersey_info.get(to_id, f'Player {to_id}')
            
            # Pass this to LLM if needed:
            print(f"PASS: {from_name} (ID {from_id}) ==> {to_name} (ID {to_id}) in frame {c}")
            frame = draw_passes(frame, from_name, to_name)
            prompt_ = build_pass_prompt(from_id, to_id)
            commentary_ = llm_generate_commentary(prompt_)
            print(commentary_)
            tts_elevanlabs(commentary_, output_audio_path+f'commentary_frame_{c}.mp3')
            speed_up_audio(output_audio_path+f'commentary_frame_{c}.mp3', output_audio_path+f'commentary_frame_{c}_{speedx}x.mp3', speed=speedx)
            pass_events.append({'frame': c, 'audio': output_audio_path+f'commentary_frame_{c}_{speedx}x.mp3'})
            print('------------ pass_events -------------')
            print(pass_events)

        print('t5 - t4 : {}'.format(int(t5-t4)))

        # scorer_id = goal_detector.update(ball_center, tracked_players)
        # if scorer_id:
        #     name = jersey_info.get(scorer_id, f'Player {scorer_id}')
        #     print(f'GOAL by {name}, {scorer_id}!')

        frame = draw_player_tracks(frame, tracked_players)
        t6 = time.time()
        print('t6 - t5 : {}'.format(int(t6-t5)))

        if overlay_jersey_info:
            frame = overlay_jersey_numbers(frame, tracked_players, jersey_info)
            t7 = time.time()
            print('t7 - t6 : {}'.format(int(t7-t6)))
        
        t8 = time.time()
        if ball_center:
            frame = draw_ball_tracks(frame, ball_center)
            print('t8 - t7 : {}'.format(int(t8-t7)))

        # frame = draw_goal_areas(frame, goal_detector.goal_area_left, goal_detector.goal_area_right)

        if tracked_folder:
            print('tracked')
            cv2.imwrite(tracked_folder+'frame_{}.jpg'.format(c), frame)
            t9 = time.time()
            print('t9 - t8 : {}'.format(int(t9-t8)))

        print('------------------ {} ------------------'.format(c))
            # print(frame)
            # print('====================================')
            # print(detections)
            # print('====================================')
            # print(tracked_objects)
            # print('====================================')
        c += 1
        print('============= pass_events =============')
        print(pass_events)


if __name__ == '__main__':
    # video_path = 'data/high_res_videos/salz_vs_rm.mp4'
    # frame_folder = 'output/frames/'
    # start_end_list = [55, 65]
    # cropped_video_path = 'data/cropped_videos/video_3.mp4'
    # frames = extract_frames(video_path, frame_folder, start_end_list=start_end_list, cropped_video_path=cropped_video_path)
    # print(frames)

    video_path = 'data/cropped_videos/video_3.mp4'
    yolov8_model = 'yolov8x'
    player_class_id = 0
    ball_class_id = 32
    # annonated_folder = 'output/annotated_frames_{}/'.format(yolov8_model)
    annonated_folder = None

    show_frame = False
    conf_threshold_player = 0.4
    conf_threshold_ball = 0.25
    ocr_conf_threshold = 0.3
    # tracked_folder = 'output/tracked_frames_{}/'.format(yolov8_model)
    tracked_folder = None

    overlay_jersey_info = True
    # debug_dir = 'output/debug_dir_{}/'.format(yolov8_model)
    debug_dir = None

    maxlen = 20
    output_audio_path = 'output/audio_files_elevanlabs/'
    speedx = 1.85

    print(ocr_conf_threshold)

    call_all(video_path, yolov8_model, player_class_id, ball_class_id, annonated_folder=annonated_folder, show_frame=show_frame, conf_threshold_player=conf_threshold_player, conf_threshold_ball=conf_threshold_ball, ocr_conf_threshold=ocr_conf_threshold, tracked_folder=tracked_folder, overlay_jersey_info=overlay_jersey_info, debug_dir=debug_dir, maxlen=maxlen, output_audio_path=output_audio_path, speedx=speedx)

    output_video_path = 'output/video_files/final_video.mp4'
    overlay_audio(video_path, output_video_path)
