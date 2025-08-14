import cv2

def extract_frames(video_path, frame_folder, start_end_list=None, cropped_video_path=None):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps, total_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_end_list[0]*fps)
    end_frame = int(start_end_list[1]*fps)
    print(start_frame, end_frame)

    # Ensure start and end are within range
    if start_frame >= total_frames or end_frame > total_frames:
        print("Error: Time range exceeds video length.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unexpected end of video.")
            break

        cv2.imwrite(frame_folder+'frame_{}.jpg'.format(current_frame), frame)
        out.write(frame)
        print(current_frame)
        current_frame += 1

    cap.release()
    out.release()