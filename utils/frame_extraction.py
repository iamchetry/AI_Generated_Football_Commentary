import cv2

def extract_frames(video_path, frame_folder):
    cap = cv2.VideoCapture(video_path)
    frames = list()

    c = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv2.imwrite(frame_folder+'frame_{}.jpg'.format(c), frame)
        c += 1
    cap.release()

    return frames