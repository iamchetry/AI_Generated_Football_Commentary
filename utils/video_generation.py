import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


def overlay_audio(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Example pass event metadata
    pass_events = [
        {'frame': 50, 'audio': 'output/audio_files_elevanlabs/commentary_frame_50_1.85x.mp3'}
    ]

    # Convert frames to seconds
    for event in pass_events:
        event["start_time"] = event["frame"]/fps

    audio_clips = list()

    for event in pass_events:
        audio = AudioFileClip(event["audio"])
        audio = audio.set_start(event["start_time"])
        audio_clips.append(audio)

    combined_audio = CompositeAudioClip(audio_clips)

    video = VideoFileClip(input_video_path)

    # Optionally trim audio to video duration
    final_audio = combined_audio.set_duration(video.duration)

    # Overlay audio on video
    final_video = video.set_audio(final_audio)

    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
