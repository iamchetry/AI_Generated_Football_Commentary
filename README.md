# AI_Generated_Football_Commentary
This codebase creates AI-Generated Commentary from a live Football video, by following the below steps -

1. Extracts each frame of the video.
2. Detects (1) Players & (2) Ball in each frame by using YOLO.
3. Tracks (1) Identity & (2) Movement of each player across subsequent frames using DeepSORT.
4. Tracks movement of the ball across subsequent frames using Kalman filter.
5. Detects jersey numbers of the players using OCR.
6. Detects the Event (Pass/Goal). As of now, the code can only detect passes & not goals.
7. Generates a prompt once the pass it detected.
8. Passes this prompt as an input to the LLM to generate an interesting Football commentary text.
9. Converts the LLM-generated English text into a human voice audio using Elevanlabs.
10. Overlays the human audio with the original Football video.
