import re
import cv2
import easyocr

reader = easyocr.Reader(['en'])


def extract_jersey_numbers_v0(frame, tracks):
    jersey_info = {}
    h, w = frame.shape[:2]

    for track in tracks:
        if not track.is_confirmed():
            continue

        # Get bounding box and clamp it within image boundaries
        x1, y1, x2, y2 = track.to_ltrb()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Check if the crop is valid
        if x2 <= x1 or y2 <= y1:
            continue  # skip invalid boxes

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue  # skip empty crops

        # Run EasyOCR
        result = reader.readtext(cropped)
        if result:
            jersey_number = result[0][1].strip()
            if jersey_number:
                jersey_info[track.track_id] = jersey_number

    return jersey_info


def extract_jersey_numbers(frame, tracks, jersey_cache, conf_threshold, frame_id=None, debug_dir=None):
    h, w = frame.shape[:2]

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.track_id in jersey_cache:
            continue  # Already cached

        # Get and clamp bounding box
        x1, y1, x2, y2 = track.to_ltrb()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Skip invalid box
        if x2 <= x1 or y2 <= y1:
            continue

        # Apply heuristic: Only consider tall (back-facing) players
        if (y2 - y1) <= (x2 - x1):
            continue

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Step 2: Grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Step 3: CLAHE (contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Step 4: Unsharp Masking (sharpen edges)
        blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 1.0)
        sharpened = cv2.addWeighted(contrast_enhanced, 1.5, blurred, -0.5, 0)

        # Step 5: Upscale for better OCR
        processed_crop = cv2.resize(sharpened, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Run EasyOCR
        result = reader.readtext(processed_crop)

        # if result:
        #     text, confidence = result[0][1], result[0][2]
        #     jersey_number = text.strip()

        #     # Optional: confidence threshold
        #     if confidence >= conf_threshold and jersey_number:
        #         jersey_cache[track.track_id] = jersey_number

        best_number = None
        best_confidence = 0.0

        for (_, text, conf) in result:
            text = text.strip()
            if re.fullmatch(r'\d{1,2}', text) and conf >= conf_threshold:
                if conf > best_confidence:
                    best_number = text
                    best_confidence = conf

        if best_number:
            jersey_cache[track.track_id] = best_number
        else:
            print(f'OCR failed for track_id={track.track_id} at frame {frame_id} [{x1}, {y1}, {x2}, {y2}]')

            # Optionally: Save failed image crop for inspection
            if debug_dir:
                print('here in debug')
                fail_crop = frame[y1:y2, x1:x2]
                filename = f'{debug_dir}fail_track{track.track_id}_frame{frame_id}.jpg'
                cv2.imwrite(filename, fail_crop)

    return jersey_cache


def overlay_jersey_numbers(frame, tracked_objects, jersey_info):
    for track in tracked_objects:
        if track.is_confirmed() and track.track_id in jersey_info:
            x1, y1, x2, y2 = track.to_ltrb()
            jersey = jersey_info[track.track_id]
            cv2.putText(frame, f"#{jersey}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame
