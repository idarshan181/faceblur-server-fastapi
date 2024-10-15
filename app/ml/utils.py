import os
import cv2
from datetime import datetime

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_video(frames, output_path, fps, frame_size):
    """Save frames as a video file."""
    ensure_dir(os.path.dirname(output_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def save_result(image, filename):
    """Save result image with timestamp in specified format."""
    result_dir = "./result/predict"
    ensure_dir(result_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(result_dir, f"{timestamp}_{filename}")
    cv2.imwrite(save_path, image)
    return save_path

def blur_face(image, box, factor=30):
    """Apply Gaussian blur to the specified region of the image."""
    (startX, startY, endX, endY) = box
    face = image[startY:endY, startX:endX]
    face = cv2.GaussianBlur(face, (0, 0), factor)
    image[startY:endY, startX:endX] = face
    return image

def detect_faces_in_frame(frame, model, confidence_threshold=0.3, should_blur=False):
    """Detect faces in the frame and optionally blur them."""
    # The YOLO model returns a list of Results objects; we need to access the first one
    results = model.predict(frame)  # This returns a list of Results objects
    if not results:
        return frame

    # Access the first (and typically only) Results object
    detection_result = results[0]

    # Iterate over each detection in the result's boxes attribute
    for box in detection_result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        if score > confidence_threshold:
            coords = (int(x1), int(y1), int(x2), int(y2))
            if should_blur:
                frame = blur_face(frame, coords)
            else:
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {score:.2f}", (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame