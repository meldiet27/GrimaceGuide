import cv2
import os
from ..utils import generate_timestamp
from ..config import BASE_DIR
from grimaceguide.database import DatabaseManager

def apply_overlay(frame, overlay):
    """
    Applies RGB or RGBA overlay (depending on alpha channels) onto BGR frame for image processing
    """
    if overlay.shape[2] == 4:
        # Overlay for proper RGBA image
        b, g, r, a = cv2.split(overlay)
        mask = a / 255.0
        for c in range(3):
            frame[:, :, c] = (1.0 - mask) * frame[:, :, c] + mask * overlay[:, :, c]
    else:
        # For RGB with no alpha channel
        # Applies 50% transparency
        overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
        frame = cv2.addWeighted(frame, 0.5, overlay_resized, 0.5, 0)
    return frame

def ensure_dict(data):
    """
    Converts JSON str data to dictionary if not
    """
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            print("Failed to decode JSON string to dict.")
            return {}
    return data

def capture_image_with_overlay():
    """Opens Webcam with cat outline overlay"""
    overlay_path = os.path.join(BASE_DIR, "imagesGUI", "cat_face_outline.png")
    overlay = None
    if os.path.exists(overlay_path):
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    cap = cv2.VideoCapture(0) #Initializing webcam
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Press SPACE to capture, ESC to cancel.")
    saved_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't capture frame. Exiting ...")
            break

        # Resize overlay to match frame
        if overlay is not None:
            overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
            frame = apply_overlay(frame, overlay_resized)

        cv2.imshow("Align your cat's face with the outline and press SPACE to capture.", frame)
        key = cv2.waitKey(1)

        if key % 256 == 32:  # SPACE key to capture image
            timestamp = generate_timestamp()
            saved_path = os.path.join(BASE_DIR, f"camera_capture_{timestamp}.jpg") #Saves image with timestamp
            cv2.imwrite(saved_path, frame)
            print(f"Image saved to {saved_path}")

            #Save image to database
            from grimaceguide.database import DatabaseManager
            db = DatabaseManager()
            filename = os.path.basename(saved_path)
            image_id = db.store_image(filename, saved_path)
            print(f"Image stored in DB with ID: {image_id}")
            break

        elif key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path

