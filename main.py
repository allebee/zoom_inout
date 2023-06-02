import base64
import io
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)

# Initialize face detector and shape predictor outside of the video_feed function
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model_needed _zoom.dat')

# Set parameters for oval detection
width = 300
height = 400
oval_center = (320, 240)  # Default center for 640x480 camera resolution
inner_center = oval_center
axes = (width//2, height//2)
inner_axes = (int(width//3), int(height//3))
text = "Please zoom in"
count = 0


@app.route('/video_feed', methods=['POST'])
def video_feed():
    global count, width, height, text, oval_center, inner_center, axes, inner_axes

    frame_data = request.data
    _, frame_data = frame_data.decode().split(',', 1)
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    face_in_oval = False
    for face in faces:
        # Get the facial landmarks for the current face
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        landmark_points = landmarks[:3]
        imagePoints = landmarks[16:18]
        for (x, y) in landmark_points:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        distances_inner = ((landmark_points[:, 0]-inner_center[0])/inner_axes[0])**2 + (
            (landmark_points[:, 1]-inner_center[1])/inner_axes[1])**2

        distances_outer = ((landmark_points[:, 0]-oval_center[0])/axes[0])**2 + (
            (landmark_points[:, 1]-oval_center[1])/axes[1])**2

        if np.all(distances_inner >= 1) and np.all(distances_outer <= 1):
            face_in_oval = True
    # Image processing logic
    # ...
    if face_in_oval:
        count += 1
        cv2.ellipse(frame, oval_center, axes,
                    0, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(frame, inner_center,
                    inner_axes, 0, 0, 360, (0, 255, 0), 2)
    else:
        cv2.ellipse(frame, oval_center, axes,
                    0, 0, 360, (255, 0, 0), 2)
        cv2.ellipse(frame, inner_center,
                    inner_axes, 0, 0, 360, (255, 0, 0), 2)

    if count == 50:
        print("count is enough")
        width //= 1.5
        height //= 1.5
        width = int(width)
        height = int(height)
        # Update oval_center based on new width and height
        oval_center = (320, 240)
        inner_center = oval_center  # Update inner_center based on new oval_center
        # Update axes based on new width and height
        axes = (width//2, height//2)
        # Update inner_axes based on new width and height
        inner_axes = (int(width//3), int(height//3))
        text = "Please zoom out"
        count += 1
    elif count == 100:
        text = "You showed your liveness successfully"

    # Draw the overlay and text on the frame
    # ...
    coordinates = (25, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(frame, text, coordinates, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    if image is not None:
        frame_bw = frame
        _, frame_bw_encoded = cv2.imencode('.jpg', frame_bw)
        frame_bw_bytes = frame_bw_encoded.tobytes()
        frame_bw_io = io.BytesIO(frame_bw_bytes)

        return Response(response=frame_bw_io.getvalue(), status=200, mimetype='image/jpeg')
    else:
        return Response(response="Frame is empty", status=400)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
