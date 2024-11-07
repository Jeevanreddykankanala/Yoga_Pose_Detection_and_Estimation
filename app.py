from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__, static_folder='static')

# Define the path to your images
IMAGE_FOLDER = os.path.join(app.static_folder, 'images')
if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError(f"Image folder '{IMAGE_FOLDER}' does not exist.")

images = sorted(os.listdir(IMAGE_FOLDER))
if not images:
    raise FileNotFoundError("No images found in the image folder.")

current_image_index = 0

class MediapipePose:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame, results.pose_landmarks

    def get_landmarks(self, landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return None

def compare_poses(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return False
    distances = [np.linalg.norm(np.array(l1) - np.array(l2)) for l1, l2 in zip(landmarks1, landmarks2)]
    return np.mean(distances) < 0.3

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise ValueError("Unable to access the webcam.")
        self.pose_detector = MediapipePose()

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frames(self):
        global current_image_index

        reference_image_path = os.path.join(IMAGE_FOLDER, images[current_image_index])
        reference_image = cv2.imread(reference_image_path)
        reference_frame, reference_results = self.pose_detector.process_frame(reference_image)
        reference_landmarks = self.pose_detector.get_landmarks(reference_results)

        while True:
            success, frame = self.video.read()
            if not success:
                continue

            frame, results = self.pose_detector.process_frame(frame)
            camera_landmarks = self.pose_detector.get_landmarks(results)

            if compare_poses(camera_landmarks, reference_landmarks):
                status_message = 'Pose is correct.'
                color = (0, 255, 0)  # Green
            else:
                status_message = 'Pose is not correct.'
                color = (0, 0, 255)  # Red

            cv2.putText(frame, status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    global current_image_index
    image_filename = images[current_image_index]
    return render_template('index.html', image_filename=image_filename)

@app.route('/video_feed')
def video_feed():
    return Response(Camera().get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next')
def next_pose():
    global current_image_index
    if current_image_index < len(images) - 1:
        current_image_index += 1
    return redirect(url_for('index'))

@app.route('/previous')
def previous_pose():
    global current_image_index
    if current_image_index > 0:
        current_image_index -= 1
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
