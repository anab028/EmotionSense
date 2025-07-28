import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_faces_from_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

        frame_count += 1

    cap.release()

