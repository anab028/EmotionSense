# EmotionSense
Real-Time Facial Emotion Recognition using Deep Learning (CNN + Webcam)

# 🎭 EmotionSense – Real-Time Emotion Recognition from Webcam

EmotionSense is a real-time facial emotion recognition system that captures live webcam input, detects faces, and classifies emotional states using a custom-trained CNN.

---

## 📌 Features

- 📷 Real-time emotion prediction via webcam
- 💡 Live face detection using OpenCV and MediaPipe
- 🧠 Deep learning-based CNN model trained on FER2013
- 🛠️ Modular structure for training, inference, and preprocessing

---


---

## 🧰 Tech Stack

- Python
- PyTorch
- OpenCV
- MediaPipe
- Torchvision

---

## 🗂️ Project Structure

EmotionSense/

## 🗂️ Project Structure

| File/Folder                        | Description                        |
|-----------------------------------|------------------------------------|
| `video_model/train_video_cnn.py`  | Train CNN on video frames          |
| `video_model/webcam_emotion_predictor.py` | Live webcam prediction         |
| `video_model/model.pth`           | Trained model weights              |
| `scripts/video_preprocess.py`     | Extracts faces from video          |
| `fer2013.csv`                     | Facial expression dataset (optional) |
| `requirements.txt`                | Python packages                    |
| `README.md`                       | Project overview                   |




---

## 🏁 How to Run

### 🔹 1. Install Dependencies

```bash
pip install -r requirements.txt

python video_model/train_video_cnn.py
Emotion Categories
😠 Angry

😢 Sad

😐 Neutral

😮 Surprised

😊 Happy

😧 Fear

🤢 Disgust

CNN model trained for 10 epochs on FER2013

Input: 48x48 grayscale facial images

Output: Softmax over 7 emotion classes
