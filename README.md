# Signal: Real-Time ASL Gesture Recognition

**Signal** is an AI-powered, vision-based application designed to bridge communication gaps by **translating American Sign Language (ASL) gestures into text and speech in real-time**. Using a standard webcam, the system detects hand landmarks and classifies signs instantaneously, offering an accessible tool for seamless interaction between signers and non-signers.

## 🚀 Features
* **Real-Time Detection:** Low-latency hand tracking and gesture classification via standard webcams.
* **High Accuracy:** Powered by state-of-the-art deep learning architectures trained on diverse ASL datasets.
* **Text-to-Speech Integration:** Automatically converts recognized gestures into spoken words for enhanced accessibility.
* **Lightweight & Efficient:** Optimized to run locally on standard hardware without requiring heavy GPU computing.

## 🛠️ Tech Stack
* **Frameworks & Libraries:** OpenCV, MediaPipe, TensorFlow / PyTorch
* **Language:** Python
* **GUI / Deployment (Optional):** Streamlit / Tkinter / Flask

## 📋 System Architecture
1. **Video Capture:** Streams live frames from the user's webcam.
2. **Hand Landmark Detection:** Uses MediaPipe Hands to extract 21 3D hand joint coordinates.
3. **Feature Extraction:** Preprocesses coordinates to normalize scale and rotation variances.
4. **Classification Model:** A trained Neural Network (LSTM / CNN / Random Forest) predicts the corresponding ASL alphabet or word.
5. **Output Delivery:** Displays the text on-screen and triggers a Text-to-Speech (TTS) engine.

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd signal---real-time-asl-gesture-recognition
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

## 📈 Future Roadmap
* Expand the dataset to include full sentences and dynamic phrases instead of single letters.
* Implement bidirectional translation (Text-to-ASL via avatar animation).
* Deploy as a mobile application for on-the-go accessibility.
