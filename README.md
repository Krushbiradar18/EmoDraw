
# 🖌️ EmoDraw: Real-Time Air Drawing & Shape Recognition with AI<br>

EmoDraw is a fun and experimental AI project that lets you draw shapes in the air using your finger — tracked via webcam — and predicts what you've drawn using a trained Convolutional Neural Network (CNN). 🎯<br>

https://github.com/krushbiradar18/EmoDraw<br>

---

## 🚀 Features<br>

- ✋ Finger tracking using **MediaPipe + OpenCV**<br>
- 🎨 Draw in the air by moving your finger — real-time canvas overlay<br>
- 🧠 Predicts the shape using a trained **CNN model**<br>
- 💾 Save your drawings and collect custom datasets<br>
- 🧽 Eraser mode to fix parts of the sketch<br>
- 🔍 Class prediction with confidence score<br>
- 📷 Works in real-time with webcam<br>

---

## 🧠 Built With<br>

- `Python`<br>
- `OpenCV`<br>
- `MediaPipe`<br>
- `TensorFlow / Keras`<br>
- `NumPy`<br>

---

## 📁 Folder Structure<br>

EmoDraw/
├── hand_draw.py              # Real-time drawing & prediction script
├── train_emodraw_cnn.py      # CNN training script
├── dataset/                  # Folder where labeled drawings are stored
├── emodraw_cnn_model.h5      # Trained CNN model
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions<br>

1. **Clone the repo**<br>
   `git clone https://github.com/krushbiradar18/EmoDraw.git`<br>
   `cd EmoDraw`<br>

2. **Set up a virtual environment**<br>
   `python3 -m venv venv`<br>
   `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)<br>

3. **Install dependencies**<br>
   `pip install -r requirements.txt`<br>

4. **Run the air drawing app**<br>
   `python hand_draw.py`<br>

---

## 🖼️ How It Works<br>

- The webcam captures your hand.<br>
- MediaPipe tracks your **index fingertip**.<br>
- You can toggle between:<br>
  - `D` → Start/stop drawing<br>
  - `E` → Eraser mode<br>
  - `S` → Save canvas<br>
  - `P` → Predict the shape<br>
  - `C` → Clear the canvas<br>
  - `Q` → Quit<br>

---

## 🤖 Training Your Own Model<br>

You can collect your own dataset using the drawing tool:<br>

`python hand_draw.py`<br>

1. Toggle draw mode (`D`)<br>
2. Draw a shape in the air<br>
3. Press `S` to save it (label is chosen at startup)<br>
4. Repeat for other shapes<br>
5. Once done, train using:<br>
   `python train_emodraw_cnn.py`<br>

---

## 📌 Example Classes<br>

```python
class_names = ['star', 'sun', 'tree', 'smiley_face', 'flower', 'heart']

You can change or expand these by modifying your dataset folder structure.

⸻

🙋‍♀️ Author

Krushnali Biradar
GitHub | LinkedIn

⸻

💡 Future Ideas
	•	Add gesture-based mode switching (no keyboard needed)
	•	Integrate sound or emoji overlay for detected shapes
	•	Deploy as a Streamlit or Gradio app

⸻

🪄 License

This project is open-source and free to use under the MIT License.

