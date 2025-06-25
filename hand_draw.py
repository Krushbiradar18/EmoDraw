import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tensorflow as tf



# Ask user for class label (used when saving drawings)
class_label = input("Enter current class label (e.g., star, sun, tree, smiley_face, etc.): ").strip()

# Create dataset folder if not exists
dataset_path = os.path.join("dataset", class_label)
os.makedirs(dataset_path, exist_ok=True)

# Load trained CNN model
model = tf.keras.models.load_model("emodraw_cnn_model.h5")

# TODO: Replace this with your actual 6 classes in the order used during training
class_names = ['star', 'sun', 'tree', 'smiley_face', 'flower', 'home']

print("Model output shape:", model.output_shape)
print("Length of class_names:", len(class_names))

# Initialize camera and MediaPipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

canvas = None
drawing = False
eraser = False
prev_point = None  # keeps track of the previous fingertip position for smooth lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None or canvas.shape != frame.shape:
        canvas = np.zeros_like(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if lm_list:
                x, y = lm_list[8][1], lm_list[8][2]
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)

                if drawing:
                    if eraser:
                        cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                        prev_point = (x, y)  # track eraser movement too
                    else:
                        # draw continuous line from previous point to current
                        if prev_point is not None:
                            cv2.line(canvas, prev_point, (x, y), (255, 255, 255), thickness=5)
                        prev_point = (x, y)
                else:
                    prev_point = None  # lift the pen when drawing mode is off

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    mode_text = "Eraser" if eraser else "Drawing"
    cv2.putText(frame, f'Mode: {mode_text} {"ON" if drawing else "OFF"}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drawing else (0, 0, 255), 2)

    cv2.imshow("EmoDraw - Air Drawing & Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        drawing = not drawing
        prev_point = None
    elif key == ord('e'):
        eraser = not eraser
    elif key == ord('s'):
        filename = f"{class_label}_{int(time.time())}.png"
        save_path = os.path.join(dataset_path, filename)
        cv2.imwrite(save_path, canvas)
        print(f"[INFO] Drawing saved as {save_path}")
    elif key == ord('p'):
        # Preprocess canvas for prediction
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        normalized = resized.astype('float32') / 255.0
        input_img = normalized.reshape(1, 64, 64, 1)

        # Predict
        predictions = model.predict(input_img, verbose=0)
        pred_index = predictions.argmax()
        pred_label = class_names[pred_index]
        confidence = predictions[0][pred_index]

        print(f"Prediction: {pred_label} (Confidence: {confidence:.2f})")

        cv2.putText(frame, f"Prediction: {pred_label} ({confidence:.2f})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Print all class probabilities
        for i, cname in enumerate(class_names):
            print(f"{cname}: {predictions[0][i]:.3f}")
    elif key == ord('c'):
        canvas[:] = 0
        prev_point = None
        print("[INFO] Canvas cleared")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()