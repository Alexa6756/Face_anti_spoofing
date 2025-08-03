import cv2
import torch
import numpy as np
from model import AntiSpoofCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AntiSpoofCNN()
model.load_state_dict(torch.load("models/anti_spoof_rgb_model.pth", map_location=device))
model.eval().to(device)
print("✅ Model loaded for real-time webcam testing")

cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        label = torch.argmax(probs, dim=1).item()

    text = "REAL" if label == 1 else "FAKE"
    color = (0, 255, 0) if label == 1 else (0, 0, 255)
    cv2.putText(frame, f"{text} ({probs[0][label]:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Webcam Face Anti-Spoofing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
