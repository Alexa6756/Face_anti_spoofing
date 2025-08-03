# 🛡️ Face Anti-Spoofing System

This project implements a **real-time face anti-spoofing system** using Convolutional Neural Networks (CNNs). It detects whether a face is real or spoofed using RGB, depth, and IR data.  

##  Features
- Real-time spoof detection using webcam input.
- CNN-based architecture for robust classification.
- Grad-CAM visualizations for model interpretability.
- Achieved **98.2% accuracy** and **97.5% F1-score** on the validation dataset.

---

## **Project Structure**

```
Face_anti_spoofing/
├── data/                  # Dataset folder (Training, Val, Testing) [Not uploaded to GitHub]
│
├── src/                   # All source code and models
│   ├── dataset_loader.py  # Data loading pipeline
│   ├── train.py           # Model training & validation
│   ├── model.py           # CNN model architecture
│   ├── realtimetest.py    # Real-time webcam testing or dataset evaluation
│   ├── gradcam.py         # Grad-CAM heatmap generation
│   ├── sample.png         # Sample Grad-CAM output
│   ├── dataset_clean.csv  # Cleaned dataset paths & labels
│   └── models/            # Saved models (.pth)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/face_anti_spoofing.git
   cd face_anti_spoofing/src
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add the **data folder** (with `Training`, `Val`, and `Testing`) to the project root (not included in GitHub).

---

## **How to Run**

### 1️⃣ Train the model

```bash
python train.py
```

This will train the CNN and save the model as `models/anti_spoof_model.pth`.

---

### 2️⃣ Evaluate or run real-time webcam testing

* To run real-time webcam detection:

  ```bash
  python realtimetest.py
  ```
* To evaluate on validation dataset:

  ```bash
  python realtimetest.py --mode val
  ```

---

### 3️⃣ Generate Grad-CAM Heatmaps

```bash
python gradcam.py
```

This visualizes the model's decision-making for a sample image.

---

## **Notes**

* The `data` folder is **not uploaded to GitHub** (large dataset). Add it manually after cloning.
* The `models/` folder will be empty initially. Either **train the model** or place pre-trained `.pth` files inside it.

---
