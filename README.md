# 🛡️ Face Anti-Spoofing System

This project implements a **real-time face anti-spoofing system** using Convolutional Neural Networks (CNNs). It detects whether a face is real or spoofed using RGB, depth, and IR data.  

##  Features
- Real-time spoof detection using webcam input.
- CNN-based architecture for robust classification.
- Grad-CAM visualizations for model interpretability.
- Achieved **98.2% accuracy** and **97.5% F1-score** on the validation dataset.

---

##  Project Structure



Face\_anti\_spoofing/
├── data/                  # Dataset folder (Training, Val, Testing)
├── src/                   # All source code and models
│   ├── dataset\_loader.py  # Data loading pipeline
│   ├── train.py           # Model training & validation
│   ├── model.py           # CNN model architecture
│   ├── realtimetest.py    # Real-time webcam testing or dataset evaluation
│   ├── gradcam.py         # Grad-CAM heatmap generation
|   |--- sample.png 
│   ├── dataset\_clean.csv  # Cleaned dataset paths & labels
│   └── models/            # Saved models (.pth)
│
├── requirements.txt
├── .gitignore
└── README.md



---

##  Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/Face_anti_spoofing.git
   cd Face_anti_spoofing


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset inside the `data/` folder (if not already present).

---

##  Training the Model

To train the CNN model:

```bash
cd src
python train.py
```

* Trained model will be saved in `src/models/anti_spoof_model.pth`.
* Validation accuracy and F1-score will be displayed after each epoch.

---

##  Real-Time Testing (Webcam)

To run live webcam-based spoof detection:

```bash
cd src
python realtimetest.py
```

Press **`q`** to quit the webcam window.

---

##  Results

* **Validation Accuracy:** 98.2%
* **F1-score:** 97.5%

---

##  Notes

* `data/` and `src/models/` are ignored in `.gitignore` to avoid pushing large files to GitHub.
* Use the same dataset structure as provided in the project for correct loading.
