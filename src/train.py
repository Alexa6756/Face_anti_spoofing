import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset_loader import FaceAntiSpoofDataset
from model import AntiSpoofCNN
import os

csv_path = "dataset_clean.csv"
root_dir = "/Users/alexa_kurapati/Documents/face_anti_spoofing/data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

train_dataset = FaceAntiSpoofDataset(csv_path, root_dir=root_dir, split='train')
val_dataset = FaceAntiSpoofDataset(csv_path, root_dir=root_dir, split='val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = AntiSpoofCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        if labels.eq(-1).any():
            continue
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            if labels.eq(-1).any():
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())

    if len(true) > 0:
        precision = precision_score(true, preds, zero_division=0)
        recall = recall_score(true, preds, zero_division=0)
        f1 = f1_score(true, preds, zero_division=0)
        acc = sum(p == t for p, t in zip(preds, true)) / len(true)
        print(f" Val Accuracy: {acc*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/anti_spoof_rgb_model.pth")
print(" Model saved to models/anti_spoof_rgb_model.pth")
