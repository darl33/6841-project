import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from PIL import Image

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

class DeepfakeDataset(Dataset):
    def __init__(self, folder, transform=None, frame_rate=5):
        self.folder = folder
        self.transform = transform
        self.frame_rate = frame_rate
        self.samples = []
        for label in ["real", "fake"]:
            label_dir = os.path.join(folder, label)
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((file_path, 1 if label == "fake" else 0))
                elif file.lower().endswith(('.mp4', '.webm', '.avi', '.mov')):
                    frames = extract_frames(file_path, self.frame_rate)
                    for frame in frames:
                        self.samples.append((frame, 1 if label == "fake" else 0))
    
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = DeepfakeDataset('dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataset = DeepfakeDataset('dataset/validation', transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs.view(-1)))
            correct += (preds == labels).sum().item()
    
    accuracy = correct / len(validation_loader.dataset)
    print(f"Validation Accuracy: {accuracy:.4f}")

for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)

def predict_video(video_path, model, transform, frame_rate=5):
    frames = extract_frames(video_path, frame_rate)
    frames = [transform(Image.fromarray(frame)) for frame in frames]
    frames = torch.stack(frames).to(device)
    
    with torch.no_grad():
        outputs = model(frames)
        predictions = torch.sigmoid(outputs.view(-1))
        avg_prediction = predictions.mean().item()
    
    return "Fake" if avg_prediction > 0.5 else "Real"

torch.save(model.state_dict(), 'deepfake_detector.pth')
