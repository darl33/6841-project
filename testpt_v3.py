import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os

def get_video_frame_count(video_path):
    """Utility to get the total frame count of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

class DeepfakeDataset(Dataset):
    def __init__(self, folder, transform=None, frame_rate=5):
        self.transform = transform
        self.samples = []
        print(f"Loading dataset from: {folder}")
        for label in ["real", "fake"]:
            label_dir = os.path.join(folder, label)
            if not os.path.isdir(label_dir):
                print(f"Warning: Directory not found {label_dir}")
                continue

            for file in tqdm(os.listdir(label_dir), desc=f"Processing {label} files"):
                file_path = os.path.join(label_dir, file)
                file_lower = file.lower()

                # Assign numeric label
                numeric_label = 1 if label == "fake" else 0

                if file_lower.endswith(('.png', '.jpg', '.jpeg')):
                    # For images, the frame index is -1 (sentinel value)
                    self.samples.append((file_path, -1, numeric_label))
                elif file_lower.endswith(('.mp4', '.webm', '.avi', '.mov')):
                    # For videos, add each frame as a sample
                    total_frames = get_video_frame_count(file_path)
                    for frame_idx in range(0, total_frames, frame_rate):
                        self.samples.append((file_path, frame_idx, numeric_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame_idx, label = self.samples[idx]

        if frame_idx == -1: # It's an image file
            frame = cv2.imread(path)
        else: # It's a video frame
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                # Fallback to a black image if frame extraction fails
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert from BGR (OpenCV) to RGB (PIL)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model(model, criterion, optimizer, train_loader, validation_loader, device, num_epochs=1):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.round(torch.sigmoid(outputs.view(-1)))
                correct += (preds == labels).sum().item()
        accuracy = correct / len(validation_loader.dataset)
        print(f"Validation Accuracy: {accuracy:.4f}")

    return model

if __name__ == '__main__':
    # --- Configuration ---
    batch_size = 32
    num_epochs_head = 2 # Epochs for training the classifier head
    num_epochs_finetune = 3 # Epochs for fine-tuning the whole model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    train_dataset = DeepfakeDataset('dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataset = DeepfakeDataset('dataset/validation', transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Setup ---
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # --- STAGE 1: Train the classifier head ---
    print("\n--- Stage 1: Training the classifier head ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer_head = optim.Adam(model.fc.parameters(), lr=1e-3)
    model = train_model(model, criterion, optimizer_head, train_loader, validation_loader, device, num_epochs=num_epochs_head)

    # --- STAGE 2: Fine-tune the deeper layers ---
    print("\n--- Stage 2: Fine-tuning deeper layers ---")
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Use a lower learning rate for fine-tuning
    optimizer_finetune = optim.Adam(model.parameters(), lr=1e-5)
    model = train_model(model, criterion, optimizer_finetune, train_loader, validation_loader, device, num_epochs=num_epochs_finetune)

    # --- Save the final model ---
    torch.save(model.state_dict(), 'war_footage_detector.pth')
    print("\nModel training complete and saved to 'war_footage_detector.pth'")

# --- Batched Inference Function ---
def predict_video(video_path, model, transform, device, frame_rate=5, batch_size=32):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    predictions = []
    frames_batch = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            processed_frame = transform(image)
            frames_batch.append(processed_frame)

            if len(frames_batch) == batch_size:
                frames_tensor = torch.stack(frames_batch).to(device)
                with torch.no_grad():
                    outputs = model(frames_tensor)
                    preds = torch.sigmoid(outputs.view(-1))
                    predictions.extend(preds.cpu().numpy())
                frames_batch = [] # Clear the batch
        count += 1

    # Process any remaining frames in the last batch
    if frames_batch:
        frames_tensor = torch.stack(frames_batch).to(device)
        with torch.no_grad():
            outputs = model(frames_tensor)
            preds = torch.sigmoid(outputs.view(-1))
            predictions.extend(preds.cpu().numpy())

    cap.release()

    if not predictions:
        return "No frames processed."

    avg_prediction = np.mean(predictions)
    return "Fake" if avg_prediction > 0.5 else "Real"