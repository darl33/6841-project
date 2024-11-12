import torch
import gc
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
# from testpt_v3b import predict_video
from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('deepfake_detector.pth', weights_only=True))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    return frames
def predict_video(video_path, model, transform, frame_rate=5):
    frames = extract_frames(video_path, frame_rate)
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames).to(device)

    with torch.no_grad():
        outputs = model(frames)
        predictions = torch.sigmoid(outputs.view(-1))
        avg_prediction = predictions.mean().item()
    
    return "Fake" if avg_prediction > 0.5 else "Real"


test_video_path = input("Enter the path to the test video: ")
result = predict_video(test_video_path, model, transform)

torch.cuda.empty_cache()  # if using GPU
gc.collect()              

print(f"The video is likely {result}.")
