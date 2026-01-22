import torch
import torchvision.transforms as transforms
from PIL import Image
import time

# Load pretrained ResNet50
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
model.eval()

# ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    start = time.time()

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_index = probabilities.argmax().item()
    confidence = round(probabilities[predicted_index].item() * 100, 2)

    latency = round((time.time() - start) * 1000, 2)

    # Load labels
    with open("utils/imagenet_labels.txt") as f:
        labels = f.read().splitlines()

    return labels[predicted_index], confidence, latency
