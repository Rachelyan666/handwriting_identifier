import os
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


CKPT_PATH = os.environ.get("CKPT_PATH", "models/mnist_cnn.pt")

try:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True) 
except TypeError:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

classes = [str(c) for c in ckpt["classes"]]

model = SmallCNN(num_classes=len(classes))
model.classifier[-1] = nn.Linear(256, len(classes))
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

#match training
mean, std = (0.1307,), (0.3081,)
base_tfm = transforms.Compose([
    transforms.Grayscale(), 
    transforms.Resize((28, 28), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def preprocess(img: Image.Image, invert: bool) -> torch.Tensor:
    img = img.convert("L")
    if invert:
        img = ImageOps.invert(img)
    x = base_tfm(img).unsqueeze(0)
    return x

@torch.no_grad()
def predict(img: Image.Image, invert_colors: bool = True):
    if img is None:
        return {}
    x = preprocess(img, invert=invert_colors)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
    topk = np.argsort(-probs)[:5]
    return {classes[i]: float(probs[i]) for i in topk}

#gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload an image (digit/letter)"),
        gr.Checkbox(value=True, label="Invert colors (white ink on black)"),
    ],
    outputs=gr.Label(num_top_classes=5),
    title="Handwritten Digit/Letter Identifier",
    description=(
        "Upload a grayscale/photographed digit or letter. "
        "If your image is black text on white background, keep 'Invert colors' checked."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
