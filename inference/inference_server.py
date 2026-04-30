#! imports

import json
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

#? configuration
EFFICIENTNET_CHECKPOINT = "efficientnet_b0_best.pt"
RESNET_CHECKPOINT       = "resnet50_best.pt"
CLASS_IDX               = "class_to_idx.json"
MODEL_NAME              = "ensemble (efficientnet_b0 + resnet50)"

#! Loading class mapping
with open(CLASS_IDX) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(class_to_idx)

#! Loading threshold
with open("threshold.json") as f:
    THRESHOLD = json.load(f)["threshold"]
    print(f"Threshold loaded: {THRESHOLD:.4f}")

#! EfficientNet setup
effnet = models.efficientnet_b0()
in_features = effnet.classifier[1].in_features
effnet.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3, inplace=True),
    torch.nn.Linear(in_features, num_classes),
)
ckpt = torch.load(EFFICIENTNET_CHECKPOINT, map_location="cpu")
effnet.load_state_dict(ckpt["model_state"])
effnet.eval()
print(f"EfficientNet loaded — epoch {ckpt.get('epoch', '?')} — score {ckpt.get('score', '?'):.4f}")

#! ResNet50 setup
resnet = models.resnet50()
in_features = resnet.fc.in_features
resnet.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(in_features, num_classes),
)
ckpt = torch.load(RESNET_CHECKPOINT, map_location="cpu")
resnet.load_state_dict(ckpt["model_state"])
resnet.eval()
print(f"ResNet50 loaded — epoch {ckpt.get('epoch', '?')} — score {ckpt.get('score', '?'):.4f}")

print(f"\nEnsemble ready — {num_classes} classes — threshold={THRESHOLD:.4f}")

#! Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

#! API
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     MODEL_NAME,
        "backbones": ["efficientnet_b0", "resnet50"],
        "classes":   num_classes,
        "threshold": THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs_effnet = F.softmax(effnet(tensor)[0], dim=0)
        probs_resnet = F.softmax(resnet(tensor)[0], dim=0)
        probs        = ((probs_effnet + probs_resnet) / 2).numpy()

    top3_idx = probs.argsort()[::-1][:3]
    top3     = [[idx_to_class[i], round(float(probs[i]), 4)] for i in top3_idx]
    max_conf = float(probs.max())
    pred_idx = int(probs.argmax())

    if max_conf < THRESHOLD:
        return {
            "class":        "Unrecognized / Low Confidence",
            "confidence":   round(max_conf, 4),
            "unrecognized": True,
            "top3":         top3,
            "model":        MODEL_NAME,
        }

    return {
        "class":        idx_to_class[pred_idx],
        "confidence":   round(max_conf, 4),
        "unrecognized": False,
        "top3":         top3,
        "model":        MODEL_NAME,
    }