import torch
from torchvision import transforms as tfs
from torchvision import models
from torch import nn
import numpy as np
import cv2
from PIL import Image
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]

inference_transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(norm_mean, norm_std),
    ])
dummy_input = np.ones((224, 224, 3), dtype=np.uint8)
dummy_input = Image.fromarray(cv2.cvtColor(dummy_input, cv2.COLOR_BGR2RGB))
img_tensor = inference_transform(dummy_input)
img_tensor.unsqueeze_(0)
model = models.resnet50()
model.fc = nn.Linear(2048, 30)
model.load_state_dict(torch.load('D:/work/PatternRecognition/save_model1.pth'))
torch.onnx.export(model,img_tensor,"D:/work/PatternRecognition/test_model.onnx",export_params=True)