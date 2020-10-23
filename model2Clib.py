import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import cv2
import numpy as np

model = torch.load('weights/dataset-images_model_24.pt')
model.cpu()
model.eval()
example = torch.randn(1, 3, 416, 416)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("weights/Clib_model.pt")#保存模型位置