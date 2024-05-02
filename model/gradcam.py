import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from modules.cnn import ResNetModel  

class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.model.eval()
        self.feature_gradients = []
        self.feature_maps = []

        def save_gradients(module, grad_in, grad_out):
            self.feature_gradients.append(grad_out[0])

        self.feature_layer.register_backward_hook(save_gradients)

    def forward(self, x):
        self.feature_gradients = []
        self.feature_maps = []

        def save_feature_maps(module, input, output):
            self.feature_maps.append(output)

        self.feature_layer.register_forward_hook(save_feature_maps)

        output = self.model(x)
        return output

    def generate_cam(self, target_index=None):
        grad = self.feature_gradients[0].detach().cpu().numpy()
        fmap = self.feature_maps[0].detach().cpu().numpy()
        weights = np.mean(grad, axis=(2, 3))[0, :]
        cam = np.zeros(fmap.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * fmap[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (256, 128))  # Resize to input size
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def apply_colormap_on_image(org_img, cam, colormap=cv2.COLORMAP_JET):
    cam = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    cam = np.float32(cam) + np.float32(org_img)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

# Using the model
model = ResNetModel(in_channels=8) 
target_layer = model.encoder[-1]  # Last convolutional layer in the encoder block
grad_cam = GradCAM(model, target_layer)
