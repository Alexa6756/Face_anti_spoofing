import cv2
import torch
import numpy as np
from model import AntiSpoofCNN
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target = dict([*self.model.named_modules()])[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def generate(self, input_image, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        loss = output[:, target_class]
        loss.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AntiSpoofCNN()
model.load_state_dict(torch.load("models/anti_spoof_rgb_model.pth", map_location=device))
model.eval().to(device)

img_path = "sample.png"  
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
img_tensor = img_tensor.to(device)

gradcam = GradCAM(model, "conv2")  
cam = gradcam.generate(img_tensor)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
cv2.imshow("Grad-CAM", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
