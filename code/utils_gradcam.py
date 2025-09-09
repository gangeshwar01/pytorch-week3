# File: code/utils_gradcam.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2 # Requires opencv-python: pip install opencv-python
import os

class GradCAM:
    """ Implementation of Grad-CAM for visualizing model attention. """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, target_class=None):
        # 1. Forward pass to get activations and predictions
        self.model.zero_grad()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 2. Backward pass to get gradients
        output[:, target_class].backward()

        # 3. Calculate weights (alpha_k) using Global Average Pooling on gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # 4. Generate heatmap (L_Grad-CAM) by weighting activations
        # self.activations shape: [batch_size, channels, height, width]
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Normalize heatmap for visualization
        heatmap = F.interpolate(heatmap, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.squeeze().cpu().detach().numpy(), target_class

def denormalize_image(img_tensor):
    """ De-normalizes a CIFAR-10 image tensor for visualization. """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0)) # HWC format
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def overlay_heatmap(img, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """ Overlays the heatmap on the original image. """
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid_img = cv2.addWeighted(np.uint8(255 * img), 1 - alpha, heatmap, alpha, 0)
    return overlaid_img

def visualize_cam(cam_object, dataloader, device, classes, output_dir, num_images=4):
    """ Generates and saves Grad-CAM visualizations for a few images. """
    images, labels = next(iter(dataloader))
    images = images.to(device)

    fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
    fig.suptitle("Grad-CAM Visualizations", fontsize=16)

    for i in range(num_images):
        img_tensor = images[i].unsqueeze(0) # Add batch dimension
        original_img = denormalize_image(images[i])

        # Generate heatmap
        heatmap, pred_class_idx = cam_object.generate_heatmap(img_tensor)
        overlaid_image = overlay_heatmap(original_img, heatmap)

        pred_class_name = classes[pred_class_idx]
        true_class_name = classes[labels[i]]

        # Plot original image
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"True: {true_class_name}")
        axes[i, 0].axis('off')

        # Plot heatmap
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Heatmap")
        axes[i, 1].axis('off')

        # Plot overlay
        axes[i, 2].imshow(overlaid_image)
        axes[i, 2].set_title(f"Overlay (Pred: {pred_class_name})")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "gradcam_results.png"))
    plt.close()

# Example usage (if running this file directly for testing)
if __name__ == "__main__":
    # --- Mock objects for testing ---
    # 1. Load a real model and data if available, otherwise create mock objects.
    # This part assumes resnet_cifar10.py can be imported or re-defined here for testing.
    print("Grad-CAM utility script. Run resnet_cifar10.py to generate visualizations.")

    # Example: Create a dummy model and data for syntax check
    # model = torchvision.models.resnet18(pretrained=False) # Placeholder
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Adapt for 32x32
    # model.fc = nn.Linear(512, 10)
    # target_layer = model.layer4[-1]
    # cam = GradCAM(model, target_layer)
    # dummy_input = torch.rand(1, 3, 32, 32)
    # heatmap, _ = cam.generate_heatmap(dummy_input)
    # print(f"Generated heatmap shape: {heatmap.shape}")
