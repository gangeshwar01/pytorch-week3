
# File: code/resnet_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Import Grad-CAM utility (created in the next step)
# from utils_gradcam import GradCAM, visualize_cam

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20 # Train for more epochs (e.g., 40-60) for potentially better results
LEARNING_RATE = 0.001
OUTPUT_DIR = "../runs/cls"
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ResNet-18 Implementation ---

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.
    Two 3x3 convolutional layers with a shortcut connection.
    """
    expansion = 1 # Expansion factor for output channels

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if necessary (stride != 1 or in_channels != out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Add shortcut connection
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ ResNet architecture implementation. """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution layer (adapted for CIFAR-10: 3x3 kernel, stride 1)
        # Original ResNet for ImageNet uses kernel_size=7, stride=2, which is too aggressive for 32x32 images.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Creates a stage of residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # Save this layer's output for Grad-CAM
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18():
    """Factory function for ResNet-18."""
    return ResNet(BasicBlock, [2, 2, 2, 2])

# --- Data Loading and Preprocessing ---
def load_data():
    print("Loading CIFAR-10 data...")
    # Transformations for training data with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Transformations for test data (normalization only)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_targets

# --- Visualization Functions ---
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "curves_cls.png"))
    plt.close()

def plot_confusion_matrix(targets, preds, classes):
    cm = confusion_matrix(targets, preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

def plot_predictions(dataloader, model, device, filename_prefix, correctly_classified=True):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))
    num_to_plot = 16
    plot_idx = 1
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    for i in range(len(images)):
        if plot_idx > num_to_plot:
            break

        condition = (preds[i] == labels[i]) if correctly_classified else (preds[i] != labels[i])
        if condition:
            img = images[i].cpu() * std + mean # De-normalize for visualization
            img = img.permute(1, 2, 0) # CHW to HWC
            plt.subplot(4, 4, plot_idx)
            plt.imshow(img)
            plt.axis('off')
            title = f"True: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[preds[i]]}"
            plt.title(title, color='green' if correctly_classified else 'red', fontsize=10)
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}.png"))
    plt.close()

# --- Main Execution ---
def main():
    # Load data
    trainloader, testloader = load_data()

    # Initialize model, criterion, optimizer
    print("Initializing ResNet-18 model...")
    model = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, testloader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.2f}%")

    # Final evaluation and visualization generation
    print("Generating visualizations...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))

    # 1. Loss and Accuracy Curves
    plot_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

    # 2. Confusion Matrix
    _, _, test_preds, test_targets = evaluate(model, testloader, criterion, DEVICE)
    plot_confusion_matrix(test_targets, test_preds, CIFAR10_CLASSES)

    # 3. Prediction Grids
    plot_predictions(testloader, model, DEVICE, "preds_grid", correctly_classified=True)
    plot_predictions(testloader, model, DEVICE, "miscls_grid", correctly_classified=False)
    
    # 4. Grad-CAM (Requires separate utility file)
    # print("Generating Grad-CAM visualizations...")
    # try:
    #     target_layer = model.layer4[-1] # Target the last block of the last stage
    #     cam = GradCAM(model, target_layer)
    #     visualize_cam(cam, testloader, device=DEVICE, classes=CIFAR10_CLASSES, output_dir=OUTPUT_DIR)
    # except NameError:
    #     print("Skipping Grad-CAM: utils_gradcam.py not imported or function missing.")
    # except Exception as e:
    #     print(f"Error during Grad-CAM generation: {e}")

if __name__ == "__main__":
    main()
