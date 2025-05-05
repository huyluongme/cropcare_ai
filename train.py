import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

checkpoint_dir = './checkpoints'

if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

os.mkdir(checkpoint_dir)

data_dir = './PlantVillage-Dataset/raw/color'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# print(full_dataset)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

class_names = full_dataset.classes
# print(f"Classes ({len(class_names)}):", class_names)

import torchvision.models as models
import torch.nn as nn

from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # ví dụ 38 lớp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm

def train_model(model, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            acc = 100. * correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1} completed: Avg Loss={running_loss/len(train_loader):.4f}, Train Acc={epoch_acc:.2f}%")
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_{epoch+1}.pth')
        print(f"Model saved after epoch {epoch+1}")


if __name__ == "__main__":
    train_model(model, epochs=2)