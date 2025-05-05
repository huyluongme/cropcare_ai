import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

checkpoint_dir = './checkpoints'

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

class_names = full_dataset.classes
# print(f"Classes ({len(class_names)}):", class_names)

import torchvision.models as models
import torch.nn as nn

from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model.load_state_dict(torch.load(f'{checkpoint_dir}/model_2.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


from PIL import Image

def predict_image(image_path):
    model.eval()
    image = image_path.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    print("Predicted class:", class_names[predicted.item()])
    predicted_class = class_names[predicted.item()]
    return predicted_class

# predict_image("./PlantVillage-Dataset/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG")


def build_interface():
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload an image of a diseased plant"),
        outputs=gr.Textbox(label="Prediction result"),
        live=False,
        title="CropCare AI - Plant Disease Detection",
        description="Upload a leaf image and the system will predict the type of disease.",
        theme="default",
        allow_flagging="never",
        examples=None,
        submit_btn="Predict",
        clear_btn="Clear"
    )
    return interface

if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
