# License: BSD
# Author: Sasank Chilamkurthy
# Based on example from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

model = torch.load('output/model.pth')

was_training = model.training
model.eval()
images_so_far = 0
fig = plt.figure()

data_dir = 'preprocessed/hymenoptera_data'

image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms.Compose([transforms.ToTensor()]))
class_names = image_datasets.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img)
    #img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()  # pause a bit so that plots are updated

visualize_model_predictions(
    model,
    img_path='raw_data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)