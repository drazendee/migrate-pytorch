# License: BSD
# Author: Sasank Chilamkurthy
# Based on example from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
# Recursively find all files under raw_data/
for filepath in glob.iglob("raw_data/**/*.*", recursive=True):
    # Ignore non images
    if not filepath.endswith((".png", ".jpg", ".jpeg")):
        continue

    # Open your image and perform transformation
    image = Image.open(filepath)
    if 'train' in filepath:
        transformed_image = data_transforms['train'](image)
    else :
        transformed_image = data_transforms['val'](image)
    
    # Get the output file and folder path
    output_filepath = filepath.replace("raw_data", "preprocessed")
    output_dir = os.path.dirname(output_filepath)
    # Ensure the folder exists
    os.makedirs(output_dir, exist_ok=True)

    save_image(transformed_image, output_filepath)