import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

save_dir = 'cifar10_images'
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=None)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(1090):
   
    img, label = testset[i] 
    class_name = classes[label]
    
    filename = f'{save_dir}/{i+1}_{class_name}.png'
    img.save(filename)
    
    print(f'Saved {filename}')

print(f'\nAll images have been saved to {save_dir}/')