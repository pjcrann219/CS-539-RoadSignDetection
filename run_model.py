import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from CNN import CNN_32x32
from DatasetLoaders import RoadSignDataset

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load Dataset
dataset = RoadSignDataset(root_dir='images', annotations_dir='annotations', transform=transform)

# Split Dataset into Training and Testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoader for test dataset
batch_size = 4
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and load the model
model = CNN_32x32().to(device)
model.load_state_dict(torch.load('final_model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to visualize the images with labels
def imshow(img, labels, predictions):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('GroundTruth: ' + ' '.join('%5s' % labels[j] for j in range(len(labels))) + 
              '\nPredicted: ' + ' '.join('%5s' % predictions[j] for j in range(len(predictions))))
    plt.show()

# Run the model on a few examples
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Run the model
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# Print images with ground truth and predicted labels
imshow(torchvision.utils.make_grid(images.cpu()), labels, predicted.cpu())
