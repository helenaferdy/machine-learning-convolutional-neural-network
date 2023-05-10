import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transformation for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)

class_names = train_data.classes

# Create data loaders
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

#smaller train data of 1000 samples
train_data_subset, _ = random_split(train_data, [1000, len(train_data) - 1000])
train_loader = DataLoader(train_data_subset, batch_size=64, shuffle=True)

# Create CNN Model
class CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units*7*7,
                out_features=output_shape
            )
        )

    def forward(self, x):
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.classifier(x)
        return x


# Instantiate model
model = CNN(
    input_shape=1,
    hidden_units=5,
    output_shape=len(class_names)).to(device)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device=device):
  train_loss, train_acc = 0,0
  model.train()
  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y, y_pred.argmax(dim=1)) #logits to label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"\nTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.5f}")


def test_step(model, data_loader, loss_fn, accuracy_fn, device=device):
  test_loss, test_acc = 0,0
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      test_pred = model(X)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y, test_pred.argmax(dim=1)) #logits to label
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"\nTest loss: {test_loss:.5f} | Train accuracy: {test_acc:.5f}")
      
      
def make_predictions(model, data: list, device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


## Train model
epochs = 5
for epoch in tqdm(range(epochs)):
  print(f"\nEpoch: {epoch+1}\n---------------------------------------------------------------")
  train_step(model, train_loader, loss_fn, optimizer, accuracy_fn)
  test_step(model, test_loader, loss_fn, accuracy_fn)

# Prepare data for prediction
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# Make prediction
pred_probs = make_predictions(model=model, 
                             data=test_samples)

pred_classes = pred_probs.argmax(dim=1)
print(f"Predictions labels: {pred_classes}")


# Plot predictions result
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)
  plt.imshow(sample.squeeze(), cmap="gray")
  #calculate confidence
  confidence, predicted_class = torch.max(pred_probs[i], dim=0)
  pred_label = predicted_class
  title_text = f"pred: {pred_label} | conf: {confidence.item()*100:.1f}%"
  
  truth_label = test_labels[i]
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);