import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 45 * 24, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

data_transforms = transforms.Compose([
    transforms.Resize((180, 96)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder('./data/image/train', transform=data_transforms)
test_data = datasets.ImageFolder('./data/image/test', transform=data_transforms)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  

num_epochs = 10
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))  
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels.float().unsqueeze(1)).sum().item()
        total_train += labels.size(0)
    
    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    
    model.eval()
    total_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss_val = criterion(outputs, labels.float().unsqueeze(1))
            total_val_loss += loss_val.item()
            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels.float().unsqueeze(1)).sum().item()
            total_val += labels.size(0)
    
    val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_loss_history.append(val_loss)
    val_acc_history.append(val_accuracy)
    print(f"Epoch {epoch + 1}:")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        correct_test += (predicted == labels.float().unsqueeze(1)).sum().item()
        total_test += labels.size(0)

test_accuracy = correct_test / total_test
print(f"测试集准确率：{test_accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()