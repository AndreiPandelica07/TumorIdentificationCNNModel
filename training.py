import kagglehub
import torch
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from model2 import CNN

from torch.utils.data import DataLoader
cnn = CNN(
    input_height=128,
    input_width=128,
    num_filters=8,
    layer_count=3,
    kernel_size=3,
    pool_size=2,
    in_channels=1,
    fc_layer_neuronc=64
)
test_path=path + "/Testing"
train_path=path + "/Training"

transform= transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=10)
])
iterations=100
loss_fn=torch.nn.CrossEntropyLoss()

dataset_train=datasets.ImageFolder(root=train_path, transform=transform)
glioma_label = dataset_train.class_to_idx["glioma"]

weights = [
    3.0 if label == glioma_label else 1.0
    for label in dataset_train.targets
]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

train_loader = DataLoader(
    dataset_train,
    batch_size=32,
    sampler=sampler
)
average_loss=0
try:
    cnn=torch.load("cnn_model2.pth", weights_only=False)
except FileNotFoundError:
    pass
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.001)
for _ in range (iterations):
    total_loss=0
    print(f"Iteration {_+1}/{iterations}")
    for pictures, labels in train_loader:   
        outputs=cnn(pictures)
        loss=loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    average_loss=total_loss/len(train_loader)
    try:
        best_loss=torch.load("average_lossBest_Model2.pth", weights_only=False)
        if average_loss<best_loss:
            print("Saving the best model...")
            torch.save(cnn, "cnn_model2.pth")
            torch.save(average_loss, "average_lossBest_Model2.pth")
    except FileNotFoundError:
        torch.save(cnn, "cnn_model2.pth")
        torch.save(average_loss, "average_lossBest_Model2.pth")
    print(f"Average loss for iteration {_+1}: {average_loss:.6f}")



