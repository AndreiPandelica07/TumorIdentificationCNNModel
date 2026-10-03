import torch
import kagglehub
from model2 import CNN
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
cnn=torch.load("cnn_model2.pth", weights_only=False)

transform= transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    
])
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
dataset_test=datasets.ImageFolder(root=path+"/Testing", transform=transform)
total_correct=0
total_samples=0
confusion_matrix=torch.zeros(4,4)
misclassified_images=[]
true_labels = []
predicted_labels = []
for picture, label in dataset_test: 
    picture=picture.unsqueeze(0)
    output=cnn(picture)
    predicted_label=torch.argmax(output, dim=1).item()
    true_labels.append(label)
    predicted_labels.append(predicted_label)
    confusion_matrix[label, predicted_label] += 1
    if predicted_label==label:
        total_correct+=1
    total_samples+=1
    if predicted_label!=label and label==0 :
        misclassified_images.append((picture.squeeze(0).cpu(), label, predicted_label))
accuracy=(total_correct/total_samples)*100
print(f"{accuracy:.2f}%")

print(classification_report(
    true_labels,
    predicted_labels,
    target_names=dataset_test.classes
))
