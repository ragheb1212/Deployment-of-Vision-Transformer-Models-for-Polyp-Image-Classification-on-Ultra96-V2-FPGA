
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from vit_model import SimpleViT

# ======================= PARAMÈTRES ===========================
DATA_DIR = r"D:\polybe detection project\GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= TRANSFORMATIONS =======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================= MODÈLE =======================
model = SimpleViT(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, test_accuracies = [], []

# ======================= ENTRAÎNEMENT =======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # ======================= ÉVALUATION =======================
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    test_accuracies.append(acc)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("📘 Epoch {}/{}".format(epoch+1, EPOCHS))
    print("   🔹 Loss       : {:.4f}".format(train_losses[-1]))
    print("   🔹 Accuracy   : {:.4f}".format(acc))
    print("   🔹 Recall     : {:.4f}".format(recall))
    print("   🔹 F1-score   : {:.4f}".format(f1))
    print("   🔹 Confusion matrix :\n{}\n".format(cm))

# ======================= COURBES =======================
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Valeur")
plt.legend()
plt.title("Courbes entraînement")
plt.savefig("training_curves.png")

# ======================= SAUVEGARDE =======================
torch.save(model.state_dict(), "vit_polyp_epoch.pth")

# ======================= EXPORT ONNX =======================
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
model.eval()
torch.onnx.export(
    model,
    dummy_input,
    "vit_polyp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=9,
    dynamic_axes={"input": {0: "batch_size"}}
)
print("✅ Modèle exporté : vit_polyp.onnx")
