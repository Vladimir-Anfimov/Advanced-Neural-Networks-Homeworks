# This code was used to train a Vision Transformer model on the CIFAR-100 dataset with noisy labels.
# It must run in a Kaggle notebook with the following dataset: https://www.kaggle.com/competitions/fii-atnn-2024-project-noisy-cifar-100/overview

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn
from torch import GradScaler
from tqdm import tqdm
import wandb
from kaggle_secrets import UserSecretsClient

MODEL_NAME = "vit_small_patch16_224"

user_secrets = UserSecretsClient()
WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)
wandb.init(project="ann-project", name=f"{MODEL_NAME}-noisy-cifar-{np.random.randint(10000)}", reinit=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
enable_half = True
scaler = GradScaler(enabled=enable_half)

class CIFAR100_noisy_fine(Dataset):
    def __init__(self, root, train, transform, download):
        cifar100 = CIFAR100(root=root, train=train, transform=None, download=download)
        data, targets = tuple(zip(*cifar100))
        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError
            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError
            targets = noise_file["noisy_label"]
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class SimpleCachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        label = self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

train_img_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandAugment(num_ops=6, magnitude=9),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
])
test_img_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
])

root_dir = "/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100"
train_raw = CIFAR100_noisy_fine(root=root_dir, train=True, download=False, transform=None)
test_raw = CIFAR100_noisy_fine(root=root_dir, train=False, download=False, transform=None)

train_set = SimpleCachedDataset(train_raw, transform=train_img_transform)
test_set = SimpleCachedDataset(test_raw, transform=test_img_transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=8)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=8)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

cutmix_transform = v2.CutMix(num_classes=100, alpha=1.0)
mixup_transform = v2.MixUp(num_classes=100, alpha=1.0)
cutmix_or_mixup = v2.RandomChoice([cutmix_transform, mixup_transform])

best_acc = 0.0
num_epochs = 20
patience = 3
no_improve_epochs = 0
best_model_path = "/kaggle/working/best_model.pth"

def train_one_epoch():
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs, targets = cutmix_or_mixup(inputs, targets)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        hard_targets = targets.argmax(dim=1)
        total += targets.size(0)
        correct += predicted.eq(hard_targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

@torch.inference_mode()
def val():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

@torch.inference_mode()
def inference():
    best_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)
    best_model.eval()

    all_labels = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = best_model(inputs)
        predicted = outputs.argmax(dim=1)
        all_labels.extend(predicted.tolist())

    return all_labels

with tqdm(range(num_epochs), desc="Training") as tbar:
    for epoch in tbar:
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = val()
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1

        wandb.log({
            "epoch": epoch + 1,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "best_validation_accuracy": best_acc
        })

        tbar.set_postfix({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.2f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.2f}",
            "best": f"{best_acc:.2f}"
        })

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered! No improvement in the last {patience} epochs.")
            break

data = {"ID": [], "target": []}
for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("/kaggle/working/submission.csv", index=False)