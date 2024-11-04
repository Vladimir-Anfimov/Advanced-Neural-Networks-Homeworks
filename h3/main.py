import torch
from torch import GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import Config
from torchvision.transforms import v2 as v2
from torch import nn
import wandb

CONFIG_PATH = input("Enter a config path: ")
if not CONFIG_PATH:
    raise ValueError("Please provide a config path")


config = Config(CONFIG_PATH)

wandb.init(
    project="advanced-neural-networks-hw3",
    config=config,
    name=f"{config['model']}_{config['dataset']}",
    reinit=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
pin_memory = True if torch.cuda.is_available() else False
enabled_half_precision = True if torch.cuda.is_available() else False
scaler = GradScaler(device=device, enabled=enabled_half_precision)

class ConditionalTransforms:
    def __init__(self, train=True):
        self.train = train

        if self.train:
            transforms_pre_norm, transforms_post_norm = config.get_transforms()

            self.transforms = v2.Compose([
                v2.ToImage(),
                *transforms_pre_norm,
                v2.ToDtype(torch.float32, scale=True),
                config.get_normalization(),
                *transforms_post_norm,               
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                config.get_normalization(),
            ])

    def __call__(self, img):
        return self.transforms(img)


class CachedImagesDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.images = [image for image, _ in dataset]
        self.labels = [label for _, label in dataset]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

train_set_raw, test_set_raw = config.get_dataset()
train_set = CachedImagesDataset(train_set_raw, transform=ConditionalTransforms(train=True))
test_set = CachedImagesDataset(test_set_raw, transform=ConditionalTransforms(train=False))


num_workers = config.options["num_workers"]
prefetch_factor = config.options["prefetch_factor"]

train_loader = DataLoader(
    train_set,
    batch_size=config.options["batch_size"],
    shuffle=True, 
    pin_memory=pin_memory,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor)

test_loader = DataLoader(
    test_set,
    batch_size=config.options["test_batch_size"],
    shuffle=False,
    pin_memory=pin_memory,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor)

model = config.get_model().to(device)

wandb.watch(model, log="all", log_freq=10)

criterion = nn.CrossEntropyLoss()
optimizer = config.get_optimizer(model.parameters())
scheduler = config.get_scheduler(optimizer)

data_augmentations = config.options["data_augmentations"]
cutmix_or_mixup = v2.RandomChoice(data_augmentations) if data_augmentations else None

def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        if cutmix_or_mixup:
            inputs, targets = cutmix_or_mixup(inputs, targets)
        inputs, targets = inputs.to(device, non_blocking=pin_memory), targets.to(device, non_blocking=pin_memory)
        
        with torch.autocast(device.type, enabled=enabled_half_precision):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        
        if cutmix_or_mixup:
            correct += predicted.eq(targets.argmax(dim=1)).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()


    return 100.0 * correct / total


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=pin_memory), targets.to(device, non_blocking=pin_memory)
        with torch.autocast(device.type, enabled=enabled_half_precision):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


best = 0.0
epochs = list(range(config.options["epochs"]))

patience  = 5
epochs_no_improve = 0

with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()
            
        if scheduler:
            scheduler.step(val_acc)

        if val_acc > best:
            best = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}") 

        wandb.log({
            "epoch": epoch + 1,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "best_validation_accuracy": best
        })

        if epochs_no_improve == patience and config.options["early_stopping"]:
            print("Early stopping")
            break
