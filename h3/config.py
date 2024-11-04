import json
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision.transforms import v2

from models import MLP, LeNet, PreActResNet, ResNet18

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.options = json.load(f)

        self.validate()

    def validate(self):
        assert "dataset" in self.options, "dataset not found in config"
        assert self.options["dataset"] in ["MNIST", "CIFAR-10", "CIFAR-100"], "unsupported dataset"

        assert "batch_size" in self.options, "batch_size not found in config"
        assert isinstance(self.options["batch_size"], int), "batch_size must be an integer"

        assert "test_batch_size" in self.options, "test_batch_size not found in config"
        assert isinstance(self.options["test_batch_size"], int), "test_batch_size must be an integer"

        assert "epochs" in self.options, "epochs not found in config"
        assert isinstance(self.options["epochs"], int), "num_epochs must be an integer"

        assert "learning_rate" in self.options, "learning_rate not found in config"
        assert isinstance(self.options["learning_rate"], float), "learning_rate must be a float"

        assert "model" in self.options, "model not found in config"
        assert self.options["model"] in ["resnet18", "PreActResNet-18"] if self.options["dataset"] != "MNIST" else ["LeNet", "MLP"], "unsupported model for dataset"

        assert "optimizer" in self.options, "optimizer not found in config"
        assert self.options["optimizer"] in [
            "SGD",
            "SGD_momentum",
            "SGD_nesterov",
            "SGD_weight_decay",
            "Adam",
            "AdamW",
            "RmsProp"], "unsupported optimizer"

        assert "scheduler" in self.options, "scheduler not found in config"
        assert self.options["scheduler"] in ["StepLR", "ReduceLROnPlateau", ""], "unsupported scheduler"

        assert "early_stopping" in self.options, "early_stopping not found in config"
        assert self.options["early_stopping"] in [False, True], "unsupported early_stopping"

        assert "data_augmentations" in self.options, "data_augmentations not found in config"
        for data_augmentation in self.options["data_augmentations"]:
            assert data_augmentation in [
                "RandomHorizontalFlip",
                "RandomRotation",
                "ColorJitter",
                "RandomGrayscale",
                "RandomPerspective",
                "RandomAffine",
                "RandomCrop",
                "RandomErasing",
                "MixUp",
                "CutMix"], "unsupported data augmentation"

    def __getitem__(self, key):
        return self.options[key]

    def __str__(self):
        return str(self.options)
    
    def get_dataset(self):
        match self.options["dataset"]:
            case "MNIST":
                return MNIST(root='./data', train=True, download=True), MNIST(root='./data', train=False, download=True)
            case "CIFAR-10":
                return CIFAR10(root='./data', train=True, download=True), CIFAR10(root='./data', train=False, download=True)
            case "CIFAR-100":
                return CIFAR100(root='./data', train=True, download=True), CIFAR100(root='./data', train=False, download=True)
        
    def get_optimizer(self, model_parameters):
        match self.options["optimizer"]:
            case "SGD":
                return SGD(model_parameters, lr=self.options["learning_rate"])
            case "SGD_momentum":
                return SGD(model_parameters, lr=self.options["learning_rate"], momentum=0.9)
            case "SGD_nesterov":
                return SGD(model_parameters, lr=self.options["learning_rate"], momentum=0.9, nesterov=True)
            case "SGD_weight_decay":
                return SGD(model_parameters, lr=self.options["learning_rate"], weight_decay=5e-4)
            case "Adam":
                return Adam(model_parameters, lr=self.options["learning_rate"])
            case "AdamW":
                return AdamW(model_parameters, lr=self.options["learning_rate"])
            case "RmsProp":
                return RMSprop(model_parameters, lr=self.options["learning_rate"])
            
    
    def get_scheduler(self, optimizer):
        match self.options["scheduler"]:
            case "StepLR":
                return StepLR(optimizer, step_size=10, gamma=0.1)
            case "ReduceLROnPlateau":
                return ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
            case "":
                return None
            
    def get_images_size(self):
        match self.options["dataset"]:
            case "MNIST":
                return 28
            case "CIFAR-10":
                return 32
            case "CIFAR-100":
                return 32
            
    def get_transforms(self):
        data_augmentations = self.options["data_augmentations"]
        transforms_pre_norm = []
        for data_augmentation in data_augmentations:
            match data_augmentation:
                case "RandomHorizontalFlip":
                    transforms_pre_norm.append(v2.RandomHorizontalFlip())
                case "RandomRotation":
                    transforms_pre_norm.append(v2.RandomRotation(degrees=15))
                case "ColorJitter":
                    transforms_pre_norm.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
                case "RandomGrayscale":
                    transforms_pre_norm.append(v2.RandomGrayscale(p=0.1))
                case "RandomPerspective":
                    transforms_pre_norm.append(v2.RandomPerspective(distortion_scale=0.1, p=0.1))
                case "RandomAffine":
                    transforms_pre_norm.append(v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.0), shear=15))
                case "RandomCrop":
                    imgSize = self.get_images_size()
                    transforms_pre_norm.append(v2.RandomCrop(imgSize, padding=4))

        transforms_post_norm = []
        if 'RandomErasing' in data_augmentations:
            transforms_post_norm.append(
                v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)))

        return transforms_pre_norm, transforms_post_norm
    
    def get_data_augmentations(self):
        output_layer_size = 100 if self.options["dataset"] == "CIFAR-100" else 10
        augmentations = []
        for data_augmentation in self.options["data_augmentations"]:
            match data_augmentation:
                case "MixUp":
                    augmentations.append(v2.CutMix(num_classes=output_layer_size))
                case "CutMix":
                    augmentations.append(v2.MixUp(num_classes=output_layer_size))
    
    def get_model(self):
        output_layer_size = 100 if self.options["dataset"] == "CIFAR-100" else 10

        match self.options["model"]:
            case "resnet18":
                return ResNet18(output_layer_size)
            case "PreActResNet-18":
                return PreActResNet(output_layer_size)
            case "LeNet":
                return LeNet()
            case "MLP":
                return MLP()
        
    # returns mean and std for normalization
    def get_normalization(self):
        match self.options["dataset"]:
            case "MNIST":
                return v2.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                    inplace=True
                )
            case "CIFAR-10":
                return v2.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                    inplace=True
                )
            case "CIFAR-100":
                return v2.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                    inplace=True
                )