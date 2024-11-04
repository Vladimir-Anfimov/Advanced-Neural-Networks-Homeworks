Wandb Dashboard: https://wandb.ai/vladimir-anfimov-alexandru-ioan-cuza/advanced-neural-networks-hw3

# 1. During sweeps I had the following variations:
    1.1. Models
        1.1.1. ResNet18, but with some modifications to make it work with the CIFAR-100 dataset, namely the first convolutional layer has a kernel size of 3 and the stride of 1, also the max pooling was replaced by the identity function in order to minimize the loss of information.
        This is because ResNet18 was initially designed for the ImageNet dataset, which has images of 224x224 pixels, while CIFAR-100 has images of 32x32 pixels.
        1.1.2. PreActResNet-18
    1.2. Optimizers
        1.2.1. SGD
        1.2.2. AdamW
    1.3. Transfomations
        1.3.1. RandomHorizontalFlip
        1.3.2. RandomCrop
        1.3.3. RandomRotation
        1.3.4. ColorJitter
        1.3.5. RandomErasing
    1.4. Data Augmentation
        1.4.1. Cutout
        1.4.2. Mixup

{
  "model": {
    "value": "PreActResNet-18"
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          41,
          55,
          105
        ],
        "2": [
          1,
          41,
          55,
          105
        ],
        "3": [
          1,
          2,
          13,
          16,
          23,
          55
        ],
        "4": "3.10.14",
        "5": "0.18.3",
        "8": [
          1,
          2,
          5
        ],
        "12": "0.18.3",
        "13": "linux-x86_64"
      },
      "cli_version": "0.18.3",
      "python_version": "3.10.14"
    }
  },
  "epochs": {
    "value": 100
  },
  "dataset": {
    "value": "CIFAR-100"
  },
  "optimizer": {
    "value": "SGD"
  },
  "scheduler": {
    "value": "ReduceLROnPlateau"
  },
  "batch_size": {
    "value": 64
  },
  "learning_rate": {
    "value": 0.001
  },
  "early_stopping": {
    "value": false
  },
  "test_batch_size": {
    "value": 2000
  },
  "data_augmentations": {
    "value": [
      "RandomCrop",
      "RandomHorizontalFlip",
      "ColorJitter",
      "RandomErasing"
    ]
  }
}

# Results
| #  | Model           | Dataset    | Optimizer | Scheduler           | Epochs | Batch Size | Learning Rate | Early Stopping | Test Batch Size | Data Augmentations                                  | WandB CLI Version | Python Version |
|----|----------------|------------|-----------|---------------------|--------|------------|---------------|----------------|----------------|-----------------------------------------------------|-------------------|----------------|
| 1  | PreActResNet-18 | CIFAR-100  | SGD       | ReduceLROnPlateau   | 100    | 64         | 0.001         | false          | 2000           | RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing | 0.18.3           | 3.10.14        |
| 2  | PreActResNet-18 | CIFAR-100  | SGD       | ReduceLROnPlateau   | 100    | 64         | 0.001         | false          | 2000           | RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing | 0.18.3           | 3.10.14        |
| 3  | resnet18       | CIFAR-100  | AdamW     | ReduceLROnPlateau   | 100    | 64         | 0.001         | false          | 2000           | RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing | 0.18.5           | 3.10.12        |
| 4  | resnet18       | CIFAR-100  | AdamW     | ReduceLROnPlateau   | 100    | 64         | 0.001         | false          | 2000           | RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing | 0.18.3           | 3.10.14        |
| 5  | resnet18       | CIFAR-100  | AdamW     | ReduceLROnPlateau   | 100    | 64         | 0.001         | false          | 2000           | RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing | 0.18.3           | 3.10.14        |


