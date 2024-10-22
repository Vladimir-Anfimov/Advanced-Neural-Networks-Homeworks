# Experiment Results

| Experiments                                            | Test Acc. | Val Acc. |
|--------------------------------------------------------|-----------|----------|
| 1. DA-v1, SGD                                          | 71.23%    | 70.95%   |
| 2. DA-v1, Adam                                         | 69.11%    | 68.87%   |
| 3. DA-v2, SGD, Cutmix + Mixup                          | 76.99%    | 76.55%   |
| 4. DA-v2, SGD                                          | 63.02%    | 84.00%   |

## Experiment Details

1. [RandomAffine + SGD + RandomErasing: Train: 92.37, Val: 70.95, Best: 71.23](https://www.kaggle.com/code/vladimiranfimov/vladimir-anfimov-v1-with-randomaffine/edit)
    - Uses DA-v1 = RandomHorizontalFlip + ColorJitter + Normalize + RandomErasing + RandomAffine
    - Batch size: 64
    - ReduceLROnPlateau: Yes

2. [RandomCrop + Adam + RandomErasing: Train: 99.72, Val: 68.87, Best: 69.11](https://www.kaggle.com/code/vladimiranfimov/vladimir-anfimov-v2-with-randomcrop-adam/edit)
    - Uses DA-v1 = RandomHorizontalFlip + ColorJitter + Normalize + RandomErasing + RandomCrop
    - Batch size: 64
    - ReduceLROnPlateau: Yes

3. [RandomCrop + SGD + Cutmix + Mixup + RandomErasing: Train: 72.66, Val: 76.55, Best: 76.99](https://www.kaggle.com/code/vladimiranfimov/vladimir-anfimov-v3-with-randomcrop-and-mixup/edit)
    - Uses DA-v2 = RandomHorizontalFlip + ColorJitter + Normalize + RandomErasing + RandomCrop + Cutmix + Mixup
    - Batch size: 64
    - ReduceLROnPlateau: Yes

4. [SGD + RandomCrop: 60% Train Acc, 84% Val Acc](https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/vladimir-448a650a-12c9-4b79-84e6-924c38134277.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20241020/auto/storage/goog4_request%26X-Goog-Date%3D20241020T075615Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D9aafc66a36d668a2f83910d3e14cd2ed8d31fcbde015d6c918d961899caec4a4dc05e8e06c49b920b350e8b01b7da0cc3cd35afe2a38abae78065859b2328585ec54806224ae916a39bb2793bdfb6edc3bd588cc35cc33bc140ff17a725471b3de482a826a1b7e26c1a093a2275b802888d581b35a88971e19caf3f2b73601bf15f8170e1ed830b183aabb12c8a50a829aa6c8a6d1bf574aab01f6d18db5009c2dcc8daf2ea9d4b6103767eb5ab953a7016f0cf7e79f722185c4c1abe8ea4c5aa05c772f671c52a5c46ea23ddc6f7d9534b018c8b48439b9405f0aab0106530159ff17893acefb71d853536300830b0312cbdd6e4586adfed374341a2a2ebd6e)
    - Uses DA-v2 = RandomHorizontalFlip + ColorJitter + Normalize + RandomErasing + RandomCrop
    - Batch size: 64
    - ReduceLROnPlateau: Yes
