import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
from PIL import Image

import torchvision.transforms as T

from models.compression_state_estimator import (
    CompressionStateEstimator
)


class CompressionDataset(Dataset):

    def __init__(self, root_dir):

        self.paths = list(
            Path(root_dir).glob("*.jpg")
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        img = Image.open(path).convert("RGB")

        img = self.transform(img)

        filename = path.stem

        quality = 1.0

        if "_q10" in filename:
            quality = 0.1

        elif "_q20" in filename:
            quality = 0.2

        elif "_q40" in filename:
            quality = 0.4

        elif "_q60" in filename:
            quality = 0.6

        return img, torch.tensor([quality])


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = CompressionDataset(
    "benchmarks/compressionbench/jpeg"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = CompressionStateEstimator()

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

criterion = nn.MSELoss()

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for imgs, targets in loader:

        imgs = imgs.to(device)

        targets = targets.float().to(device)

        preds = model(imgs)

        loss = criterion(
            preds,
            targets
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss: {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "cse_trained.pt"
)

print("CSE training complete")
