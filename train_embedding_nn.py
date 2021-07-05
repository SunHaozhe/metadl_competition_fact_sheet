import os
import time
import shutil
import glob
import numpy as np
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F  
from torchvision import transforms, models


def train_embedding_nn(df, IMAGE_PATH, LABEL_COLUMN, USE_NORMALIZATION, 
    number_of_classes, device, batch_size=64, epochs=1000):
    
    print("Training the feature extractor neural network...")

    if not USE_NORMALIZATION:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    
    train_examples = np.random.choice(range(df.shape[0]), size=int(df.shape[0] * 0.7), replace=False)
    train_df = df.iloc[train_examples, :]
    val_df = df.iloc[[xx for xx in range(df.shape[0]) if xx not in train_examples], :]

    train_dataset = TorchImageDataset(train_df, IMAGE_PATH, transform=transform, label=LABEL_COLUMN)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TorchImageDataset(val_df, IMAGE_PATH, transform=transform, label=LABEL_COLUMN)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = getModel(device, number_of_classes=number_of_classes)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=5e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", verbose=True, patience=5, factor=0.1)

    checkpoints_dir = "model_checkpoints"
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir) 
    os.makedirs(checkpoints_dir)
    best_model = [np.inf, os.path.join(checkpoints_dir, "best_model.pth")] # (score, path)

    for epoch in range(epochs):
        t0 = time.time()

        # train
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            current_batch_size = y.shape[0]

            optim.zero_grad()
            
            logits = model(X)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optim.step()

            train_loss += loss.item()

            acc = (logits.argmax(dim=1) == y).sum().item() / current_batch_size
            train_acc += acc

        train_acc /= len(train_dataloader)
        train_acc *= 100
        train_loss /= len(train_dataloader)

        # validation
        val_acc = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)

                current_batch_size = y.shape[0]

                logits = model(X)
                loss = F.cross_entropy(logits, y)

                val_loss += loss.item()
                
                acc = (logits.argmax(dim=1) == y).sum().item() / current_batch_size
                val_acc += acc
                
        
        val_acc /= len(val_dataloader)
        val_acc *= 100
        val_loss /= len(val_dataloader)

        # save model checkpoint
        if val_loss < best_model[0]:
            best_model[0] = val_loss
            torch.save(model.state_dict(), best_model[1])

        epoch_time = time.time() - t0

        print("Epoch {} | Train loss {:.3f} | Train acc {:.2f}% | Val loss {:.3f} | Val acc {:.2f}% | Time {:.2f} seconds.".format(
            epoch, train_loss, train_acc, val_loss, val_acc, epoch_time))

        # scheduling learning rates
        scheduler.step(val_loss)
    return best_model[1]


def getModel(device, number_of_classes):
    model = models.resnet18(pretrained=True)

    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = True
        elif name.startswith("layer4.1"):
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, number_of_classes)
    model.to(device)
    return model



class TorchImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, IMAGE_PATH, transform=None, label="unicode_code_point"):
        super().__init__()
        self.df = df
        self.IMAGE_PATH = IMAGE_PATH
        self.transform = transform 
        self.label = label
        
        self.construct_items()
        self.convert_labels()

    def construct_items(self):
        """
        Builds self.items
        
        item: tuple
            (path, raw_label, label/target)
        """

        self.items = []

        for i in range(self.df.shape[0]):
            file_name, raw_label = self.df.iloc[i, 0], self.df.iloc[i, 1]
            path = os.path.join(self.IMAGE_PATH, file_name)
            self.items.append([path, raw_label])

    def convert_labels(self):
        """
        convert raw labels (categorical) to labels (consecutive integers starting from 0)
        """
        conversion_map = dict()
        cnt = 0
        for i in range(len(self.items)):
            if self.items[i][1] not in conversion_map:
                conversion_map[self.items[i][1]] = cnt
                cnt += 1
            self.items[i][1] = conversion_map[self.items[i][1]]

    def __getitem__(self, index):
        img_path = self.items[index][0]
        target = self.items[index][1]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img) 
        
        return img, target

    def __len__(self):
        return len(self.items)
