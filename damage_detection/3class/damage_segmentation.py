import os
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import numpy as np

class CrackSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        # Filter image files to only those with corresponding mask
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg')) and
               os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + '.jpg'))
        ])

        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.mask_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(image_file)[0] + '.jpg')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()  # Binary mask: cracks = 1, background = 0

        return image, mask

def get_dataloaders(img_dir, mask_dir, batch_size=8):
    dataset = CrackSegmentationDataset(img_dir, mask_dir, augment=True)
    val_split = int(0.2 * len(dataset))
    train_split = len(dataset) - val_split
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_split, val_split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def get_model():
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).bool()
    targets = targets.bool()
    intersection = (preds & targets).float().sum((1, 2, 3))
    union = (preds | targets).float().sum((1, 2, 3))
    dice = (2. * intersection + 1e-6) / (union + intersection + 1e-6)
    return dice.mean().item()


def train_segmentation_model(img_dir, mask_dir, epochs=10, lr=1e-4, batch_size=8, device='cuda'):
    train_loader, val_loader = get_dataloaders(img_dir, mask_dir, batch_size)
    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_dice = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc='Validation'):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_dice += dice_score(preds, masks)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Dice = {val_dice/len(val_loader):.4f}")
        if val_dice > best_val_dice:
            torch.save(model.state_dict(), './damage_detection/3class/crack_segmentation_model.pth')

    return model

def run_segmentation_on_folder(model, root_image_dir, output_dir, device, transform, threshold=0.5):
    for class_name in os.listdir(root_image_dir):
        class_dir = os.path.join(root_image_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        output_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)

        for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if not fname.endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, fname)
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(image_tensor)
                pred_mask = (torch.sigmoid(pred) > threshold).squeeze().cpu().numpy() * 255

            # Save predicted mask
            mask_image = Image.fromarray(pred_mask.astype('uint8')).resize(image.size)
            mask_path = os.path.join(output_dir, fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
            mask_image.save(mask_path)

def overlay_and_save_masks(image_dir, mask_dir, save_dir, alpha=0.4):
    os.makedirs(save_dir, exist_ok=True)

    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        mask_class_path = os.path.join(mask_dir, class_name)
        save_class_path = os.path.join(save_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        if not os.path.isdir(class_path):
            continue

        for fname in tqdm(os.listdir(class_path), desc=f"Overlaying {class_name}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, fname)
            mask_path = os.path.join(mask_class_path, fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
            save_path = os.path.join(save_class_path, fname)

            if not os.path.exists(mask_path):
                continue

            image = Image.open(img_path).convert("RGB").resize((256, 256))
            mask = Image.open(mask_path).convert("L").resize((256, 256))

            # Binarize mask
            mask = mask.point(lambda p: 255 if p > 0 else 0)

            # Create red RGBA overlay from binary mask
            red_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=(255, 0, 0))
            red_mask.putalpha(int(255 * alpha))  # Set alpha level

            image_rgba = image.convert("RGBA")
            overlayed = Image.alpha_composite(image_rgba, red_mask)

            # Save as RGB if it's a JPEG
            if fname.lower().endswith(('.jpg', '.jpeg')):
                overlayed.convert("RGB").save(save_path)
            else:
                overlayed.save(save_path)
                
if __name__ == '__main__':
    torch.manual_seed(1220)
    random.seed(1220)
    np.random.seed(1220)
    # train_segmentation_model(
    #     img_dir='./damage_detection/3class/images',
    #     mask_dir='./damage_detection/3class/masks',
    #     epochs=10,
    #     lr=1e-4,
    #     batch_size=8,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    # model.load_state_dict(torch.load('./damage_detection/3class/crack_segmentation_model_7853.pth', map_location=device))
    # model.to(device)
    # model.eval()

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image_dir = './crack_classification/full/full_crack_type'
    output_dir = './crack_classification/full/full_crack_segmentation'
    mask_dir = './crack_classification/full/full_crack_segmentation'

    run_segmentation_on_folder(model, image_dir, output_dir, device=device, transform=transform)

    overlay_and_save_masks(image_dir, mask_dir, './crack_classification/full/full_crack_segmentation_overlay')



