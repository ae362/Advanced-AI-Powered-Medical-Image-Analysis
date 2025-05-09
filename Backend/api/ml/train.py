import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from PIL import Image
import cv2
from .preprocessing import preprocess_for_model, create_brain_mask

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class BrainTumorDataset(Dataset):
    def __init__(self, images, masks, labels, transform=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask, torch.tensor(label, dtype=torch.long)

def load_and_preprocess_image(image_path):
    try:
        img_input, brain_mask = preprocess_for_model(image_path)
        return img_input[0], brain_mask
    except Exception as e:
        logger.warning(f"Error processing image {image_path}: {str(e)}")
        return None, None

def load_dataset(data_dir, num_threads=4):
    image_paths = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, img_name))
                labels.append(0 if label == 'negative' else 1)
    
    logger.info(f"Found {len(image_paths)} images")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(load_and_preprocess_image, path) for path in image_paths]
        images = []
        masks = []
        valid_indices = []
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing images")):
            img, mask = future.result()
            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask)
                valid_indices.append(idx)
    
    # Only keep labels for successfully processed images
    labels = [labels[i] for i in valid_indices]
    
    return np.array(images), np.array(masks), np.array(labels)

def create_model(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    )
    return model

def train_model(dataset_name, epochs=30, batch_size=32, num_threads=4):
    logger.info(f"{'='*20} Training {dataset_name} model {'='*20}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(base_dir, 'data', dataset_name)
    model_save_dir = os.path.join(base_dir, 'api', 'ml', 'saved_models')
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, f'{dataset_name}_best_model.pth')

    logger.info(f"Loading data from: {dataset_dir}")
    logger.info(f"Model will be saved to: {model_save_path}")

    # Load and preprocess data
    X, masks, y = load_dataset(dataset_dir, num_threads)
    
    if len(X) == 0:
        raise ValueError(f"No valid images found in the dataset: {dataset_name}")

    X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(X, masks, y, test_size=0.2, random_state=42)

    num_classes = len(np.unique(y))
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Found classes: {np.unique(y)}")

    # Define transforms for single-channel images
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Single-channel normalization
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Single-channel normalization
    ])

    # Create datasets and dataloaders
    train_dataset = BrainTumorDataset(X_train, masks_train, y_train, transform=train_transform)
    val_dataset = BrainTumorDataset(X_val, masks_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads, pin_memory=True)

    # Create model
    model = create_model(num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, masks, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}")

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, masks, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")

    logger.info(f"Training completed for {dataset_name}.")

def main():
    logger.info("Starting model training...")
    datasets = ['brain_tumor', 'cancer']
    
    for dataset in datasets:
        try:
            train_model(dataset)
            logger.info(f"Training for {dataset} completed successfully.")
        except Exception as e:
            logger.error(f"Error training {dataset} model: {str(e)}")
            logger.exception("Traceback:")
            continue

    logger.info("Model training completed!")

if __name__ == "__main__":
    main()