import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_model_path(disease_name: str) -> str:
    """Generate a consistent path for saving models"""
    base_dir = os.path.join('api/ml/saved_models')
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{disease_name.lower().replace(' ', '_')}_best_model.pth")

def preprocess_image(img_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

class DiseaseDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(disease_name, training_data, epochs=10, batch_size=32):
    logger.info(f"Starting model training for {disease_name}")
    logger.info(f"Training data classes: {list(training_data.keys())}")
    
    if not training_data:
        raise ValueError("No training data provided")

    images = []
    labels = []
    for class_name, image_paths in training_data.items():
        for image_path in image_paths:
            images.append(image_path)
            labels.append(1 if class_name == 'positive' else 0)

    if not images:
        raise ValueError("No valid images found in training data")

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DiseaseDataset(X_train, y_train, transform=train_transform)
    val_dataset = DiseaseDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)

        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = get_model_path(disease_name)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")

    return model_path, val_acc.item(), val_acc.item()

def run_analysis(disease_name, img_tensor):
    model_path = get_model_path(disease_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for disease: {disease_name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        class_label = 'positive' if predicted.item() == 1 else 'negative'
    
    return class_label, confidence.item()

def generate_gradcam(model, img_tensor, target_layer):
    model.eval()
    img_tensor = img_tensor.to(next(model.parameters()).device)

    feature_extractor = model.features
    classifier = model.classifier

    features = feature_extractor(img_tensor)
    output = classifier(features.view(features.size(0), -1))

    pred_index = output.argmax(dim=1).item()

    target = output[0][pred_index]
    target.backward()

    gradients = model.features[-1].grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap)

    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return Image.fromarray(superimposed_img)

def evaluate_model(disease_name, test_data):
    model_path = get_model_path(disease_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for disease: {disease_name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = []
    labels = []
    for class_name, image_paths in test_data.items():
        for image_path in image_paths:
            images.append(image_path)
            labels.append(1 if class_name == 'positive' else 0)

    test_dataset = DiseaseDataset(images, labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }