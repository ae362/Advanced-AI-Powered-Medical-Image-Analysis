import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import cv2
from .preprocessing import preprocess_for_model, create_brain_mask, generate_masked_gradcam
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class BrainFocusedModel(nn.Module):
    def __init__(self, num_classes=1):
        super(BrainFocusedModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.attention1 = AttentionGate(1280, 112, 56)
        self.attention2 = AttentionGate(1280, 192, 96)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280 + 112 + 192, num_classes)

    def forward(self, x):
        features = self.efficientnet.features(x)
        
        att1 = self.attention1(features, self.efficientnet.features[4](x))
        att2 = self.attention2(features, self.efficientnet.features[6](x))
        
        x = self.avgpool(features)
        att1 = self.avgpool(att1)
        att2 = self.avgpool(att2)
        
        x = torch.cat([x, att1, att2], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_brain_focused_model(num_classes=1):
    return BrainFocusedModel(num_classes)

def compile_model(model):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion

def train_model_with_masking(train_data, train_labels, validation_data=None, epochs=50, batch_size=32):
    logger.info("Starting model training")
    logger.info(f"Training data size: {len(train_data)}")
    
    if not train_data:
        raise ValueError("No training data provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Preprocess data
    X_processed = []
    masks = []
    for image_path in train_data:
        try:
            img_input, mask = preprocess_for_model(image_path)
            X_processed.append(img_input[0])
            masks.append(mask)
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {str(e)}")
            continue

    if not X_processed:
        raise ValueError("No valid images found in training data")

    X_processed = np.array(X_processed)
    y = np.array(train_labels)

    logger.info(f"Processed {len(X_processed)} images successfully")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Split data if validation_data not provided
    if validation_data is None:
        X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X_processed, y
        X_val, y_val = validation_data

    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create and compile model
    model = create_brain_focused_model()
    model, optimizer, criterion = compile_model(model)
    model = model.to(device)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info("Saved best model")

    return model, masks

def analyze_image(model, image_path, confidence_threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image_input, brain_mask = preprocess_for_model(image_path)
    image_tensor = torch.FloatTensor(image_input).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()

    gradcam = generate_masked_gradcam(model, image_tensor, brain_mask)

    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, (224, 224))

    heatmap = np.uint8(255 * gradcam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    visualization = cv2.addWeighted(
        cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR),
        0.7,
        heatmap,
        0.3,
        0
    )

    visualization = visualization * np.expand_dims(brain_mask, -1)

    return {
        'prediction': 'positive' if prediction >= confidence_threshold else 'negative',
        'confidence': float(prediction),
        'visualization': visualization,
        'attention_map': gradcam
    }

