import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from efficientnet_pytorch import EfficientNet

def load_model(model_path, num_classes=2):
    # Load the EfficientNet model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    
    # Load the state dict with safe loading
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_confusion_matrix(model_path, test_data_path, output_path, model_name):
    # Load the model
    model = load_model(model_path)
    
    # Prepare the test dataset with correct transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Make predictions
    all_preds = []
    all_labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f'confusion_matrix_{model_name}.png'))
    plt.close()

    # Calculate metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'class_metrics': []
    }

    # Calculate per-class metrics
    for i in range(cm.shape[0]):
        precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['class_metrics'].append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })

    return metrics

# Paths configuration
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(base_path, 'saved_models')
plots_path = os.path.join(base_path, 'plots')

# Create plots directory if it doesn't exist
os.makedirs(plots_path, exist_ok=True)

# Model paths
models = {
    'brain_tumor': os.path.join(models_path, 'brain_tumor_best_model.pth'),
    'cancer': os.path.join(models_path, 'cancer_best_model.pth')
}

# Test data paths - Update these with your actual test data paths
test_data = {
    'brain_tumor': os.path.join(base_path, 'test_data', 'brain_tumor'),
    'cancer': os.path.join(base_path, 'test_data', 'cancer')
}

def main():
    print("Starting confusion matrix generation...")
    
    # First, verify paths exist
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            continue
            
        if not os.path.exists(test_data[model_name]):
            print(f"Warning: Test data directory not found at {test_data[model_name]}")
            continue
            
        print(f"\nGenerating confusion matrix for {model_name} model...")
        try:
            metrics = generate_confusion_matrix(
                model_path=model_path,
                test_data_path=test_data[model_name],
                output_path=plots_path,
                model_name=model_name
            )
            
            print(f"\nResults for {model_name.replace('_', ' ').title()} Model:")
            print("Confusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
            for class_metric in metrics['class_metrics']:
                print(f"\nClass {class_metric['class']}:")
                print(f"Precision: {class_metric['precision']:.4f}")
                print(f"Recall: {class_metric['recall']:.4f}")
                print(f"F1-score: {class_metric['f1_score']:.4f}")
                
        except Exception as e:
            print(f"Error processing {model_name} model: {str(e)}")

if __name__ == "__main__":
    main()