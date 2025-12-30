"""
CNN Image Classifier Training Script
Classes: cooking_pot, cup, knife, pen
University Project: Übungsblatt 2 - Einführung ins Deep Learning
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from dataset_utils import split_dataset, create_dataloaders


def get_device():
    """
    Get the appropriate device (CUDA or CPU).
    
    Returns:
        torch.device: Selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
    return device


def create_model(num_classes=4, device='cpu'):
    """
    Create ResNet18 model with custom final layer.
    
    Args:
        num_classes: Number of output classes
        device: Device to move model to
        
    Returns:
        nn.Module: Configured ResNet18 model
    """
    print("\nInitializing ResNet18 model...")
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Enable fine-tuning for all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Move to device
    model = model.to(device)
    
    print(f"✓ Model initialized with {num_classes} output classes")
    print(f"✓ All layers enabled for fine-tuning")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, test_loader, device):
    """
    Validate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        float: Validation accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train_model(model, train_loader, test_loader, device, epochs=15, lr=0.0001):
    """
    Complete training loop with validation and checkpointing.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        nn.Module: Trained model
    """
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Try to create TensorBoard writer, skip if it fails
    writer = None
    try:
        import os
        log_dir = os.path.abspath('runs/4class_resnet18')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"✓ TensorBoard logging enabled: {log_dir}")
    except Exception as e:
        print(f"⚠️  TensorBoard logging disabled (path issue with special characters)")
        writer = None
    
    best_val_acc = 0.0
    
    print(f"Optimizer: Adam (lr={lr})")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Epochs: {epochs}")
    print(f"TensorBoard logs: runs/4class_resnet18")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_acc = validate(model, test_loader, device)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Val Acc: {val_acc:5.2f}%", end="")
        
        # TensorBoard logging (if available)
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(" ← Best model saved!")
        else:
            print()
    
    if writer is not None:
        writer.close()
    
    print("-" * 60)
    print(f"✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)
    
    return model


def evaluate_model(model, test_loader, test_dataset, device):
    """
    Evaluate model and generate visualizations.
    
    Args:
        model: Trained neural network model
        test_loader: Test data loader
        test_dataset: Test dataset
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_acc = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("✓ Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()
    
    # Show sample predictions
    show_predictions(model, test_loader, test_dataset, device)
    
    print("=" * 60)


def show_predictions(model, test_loader, test_dataset, device, num_images=6):
    """
    Display sample predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        test_dataset: Test dataset
        device: Device to use
        num_images: Number of images to display
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(15, 6))
    for idx in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, 3, idx+1, xticks=[], yticks=[])
        
        # Denormalize image
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        true_label = test_dataset.classes[labels[idx]]
        pred_label = test_dataset.classes[preds[idx]]
        color = 'green' if labels[idx] == preds[idx] else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    print("✓ Sample predictions saved to 'sample_predictions.png'")
    plt.show()


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train CNN Image Classifier')
    parser.add_argument('--skip-split', action='store_true',
                        help='Skip dataset splitting (use existing train/test folders)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CNN Image Classifier - Training Pipeline")
    print("Übungsblatt 2 - Einführung ins Deep Learning")
    print("=" * 60)
    
    # Device configuration
    device = get_device()
    
    # Step 1: Split dataset (if needed)
    if not args.skip_split:
        try:
            split_dataset()
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            return
    else:
        print("\n⏭️  Skipping dataset split (using existing folders)")
    
    # Step 2: Create dataloaders
    try:
        train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
            batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Step 3: Create model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes=num_classes, device=device)
    
    # Step 4: Train model
    model = train_model(
        model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr
    )
    
    # Step 5: Evaluate model
    evaluate_model(model, test_loader, test_dataset, device)
    
    print("\n✓ Training pipeline complete!")
    print(f"✓ Best model saved to 'best_model.pth'")
    print(f"✓ View training logs: tensorboard --logdir=runs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
