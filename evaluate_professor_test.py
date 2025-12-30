"""
Evaluate model on professor's official test data
EiDL_TestData_WiSe25
"""

import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
TEST_DATA_PATH = 'EiDL_TestData_WiSe25'
MODEL_PATH = 'best_model.pth'
CLASS_NAMES = ['Cooking Pot', 'Cup', 'Knife', 'Pen']  # Match folder names exactly

def load_model(model_path, num_classes=4, device='cpu'):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_test_transform():
    """Get transform for test images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def evaluate_on_professor_data():
    """Evaluate model on professor's test data"""
    
    print("\n" + "=" * 70)
    print("EVALUATION ON PROFESSOR'S OFFICIAL TEST DATA")
    print("=" * 70)
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"\n❌ Error: Test data folder '{TEST_DATA_PATH}' not found!")
        print("Please make sure the professor's test data is in the correct location.")
        return
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"Loading model from '{MODEL_PATH}'...")
    try:
        model = load_model(MODEL_PATH, len(CLASS_NAMES), device)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using: python src/train.py")
        return
    
    # Load test data
    print(f"\nLoading test data from '{TEST_DATA_PATH}'...")
    transform = get_test_transform()
    
    try:
        test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False
        )
        
        print(f"✓ Test data loaded")
        print(f"  Total images: {len(test_dataset)}")
        print(f"  Classes found: {test_dataset.classes}")
        print(f"  Images per class:")
        
        # Count images per class
        class_counts = {}
        for _, label in test_dataset.samples:
            class_name = test_dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"    - {class_name}: {count} images")
            
    except Exception as e:
        print(f"\n❌ Error loading test data: {e}")
        return
    
    # Evaluate
    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "=" * 70)
    print("RESULTS ON PROFESSOR'S TEST DATA")
    print("=" * 70)
    print(f"\n✓ Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"  Correctly classified: {sum(all_preds == np.array(all_labels))}/{len(all_labels)}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(test_dataset.classes):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_preds)[class_mask] == i).sum() / class_mask.sum()
            correct = (np.array(all_preds)[class_mask] == i).sum()
            total = class_mask.sum()
            print(f"  {class_name:15s}: {class_acc * 100:5.1f}% ({correct}/{total})")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("(Rows = True Class, Columns = Predicted Class)")
    print("\n" + " " * 15 + "  ".join(f"{c[:8]:8s}" for c in test_dataset.classes))
    for i, class_name in enumerate(test_dataset.classes):
        print(f"{class_name:15s}", end="")
        for j in range(len(test_dataset.classes)):
            print(f"  {cm[i][j]:8d}", end="")
        print()
    
    # Detailed classification report
    print("\n" + "-" * 70)
    print("Detailed Classification Report:")
    print("-" * 70)
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=test_dataset.classes,
        digits=3
    ))
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title(f'Confusion Matrix - Professor Test Data\nOverall Accuracy: {accuracy * 100:.2f}%')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    output_file = 'confusion_matrix_professor_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to '{output_file}'")
    plt.close()
    
    # Identify misclassifications
    print("\n" + "-" * 70)
    print("Misclassification Analysis:")
    print("-" * 70)
    
    misclassified = []
    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            true_class = test_dataset.classes[all_labels[i]]
            pred_class = test_dataset.classes[all_preds[i]]
            img_path = test_dataset.samples[i][0]
            misclassified.append((img_path, true_class, pred_class))
    
    if misclassified:
        print(f"\nTotal misclassifications: {len(misclassified)}")
        print("\nMisclassified images:")
        for img_path, true_class, pred_class in misclassified:
            img_name = os.path.basename(img_path)
            print(f"  {img_name:25s} | True: {true_class:12s} | Predicted: {pred_class:12s}")
    else:
        print("\n🎉 Perfect! No misclassifications!")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Final Accuracy on Professor's Test Data: {accuracy * 100:.2f}%")
    print(f"✓ Confusion matrix saved: {output_file}")
    print("\n")

if __name__ == "__main__":
    evaluate_on_professor_data()
