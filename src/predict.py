"""
CNN Image Classifier - File-based Prediction
Predict class for images from files
"""

import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


def load_model(model_path='best_model.pth', num_classes=4, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_prediction_transform():
    """
    Get transform for prediction (same as test transform).
    
    Returns:
        transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(image_path, model, transform, class_names, device):
    """
    Predict class for a single image.
    
    Args:
        image_path: Path to image file
        model: Trained model
        transform: Image transform
        class_names: List of class names
        device: Device to use
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            class_name = class_names[pred.item()]
            conf_value = confidence.item()
        
        return class_name, conf_value, img
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")


def display_prediction(img, predicted_class, confidence):
    """
    Display image with prediction.
    
    Args:
        img: PIL Image
        predicted_class: Predicted class name
        confidence: Prediction confidence
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict image class')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['cooking_pot', 'cup', 'knife', 'pen'],
                        help='Class names')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CNN Image Classifier - Prediction")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from '{args.model}'...")
    try:
        model = load_model(args.model, len(args.classes), device)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Model file '{args.model}' not found")
        print("Please train the model first using train.py")
        return
    
    # Get transform
    transform = get_prediction_transform()
    
    # Predict
    print(f"\nPredicting class for '{args.image_path}'...")
    try:
        predicted_class, confidence, img = predict_image(
            args.image_path, model, transform, args.classes, device
        )
        
        print(f"\n✓ Prediction: {predicted_class}")
        print(f"✓ Confidence: {confidence:.2%}")
        
        # Display result
        display_prediction(img, predicted_class, confidence)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
