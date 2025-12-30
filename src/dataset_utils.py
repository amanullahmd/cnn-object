"""
Dataset utilities for CNN Image Classifier
Handles dataset splitting, loading, and preprocessing
"""

import os
import shutil
import random
from typing import Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def split_dataset(
    source_dir: str = 'EiDL_CNN_WiSe_25_26',
    train_dir: str = 'dataset/train',
    test_dir: str = 'dataset/test',
    split_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Automatically split dataset into training and testing sets.
    
    Args:
        source_dir: Directory containing class subdirectories with images
        train_dir: Output directory for training images
        test_dir: Output directory for testing images
        split_ratio: Ratio of images to use for training (default: 0.8)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_count, test_count)
        
    Raises:
        FileNotFoundError: If source_dir doesn't exist
        ValueError: If no valid class directories found
    """
    # Validate source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Source directory '{source_dir}' does not exist. "
            f"Please ensure your dataset is in the correct location."
        )
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    total_train = 0
    total_test = 0
    class_count = 0
    
    print(f"\nSplitting dataset from '{source_dir}'...")
    print(f"Train/Test ratio: {split_ratio:.0%}/{(1-split_ratio):.0%}")
    print("-" * 50)
    
    # Iterate through class directories
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Get all image files
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(images) == 0:
            print(f"⚠️  Warning: No images found in '{class_name}'")
            continue
        
        # Warn if insufficient images
        if len(images) < 100:
            print(f"⚠️  Warning: Class '{class_name}' has only {len(images)} images (recommended: ≥100)")
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split index
        split_index = int(len(images) * split_ratio)
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy training images
        for img in images[:split_index]:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy testing images
        for img in images[split_index:]:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
        
        train_count = split_index
        test_count = len(images) - split_index
        total_train += train_count
        total_test += test_count
        class_count += 1
        
        print(f"✓ {class_name:15s}: {train_count:4d} train, {test_count:4d} test (total: {len(images):4d})")
    
    if class_count == 0:
        raise ValueError(
            f"No valid class directories found in '{source_dir}'. "
            f"Expected directories with image files."
        )
    
    print("-" * 50)
    print(f"Total: {total_train} train, {total_test} test images across {class_count} classes")
    print(f"✓ Dataset split complete!\n")
    
    return total_train, total_test


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for training and testing.
    
    Args:
        image_size: Target image size (default: 224 for ResNet18)
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Test transforms without augmentation
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform_train, transform_test


def create_dataloaders(
    train_dir: str = 'dataset/train',
    test_dir: str = 'dataset/test',
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    """
    Create DataLoader instances for training and testing.
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing testing images
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset)
        
    Raises:
        FileNotFoundError: If train_dir or test_dir doesn't exist
    """
    # Validate directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory '{train_dir}' does not exist. "
            f"Please run dataset splitting first."
        )
    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Testing directory '{test_dir}' does not exist. "
            f"Please run dataset splitting first."
        )
    
    # Get transforms
    transform_train, transform_test = get_transforms()
    
    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}\n")
    
    return train_loader, test_loader, train_dataset, test_dataset
