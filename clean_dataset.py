"""
Clean corrupted images from dataset
"""
import os
from PIL import Image

def clean_directory(directory):
    """Remove corrupted images from directory"""
    removed = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                try:
                    # Try to actually load and convert the image
                    with Image.open(filepath) as img:
                        img.verify()  # Verify it's a valid image
                    # Re-open and actually load the data
                    with Image.open(filepath) as img:
                        img.load()  # Force load the image data
                        img.convert('RGB')  # Try to convert to RGB
                except Exception as e:
                    print(f"Removing corrupted: {filepath} ({str(e)[:50]})")
                    os.remove(filepath)
                    removed.append(filepath)
    return removed

print("Cleaning train dataset...")
removed_train = clean_directory('dataset/train')

print("\nCleaning test dataset...")
removed_test = clean_directory('dataset/test')

print(f"\n✓ Removed {len(removed_train)} corrupted images from train")
print(f"✓ Removed {len(removed_test)} corrupted images from test")
print(f"✓ Total removed: {len(removed_train) + len(removed_test)}")
