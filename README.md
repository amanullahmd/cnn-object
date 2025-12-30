# CNN Image Classifier for Household Objects

This project implements a Convolutional Neural Network to classify four types of household objects: cooking pots, cups, knives, and pens. It was developed as part of the "Einführung ins Deep Learning" course (Übungsblatt 2) at Technische Hochschule Mittelhessen during Winter Semester 2025/26.

## Project Overview

The classifier uses transfer learning with a pretrained ResNet18 model, fine-tuned on a custom dataset of household objects. After training for 15 epochs, the model achieved an overall accuracy of 93.4% on the test set.

## Results

The model performs well across all four classes, with particularly strong results on cooking pots and cups:

| Object Class | Test Accuracy |
|-------------|---------------|
| Cooking Pot | 98.4%         |
| Cup         | 96.8%         |
| Pen         | 94.6%         |
| Knife       | 93.1%         |

**Overall Test Accuracy: 93.4%**

The confusion matrix (included as `confusion_matrix.png`) shows that the main challenge for the model is distinguishing between knives and pens, which makes sense given their similar elongated shapes. However, even this confusion is minimal, with the model maintaining high accuracy across all classes.

## Getting Started

### Prerequisites

You'll need Python 3.8 or higher. All required packages are listed in `requirements.txt`.

### Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset Structure

The original dataset should be organized in the `EiDL_CNN_WiSe_25_26` folder with subdirectories for each class:

```
EiDL_CNN_WiSe_25_26/
├── cooking_pot/
├── cup/
├── knife/
└── pen/
```

Each subdirectory should contain at least 100 images of that object class.

## Training the Model

To train the model from scratch:

```bash
python src/train.py
```

The training script will:
- Automatically split your dataset into 80% training and 20% testing
- Apply data augmentation to the training images (random flips, rotations, color adjustments)
- Train for 15 epochs using the Adam optimizer
- Save the best performing model as `best_model.pth`
- Generate a confusion matrix and sample predictions

Training takes approximately 15-20 minutes on a GPU, or longer on CPU.

## Using the Trained Model

### Option 1: Predict a Single Image

```bash
python src/predict.py path/to/your/image.jpg
```

This will display the image with the predicted class and confidence score.

### Option 2: GUI Application

```bash
python src/predict_gui.py
```

This opens a graphical interface where you can:
- Click "Upload Image" to select an image file
- See the prediction with confidence score
- Try multiple images easily

## Project Structure

```
übung2/
├── src/
│   ├── train.py              # Main training script
│   ├── predict.py            # Command-line prediction
│   ├── predict_gui.py        # GUI for predictions
│   └── dataset_utils.py      # Dataset handling utilities
├── EiDL_CNN_WiSe_25_26/      # Original dataset
├── dataset/                   # Auto-generated train/test split
├── runs/                      # TensorBoard training logs
├── best_model.pth            # Trained model weights (~45 MB)
├── confusion_matrix.png      # Evaluation results
├── sample_predictions.png    # Example predictions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### Model Architecture

The classifier is based on ResNet18, a proven architecture for image classification. I used transfer learning by:
1. Loading weights pretrained on ImageNet
2. Replacing the final layer to output 4 classes instead of 1000
3. Fine-tuning all layers on our custom dataset

### Training Configuration

- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32 images
- **Epochs**: 15
- **Data Augmentation**: Random horizontal flips, rotations (±20°), color jitter, and affine transformations
- **Image Size**: 224×224 pixels (ResNet18 standard)

### Data Augmentation

During training, images are randomly transformed to help the model generalize better:
- Horizontal flips (50% chance)
- Rotations up to 20 degrees
- Color adjustments (brightness, contrast, saturation)
- Small affine transformations

Test images are only resized and normalized, without augmentation.

## Observations and Challenges

### What Worked Well

The model performs exceptionally well on cooking pots and cups, achieving over 96% accuracy. These objects have distinctive shapes that the model learned to recognize reliably.

### Main Challenge

The most common confusion is between knives and pens (15 total misclassifications in the test set). This makes sense because:
- Both are elongated objects
- They can have similar colors (black, silver)
- The viewing angle significantly affects their appearance

Despite this challenge, the model still maintains over 93% accuracy on both classes.

### Potential Improvements

To further improve the model, you could:
- Collect more diverse images of knives and pens from different angles
- Use a deeper network like ResNet50
- Implement more aggressive data augmentation
- Add attention mechanisms to focus on distinctive features

## Requirements

Main dependencies (see `requirements.txt` for complete list):
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.7.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.2.0
- Pillow >= 9.0.0

## Notes

- The trained model file (`best_model.pth`) is approximately 45 MB
- For best results, use clear, well-lit images with the object clearly visible
- Training logs can be viewed with TensorBoard: `tensorboard --logdir=runs`

## Assignment Completion

This project fulfills all requirements of Übungsblatt 2:
- ✓ Custom dataset with 100+ images per class
- ✓ CNN implementation using transfer learning
- ✓ Training with data augmentation
- ✓ Comprehensive evaluation with confusion matrix
- ✓ Identification of corner cases (knife/pen confusion)
- ✓ Interactive prediction tools (GUI and command-line)
- ✓ Achieved >90% accuracy (93.4%)

---

**Course**: Einführung ins Deep Learning  
**Semester**: WiSe 25/26  
**Institution**: Technische Hochschule Mittelhessen (THM) Giessen
