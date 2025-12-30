"""
CNN Image Classifier - GUI Version with File Upload
Interactive GUI for image classification
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
import torch.nn as nn
import os


class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.device = None
        self.transform = None
        self.class_names = ['cooking_pot', 'cup', 'knife', 'pen']
        self.current_image = None
        
        # Create GUI
        self.create_widgets()
        
        # Load model
        self.load_model()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title_label = tk.Label(
            self.root,
            text="CNN Image Classifier",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Upload an image to classify: cup, knife, pen, or cooking pot",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666'
        )
        subtitle_label.pack(pady=5)
        
        # Upload button
        self.upload_btn = tk.Button(
            self.root,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3
        )
        self.upload_btn.pack(pady=20)
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Image label
        self.image_label = tk.Label(
            self.image_frame,
            text="No image uploaded",
            font=("Arial", 12),
            bg='white',
            fg='#999'
        )
        self.image_label.pack(expand=True)
        
        # Result frame
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Result label
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        self.result_label.pack()
        
        # Confidence label
        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666'
        )
        self.confidence_label.pack()
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10),
            bg='#e0e0e0',
            fg='#333',
            anchor='w',
            relief=tk.SUNKEN
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.status_label.config(text="Loading model...")
            self.root.update()
            
            # Device configuration
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model - check multiple possible locations (Railway deployment compatible)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            possible_paths = [
                os.path.join(project_root, 'best_model.pth'),  # Default: project root
                os.path.join(os.getcwd(), 'best_model.pth'),   # Current working directory
                'best_model.pth',                               # Relative path
                os.path.join(script_dir, '..', 'best_model.pth')  # Relative to script
            ]
            
            model_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    model_path = abs_path
                    break
            
            if model_path is None:
                messagebox.showwarning(
                    "Warning",
                    "Model not loaded! Please check if best_model.pth exists.\n\n"
                    "The model might still be training. Wait for training to complete,\n"
                    "then restart the GUI.\n\n"
                    "Expected location: " + os.path.join(project_root, 'best_model.pth')
                )
                self.status_label.config(text="Model not found!")
                return
            
            # Load model architecture
            self.model = models.resnet18(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_names))
            
            # Load weights with error handling
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as load_error:
                messagebox.showerror(
                    "Error",
                    f"Failed to load model weights:\n{str(load_error)}\n\n"
                    "The model file might be corrupted or still being written.\n"
                    "If training is in progress, please wait for it to complete."
                )
                self.status_label.config(text="Model loading failed!")
                return
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            self.status_label.config(text=f"Model loaded successfully on {device_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_label.config(text="Model loading failed!")
    
    def upload_image(self):
        """Open file dialog to upload image"""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded! Please check if best_model.pth exists.")
            return
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            self.status_label.config(text="Processing image...")
            self.root.update()
            
            # Load and display image
            img = Image.open(file_path).convert('RGB')
            self.current_image = img
            
            # Display image (resize for display)
            display_img = img.copy()
            display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Classify image
            self.classify_image(img)
            
            self.status_label.config(text=f"Classified: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            self.status_label.config(text="Error processing image")
    
    def classify_image(self, img):
        """Classify the uploaded image"""
        try:
            # Preprocess
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
                class_name = self.class_names[pred.item()]
                conf_value = confidence.item()
            
            # Display results
            self.result_label.config(
                text=f"Prediction: {class_name.upper()}",
                fg='#2196F3'
            )
            
            confidence_text = f"Confidence: {conf_value:.1%}"
            confidence_color = '#4CAF50' if conf_value > 0.8 else '#FF9800' if conf_value > 0.5 else '#F44336'
            
            self.confidence_label.config(
                text=confidence_text,
                fg=confidence_color
            )
            
            # Show all probabilities
            all_probs = probabilities[0].cpu().numpy()
            prob_text = "\n\nAll Probabilities:\n"
            for i, (name, prob) in enumerate(zip(self.class_names, all_probs)):
                prob_text += f"{name}: {prob:.1%}\n"
            
            print(prob_text)  # Print to console
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")


def main():
    """Main function"""
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
