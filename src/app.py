"""
CNN Image Classifier - Flask Web API
Web interface for image classification
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'

# Model variables
model = None
device = None
transform = None
class_names = ['cooking_pot', 'cup', 'knife', 'pen']

def load_model():
    """Load the trained model"""
    global model, device, transform
    
    try:
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find model file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        possible_paths = [
            os.path.join(project_root, 'best_model.pth'),
            os.path.join(os.getcwd(), 'best_model.pth'),
            'best_model.pth',
            os.path.join(script_dir, '..', 'best_model.pth')
        ]
        
        model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                model_path = abs_path
                break
        
        if model_path is None:
            print("ERROR: Model file not found!")
            return False
        
        # Load model architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(class_names))
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False

def classify_image(img):
    """Classify the image"""
    try:
        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            
            class_name = class_names[pred.item()]
            conf_value = confidence.item()
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        prob_dict = {name: float(prob) for name, prob in zip(class_names, all_probs)}
        
        return {
            'prediction': class_name,
            'confidence': conf_value,
            'probabilities': prob_dict,
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for classification"""
    
    if model is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'success': False}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False}), 400
    
    try:
        # Load image
        img = Image.open(file.stream).convert('RGB')
        
        # Classify
        result = classify_image(img)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        print("WARNING: Model failed to load. API will not work properly.")
    
    # Get port from environment or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
