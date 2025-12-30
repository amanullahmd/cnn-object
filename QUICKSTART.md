# Quick Start Guide

## For Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the GUI
```bash
python src/predict_gui.py
```

### 3. Upload an Image
- Click "📁 Upload Image"
- Select a JPG, PNG, or other image format
- See the prediction and confidence score

---

## For Railway Deployment

### 1. Prepare Repository
```bash
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. Deploy to Railway
1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Choose your `cnn-object` repository
5. Railway auto-deploys using the Dockerfile

### 3. Access Your App
- Railway provides a URL (e.g., `https://your-app.railway.app`)
- Open it in your browser
- Use the GUI to classify images

---

## Troubleshooting

### Model Not Found
- Ensure `best_model.pth` exists in project root
- File should be ~45 MB
- Check it's not in `.gitignore`

### GUI Won't Start
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Try: `python -m src.predict_gui`

### Railway Deployment Fails
- Check logs in Railway dashboard
- Verify `best_model.pth` is in repository
- Ensure Dockerfile is in project root
- See `DEPLOYMENT.md` for detailed troubleshooting

---

## Project Structure

```
cnn-object/
├── src/
│   ├── train.py           # Training script
│   ├── predict.py         # CLI prediction
│   ├── predict_gui.py     # GUI application
│   └── dataset_utils.py   # Utilities
├── best_model.pth         # Trained model (~45 MB)
├── requirements.txt       # Dependencies
├── Dockerfile             # For Railway
├── railway.json           # Railway config
├── README.md              # Project info
├── DEPLOYMENT.md          # Deployment guide
├── CHANGES.md             # What was changed
└── QUICKSTART.md          # This file
```

---

## What the Model Does

Classifies images into 4 categories:
- 🍲 **Cooking Pot** (98.4% accuracy)
- ☕ **Cup** (96.8% accuracy)
- 🔪 **Knife** (93.1% accuracy)
- ✏️ **Pen** (94.6% accuracy)

**Overall Accuracy: 93.4%**

---

## Key Files Explained

| File | Purpose |
|------|---------|
| `src/predict_gui.py` | Main GUI application |
| `best_model.pth` | Trained neural network weights |
| `requirements.txt` | Python package dependencies |
| `Dockerfile` | Container configuration for Railway |
| `README.md` | Full project documentation |
| `DEPLOYMENT.md` | Detailed deployment instructions |

---

## Common Commands

```bash
# Run GUI locally
python src/predict_gui.py

# Predict single image
python src/predict.py path/to/image.jpg

# Train new model (requires dataset)
python src/train.py

# Build Docker image locally
docker build -t cnn-classifier .

# Run Docker container locally
docker run -it cnn-classifier
```

---

## Next Steps

1. **Test locally**: Run `python src/predict_gui.py`
2. **Push to GitHub**: `git push origin main`
3. **Deploy to Railway**: Follow DEPLOYMENT.md
4. **Share your app**: Get the Railway URL and share it!

---

## Need Help?

- **Local issues**: Check `README.md` and `DEPLOYMENT.md`
- **Railway issues**: See DEPLOYMENT.md troubleshooting section
- **Model questions**: See README.md Technical Details section
- **Code questions**: Check comments in `src/` files

---

**Happy classifying! 🎉**
