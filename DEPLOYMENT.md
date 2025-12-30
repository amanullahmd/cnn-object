# Railway Deployment Guide

This guide explains how to deploy the CNN Image Classifier to Railway.

## Prerequisites

- Railway account (https://railway.app)
- GitHub account with this repository
- Git installed locally

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository is clean and ready:

```bash
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. Connect to Railway

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Authorize Railway to access your GitHub account
5. Select the `cnn-object` repository

### 3. Configure Environment

Railway will automatically detect the `Dockerfile` and deploy accordingly.

**Important**: Make sure `best_model.pth` is in your repository root before deploying.

### 4. Set Up Variables (if needed)

In Railway dashboard:
- Go to your project settings
- Add any environment variables if required
- The app will use default settings from the code

### 5. Deploy

Railway will automatically:
1. Build the Docker image
2. Install dependencies from `requirements.txt`
3. Copy the model file
4. Start the application

### 6. Access Your Application

Once deployed, Railway will provide you with a URL like:
```
https://your-app-name.railway.app
```

## Important Notes

### Model File Size

The `best_model.pth` file is ~45 MB. Make sure:
- It's included in your repository
- It's NOT in `.gitignore` (we want to deploy it)
- Railway has enough storage (it does - 10GB default)

### Large Dataset Folders

The following folders are in `.gitignore` and won't be pushed:
- `EiDL_CNN_WiSe_25_26/` (original dataset)
- `EiDL_TestData_WiSe25/` (test data)
- `dataset/` (processed dataset)

This keeps your repository size manageable.

### Path Handling

The updated `predict_gui.py` automatically finds `best_model.pth` in:
1. Project root (default for Railway)
2. Current working directory
3. Relative paths

This makes it compatible with both local and Railway deployments.

## Troubleshooting

### Model Not Found

If you get "Model not found" error:
1. Verify `best_model.pth` exists in repository root
2. Check file size is ~45 MB
3. Ensure it's not in `.gitignore`
4. Redeploy after pushing the model file

### Build Fails

If the Docker build fails:
1. Check `requirements.txt` has all dependencies
2. Verify Python version compatibility (3.10)
3. Check Railway logs for specific errors
4. Ensure all source files are present

### Application Crashes

Check Railway logs:
1. Go to your project in Railway dashboard
2. Click "Logs" tab
3. Look for error messages
4. Common issues:
   - Missing dependencies
   - Model file not found
   - Port configuration issues

## Local Testing Before Deployment

Test locally first:

```bash
# Build Docker image
docker build -t cnn-classifier .

# Run container
docker run -it cnn-classifier
```

## Updating the Model

To deploy a new trained model:

1. Replace `best_model.pth` locally
2. Commit and push:
   ```bash
   git add best_model.pth
   git commit -m "Update trained model"
   git push origin main
   ```
3. Railway will automatically redeploy with the new model

## Performance Considerations

- **GPU**: Railway provides GPU options (paid tier)
- **Memory**: Default 512MB should be sufficient
- **Startup Time**: First request may take 10-30 seconds (cold start)

## Cost

Railway offers:
- Free tier: $5/month credit
- Pay-as-you-go: $0.000463/hour for compute

For a simple GUI app, you'll likely stay within free tier.

## Support

- Railway Docs: https://docs.railway.app
- Railway Community: https://railway.app/community
- GitHub Issues: Create an issue in your repository
