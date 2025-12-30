# Deployment Checklist

## Before Pushing to GitHub

- [ ] `best_model.pth` exists in project root
- [ ] `best_model.pth` is ~45 MB in size
- [ ] `.gitignore` is configured correctly
- [ ] Large folders (EiDL_CNN_WiSe_25_26, dataset) are NOT tracked
- [ ] README.md displays correctly (no broken text)
- [ ] All source files in `src/` are present
- [ ] `requirements.txt` has all dependencies

### Verify with:
```bash
# Check model file
ls -lh best_model.pth

# Check what will be pushed
git status

# Verify large folders are ignored
git check-ignore -v EiDL_CNN_WiSe_25_26/
git check-ignore -v dataset/
```

---

## GitHub Push

- [ ] Commit all changes
- [ ] Push to main branch
- [ ] Verify on GitHub.com that files look correct

### Commands:
```bash
git add .
git commit -m "Fix README, prepare for Railway deployment"
git push origin main
```

---

## Railway Deployment

- [ ] Railway account created (https://railway.app)
- [ ] GitHub connected to Railway
- [ ] New project created from repository
- [ ] Dockerfile detected automatically
- [ ] Build completes without errors
- [ ] Deployment succeeds
- [ ] App URL provided

### Steps:
1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Choose `cnn-object` repository
5. Wait for build and deployment

---

## Post-Deployment Testing

- [ ] Railway URL is accessible
- [ ] GUI loads in browser
- [ ] "Upload Image" button works
- [ ] Can select and upload test image
- [ ] Prediction displays correctly
- [ ] Confidence score shows
- [ ] No errors in Railway logs

### Test with:
1. Open Railway URL in browser
2. Click "Upload Image"
3. Select a test image
4. Verify prediction appears
5. Check Railway logs for errors

---

## Troubleshooting Checklist

If deployment fails, check:

- [ ] `best_model.pth` is in repository root
- [ ] File is not corrupted (size ~45 MB)
- [ ] `Dockerfile` is in project root
- [ ] `requirements.txt` has all dependencies
- [ ] Python version is 3.10 (in Dockerfile)
- [ ] No syntax errors in Python files
- [ ] Railway logs show specific error message

### Check Railway Logs:
1. Go to your project in Railway dashboard
2. Click "Logs" tab
3. Look for error messages
4. Search for "ERROR" or "FAILED"

---

## File Verification

### Required Files Present:
- [ ] `src/predict_gui.py` (updated for Railway)
- [ ] `src/predict.py`
- [ ] `src/train.py`
- [ ] `src/dataset_utils.py`
- [ ] `best_model.pth` (~45 MB)
- [ ] `requirements.txt`
- [ ] `README.md` (fixed)
- [ ] `Dockerfile` (new)
- [ ] `railway.json` (new)
- [ ] `.gitignore` (new)

### Documentation Files:
- [ ] `README.md` - Project overview
- [ ] `DEPLOYMENT.md` - Detailed deployment guide
- [ ] `QUICKSTART.md` - Quick reference
- [ ] `CHANGES.md` - What was changed
- [ ] `DEPLOYMENT_CHECKLIST.md` - This file

---

## Repository Size Check

```bash
# Check total size
du -sh .

# Check largest files
du -sh * | sort -rh | head -10

# Verify large folders are ignored
git ls-files | grep -E "(EiDL_CNN|dataset)" | wc -l
# Should return 0 if properly ignored
```

---

## Final Verification

Before considering deployment complete:

1. **Local Test**
   ```bash
   python src/predict_gui.py
   # Should open GUI without errors
   ```

2. **Docker Test**
   ```bash
   docker build -t cnn-classifier .
   docker run -it cnn-classifier
   # Should build and run without errors
   ```

3. **GitHub Check**
   - Visit https://github.com/amanullahmd/cnn-object
   - Verify files are there
   - Check README displays correctly
   - Confirm large folders are not present

4. **Railway Check**
   - Visit your Railway URL
   - Test image upload
   - Verify predictions work
   - Check logs for errors

---

## Success Indicators

✓ **Deployment is successful when:**
- Railway URL is accessible
- GUI loads without errors
- Image upload works
- Predictions display correctly
- No errors in Railway logs
- Model file is loaded (~45 MB)
- All 4 classes can be predicted

---

## Rollback Plan

If something goes wrong:

1. **Local Issue**: 
   - Revert changes: `git reset --hard HEAD~1`
   - Fix the issue
   - Recommit and push

2. **Railway Issue**:
   - Go to Railway dashboard
   - Click "Deployments"
   - Select previous successful deployment
   - Click "Redeploy"

3. **Model Issue**:
   - Verify `best_model.pth` is not corrupted
   - Check file size is ~45 MB
   - Retrain if necessary
   - Push new model and redeploy

---

## Support Resources

- **Railway Docs**: https://docs.railway.app
- **Railway Community**: https://railway.app/community
- **PyTorch Docs**: https://pytorch.org/docs
- **GitHub Help**: https://docs.github.com

---

## Notes

- Deployment typically takes 5-10 minutes
- First request may take 10-30 seconds (cold start)
- Model file is ~45 MB (included in deployment)
- Free tier should be sufficient for this app
- Check Railway logs for any issues

---

**Last Updated**: December 30, 2025  
**Status**: Ready for Deployment ✓
