# Deploying to Streamlit Cloud

## 📋 Prerequisites

Before deploying to Streamlit Cloud, you need:

1. ✅ Trained model file: `best_fraud_detector.pth`
2. ✅ Transaction data: `fraud_transactions_YYYYMMDD_HHMMSS.pkl` (or `.csv`)
3. ✅ All Python files in your repository
4. ✅ GitHub account
5. ✅ Streamlit Cloud account (free at https://streamlit.io/cloud)

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (if you haven't already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Fraud Detection App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ait-204-ann-data-stream.git
   git push -u origin main
   ```

2. **Make sure these files are in your repo**:
   - `streamlit_app.py`
   - `fraud_model.py`
   - `feature_engineering.py`
   - `requirements.txt`
   - `best_fraud_detector.pth` ⚠️ **IMPORTANT**
   - `fraud_transactions_*.pkl` or `.csv` ⚠️ **IMPORTANT**
   - `.streamlit/config.toml`

### Step 2: Add Large Files with Git LFS (IMPORTANT!)

Model and data files are large, so you need Git LFS:

```bash
# Install Git LFS (if not already installed)
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Add your model and data files
git add best_fraud_detector.pth
git add fraud_transactions_*.pkl

# Commit and push
git commit -m "Add model and data files with LFS"
git push
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to https://share.streamlit.io/**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in the deployment form**:
   - **Repository**: Select your repository (e.g., `YOUR_USERNAME/ait-204-ann-data-stream`)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

5. **Click "Deploy!"**

6. **Wait for deployment** (3-5 minutes)
   - Streamlit will install dependencies from `requirements.txt`
   - It will load your model and data files
   - The app will automatically start

### Step 4: Verify Deployment

Once deployed, you should see:
- ✅ Model loaded successfully
- ✅ 50,000 transactions loaded
- ✅ Interactive dashboard ready

## ⚠️ Troubleshooting

### Problem: "Model file not found"
**Solution**: Make sure `best_fraud_detector.pth` is committed to your repo using Git LFS

### Problem: "No transaction data found"
**Solution**: Make sure at least one `.pkl` or `.csv` file is committed to your repo using Git LFS

### Problem: "File size too large"
**Solution**: 
- Use Git LFS for files over 100MB
- Or compress your data files:
  ```python
  import pickle
  import gzip
  
  # Compress pickle file
  with open('fraud_transactions.pkl', 'rb') as f_in:
      with gzip.open('fraud_transactions.pkl.gz', 'wb') as f_out:
          f_out.writelines(f_in)
  ```
  Then update `streamlit_app.py` to handle `.pkl.gz` files.

### Problem: "Out of memory"
**Solution**: 
- Streamlit Cloud free tier has 1GB RAM
- Consider using a smaller subset of data for demo (e.g., 10,000 transactions)
- Or upgrade to Streamlit Cloud paid tier

### Problem: "Dependencies installation failed"
**Solution**:
- Check `requirements.txt` has correct versions
- Remove any unnecessary dependencies
- Check Streamlit Cloud logs for specific error messages

## 🎯 Alternative: Deploy with Smaller Dataset

If you hit memory limits, create a smaller dataset:

```python
# Create a smaller dataset for demo
import pandas as pd
import pickle

# Load full dataset
df = pd.read_csv('fraud_transactions_20251003_142107.csv')

# Take first 10,000 transactions
df_small = df.head(10000)

# Save
df_small.to_csv('fraud_transactions_demo.csv', index=False)
with open('fraud_transactions_demo.pkl', 'wb') as f:
    pickle.dump(df_small.to_dict('records'), f)
```

Then update `streamlit_app.py` to prefer the demo file or add it as a separate demo mode.

## 📊 Expected App URL

After deployment, your app will be available at:
```
https://YOUR_USERNAME-ait-204-ann-data-stream-streamlit-app-RANDOM.streamlit.app/
```

## 🔧 Updating Your App

To update your deployed app:
```bash
# Make changes to your code
# Commit and push
git add .
git commit -m "Update app"
git push

# Streamlit Cloud will automatically redeploy!
```

## 💡 Tips for Best Performance

1. **Use `.pkl` instead of `.csv`** - Faster loading
2. **Enable caching** - Already implemented with `@st.cache_resource` and `@st.cache_data`
3. **Reduce animation speed** - Use 0.1 seconds instead of 0.5 for faster demo
4. **Close unused tabs** - Streamlit Cloud has limited resources
5. **Monitor usage** - Check your Streamlit Cloud dashboard for resource usage

## 🎓 Sharing Your App

Once deployed, share your app:
- Copy the URL from Streamlit Cloud dashboard
- Share in presentations or reports
- Embed in documentation

The app is public by default, so anyone with the link can view it!

## 📝 Notes

- Free tier: 1GB RAM, 1 CPU core
- App sleeps after 7 days of inactivity
- Wakes up automatically when accessed
- Unlimited viewers (but be mindful of resource usage)

Good luck with your deployment! 🚀
