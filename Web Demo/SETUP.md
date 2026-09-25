# 🏦 Bank Marketing Demo - Setup Guide

Complete setup and installation instructions for the Streamlit web application.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Disk Space**: 500MB
- **OS**: Windows, macOS, or Linux

### Python Version Check
```bash
python --version
# or
python3 --version
```

## Installation Steps

### Step 1: Clone or Navigate to Project
```bash
# Navigate to the project directory
cd /home/qz/VS_Code/Project_ML/Web\ Demo

# Or if in a different location, adjust the path accordingly
```

### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually:
pip install streamlit pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost shap plotly
```

### Step 5: Verify Installation
```bash
python -c "import streamlit; print(streamlit.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

## Running the Application

### Basic Launch
```bash
streamlit run app.py
```

### With Custom Port
```bash
streamlit run app.py --server.port 8501
```

### With Custom Server Address
```bash
streamlit run app.py --server.address localhost --server.port 8501
```

### Full Configuration
```bash
streamlit run app.py \
    --logger.level=debug \
    --client.showErrorDetails=true \
    --server.port=8501
```

## Access the Application

After running the command above:
1. Open your web browser
2. Navigate to: `http://localhost:8501`
3. The app should load within seconds

## Troubleshooting

### Issue 1: Module Not Found Error
**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
# Verify virtual environment is activated (should see (venv) in terminal)
which python  # macOS/Linux
where python  # Windows

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

### Issue 2: Data File Not Found
**Error**: `Data file not found. Please ensure processed_bank_data.csv is in Models folder.`

**Solution**:
```bash
# Check if the file exists
ls -la ../Models/processed_bank_data.csv  # macOS/Linux
dir ..\Models\processed_bank_data.csv     # Windows

# If not found, ensure you have the correct path:
# Expected structure:
# Project_ML/
#   ├── Models/
#   │   └── processed_bank_data.csv
#   └── Web Demo/
#       └── app.py
```

### Issue 3: Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using port 8501:
# macOS/Linux:
lsof -ti:8501 | xargs kill -9

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue 4: Memory Issues
**Error**: `MemoryError` or slow performance

**Solution**:
```bash
# Run with limited cache:
streamlit run app.py --client.maxUploadSize=50

# If still slow, restart Streamlit:
streamlit cache clear
streamlit run app.py
```

### Issue 5: Permission Denied on macOS/Linux
**Error**: `Permission denied`

**Solution**:
```bash
# Make script executable
chmod +x app.py

# Or run with Python explicitly
python app.py
streamlit run app.py
```

### Issue 6: Package Version Conflicts
**Error**: `Version conflict` or `Requirement already satisfied`

**Solution**:
```bash
# Create fresh environment
deactivate
rm -rf venv  # macOS/Linux: use 'rmdir /s venv' on Windows
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Project Structure

```
Web Demo/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── SETUP.md                  # Setup guide (this file)
├── .gitignore               # Git ignore rules
├── .streamlit_config.toml    # Streamlit configuration
└── __pycache__/             # Python cache (auto-generated)

Project_ML/
├── Models/
│   ├── processed_bank_data.csv    # Dataset
│   ├── Logistic_Regression.ipynb
│   └── XGBoost.ipynb
├── Feature Engineering + PreProcessing/
│   └── preprocessing.ipynb
├── EDA + Data/
│   └── bank_marketing_eda.ipynb
└── Web Demo/
    └── (this folder)
```

## Configuration

### Streamlit Configuration File
Edit `.streamlit_config.toml` to customize:
- Theme colors
- Font settings
- Server options
- Logger settings

### App Configuration
Edit `config.py` to modify:
- Data paths
- Model hyperparameters
- Feature settings
- UI colors and messages

### Environment Variables
You can set environment variables before running:

**macOS/Linux:**
```bash
export STREAMLIT_LOGGER_LEVEL=debug
export STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true
streamlit run app.py
```

**Windows:**
```cmd
set STREAMLIT_LOGGER_LEVEL=debug
set STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true
streamlit run app.py
```

## Development Tips

### Enable Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

### Clear Cache
```bash
streamlit cache clear
```

### Run Tests (if added)
```bash
pytest tests/
```

### Generate Requirements from Environment
```bash
pip freeze > requirements.txt
```

## Performance Optimization

### For Faster Loading:
1. **Cache data**: The app caches data automatically with `@st.cache_data`
2. **Pre-train models**: Models are cached with `@st.cache_resource`
3. **Limit data size**: Process only necessary columns

### For Production Deployment:
```bash
streamlit run app.py \
    --server.port 80 \
    --server.address 0.0.0.0 \
    --logger.level=warning \
    --client.showErrorDetails=false
```

## Deployment Options

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Docker (if Dockerfile provided)
```bash
docker build -t bank-marketing-demo .
docker run -p 8501:8501 bank-marketing-demo
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select this app file

## Upgrading Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade streamlit
```

## System-Specific Notes

### macOS
- Use `python3` instead of `python` if Python 2 is also installed
- May need to install developer tools: `xcode-select --install`

### Linux (Ubuntu/Debian)
```bash
# Install Python and pip
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Install system dependencies for some packages
sudo apt-get install build-essential python3-dev
```

### Windows
- Use Command Prompt or PowerShell
- May need to add Python to PATH if not done during installation
- Use `python` instead of `python3`

## Support and Documentation

- **Streamlit Docs**: https://docs.streamlit.io
- **Scikit-learn Docs**: https://scikit-learn.org
- **Pandas Docs**: https://pandas.pydata.org
- **NumPy Docs**: https://numpy.org

## Next Steps

1. Run the application: `streamlit run app.py`
2. Explore the different pages
3. Interact with the model
4. Check the README for feature details
5. Review the notebooks for deeper insights

---

**Last Updated**: May 2026

**Need Help?** Check the troubleshooting section or refer to official documentation links above.
