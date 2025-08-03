@echo off
echo ====================================================================
echo 🤖 DAIRY ANALYTICS ML MODEL TRAINING - ENHANCED BATCH SCRIPT
echo ====================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo 🔧 Please install Python and try again
    pause
    exit /b 1
)

echo ✅ Python detected. Initializing enhanced ML training...
echo.

REM Create models directory if it doesn't exist
if not exist "models" mkdir models

echo 🚀 TRAINING OPTIONS:
echo    • Quick Training (~3 minutes, 3 stores)
echo    • Full Training (~10 minutes, all stores)
echo.

REM Default to quick training for better user experience
echo 🔄 Starting QUICK TRAINING with progress tracking...
echo ⏱️  Estimated time: 3-5 minutes
echo 📊 Progress bar and timing will be displayed
echo ────────────────────────────────────────────────────────────────────

python train_models.py --quick_train

if errorlevel 1 (
    echo.
    echo ❌ ERROR: Model training failed!
    echo 🔧 Troubleshooting suggestions:
    echo    1. Check if data files exist in data/ directory
    echo    2. Ensure sufficient memory (4GB+ recommended)
    echo    3. Try running manually: python train_models.py --quick_train
    echo    4. Check Python dependencies: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo 🎉 TRAINING COMPLETED SUCCESSFULLY!
echo ====================================================================
echo 💾 Models saved in: models/ directory
echo 📁 Files created:
echo    • latest_trained_models.pkl (for instant loading)
echo    • Timestamped backup files
echo.
echo 🚀 NEXT STEPS:
echo    1. Run: streamlit run app.py
echo    2. Navigate to: 🤖 Advanced ML Analytics
echo    3. Click: 🚀 Load Pre-trained Models
echo    4. Enjoy instant ML predictions!
echo.
echo ⚡ TIP: Models will now load in under 2 seconds!
echo ====================================================================
pause
