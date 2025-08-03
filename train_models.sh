#!/bin/bash

echo "========================================="
echo "  DAIRY ANALYTICS ML MODEL PRE-TRAINING"
echo "========================================="
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python and try again"
    exit 1
fi

echo "Python detected. Starting model training..."
echo

# Create models directory if it doesn't exist
mkdir -p models

# Run the training script with quick training for demo
echo "Training ML models (Quick mode for demo)..."
python train_models.py --quick_train

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Model training failed!"
    echo "Check the error messages above"
    exit 1
fi

echo
echo "========================================="
echo "  TRAINING COMPLETED SUCCESSFULLY!"
echo "========================================="
echo
echo "Models saved in: models/"
echo "You can now run: streamlit run app.py"
echo
