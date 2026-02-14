#!/bin/bash
# Setup script for Google Earth Engine Local Analysis
# Run this script to set up your environment

set -e  # Exit on error

echo "=========================================="
echo "GEE Irrigation Mapping - Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment exists
if conda env list | grep -q "^gee "; then
    echo "⚠ Environment 'gee' already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n gee -y
    else
        echo "Skipping environment creation."
        ENV_EXISTS=true
    fi
fi

# Create environment if needed
if [ -z "$ENV_EXISTS" ]; then
    echo "Creating conda environment 'gee'..."
    conda env create -f environment.yml
    echo "✓ Environment created"
    echo ""
fi

# Activate environment
echo "To activate the environment, run:"
echo "  conda activate gee"
echo ""

# Check if already authenticated
CRED_FILE="$HOME/.config/earthengine/credentials"
if [ -f "$CRED_FILE" ]; then
    echo "✓ Earth Engine credentials found"
    echo ""
    echo "Testing authentication..."
    conda run -n gee python test_auth.py
else
    echo "⚠ No Earth Engine credentials found"
    echo ""
    echo "To authenticate with Earth Engine:"
    echo "  1. conda activate gee"
    echo "  2. earthengine authenticate"
    echo "  3. Follow the browser prompts"
    echo ""
    read -p "Do you want to authenticate now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting authentication..."
        conda run -n gee earthengine authenticate
        echo ""
        echo "Testing authentication..."
        conda run -n gee python test_auth.py
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. conda activate gee"
echo "  2. jupyter notebook"
echo "  3. Open: zambezi_irrigation_mapping.ipynb"
echo ""
echo "For help, see:"
echo "  - LOCAL_SETUP.md (detailed setup)"
echo "  - AUTHENTICATION_GUIDE.md (auth troubleshooting)"
echo "  - QUICK_REFERENCE.md (command cheat sheet)"
echo ""
