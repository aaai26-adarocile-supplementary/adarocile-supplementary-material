#!/bin/bash

# Setup script for Rocile and AdaRocile package
echo "Setting up Rocile and AdaRocile package..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
else
    echo "Error: Failed to install dependencies."
    exit 1
fi

# Install the package in development mode
echo "Installing package in development mode..."
pip3 install -e .

if [ $? -eq 0 ]; then
    echo "Package installed successfully!"
else
    echo "Error: Failed to install package."
    exit 1
fi

# Run tests to verify installation
echo "Running tests to verify installation..."
python3 -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Warning: Some tests failed. Please check the output above."
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "You can now use the package:"
echo "  - Run training: python3 train.py"
echo "  - Run evaluation: python3 evaluate.py"
echo "  - Import in Python: from mypackage.model import Rocile, AdaRocile"
echo ""
echo "For more information, see the README.md file." 