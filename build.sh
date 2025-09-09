#!/bin/bash

# Verify we're in the right directory
echo "Current directory: $(pwd)"
echo "Listing directory contents:"
ls -la

# Install Python dependencies
pip install -r requirements.txt

# Install spaCy model
pip install spacy
python -m spacy download en_core_web_sm

# Make sure templates directory exists
mkdir -p templates

# Verify templates directory
echo "Checking templates directory:"
ls -la templates/

echo "Build completed successfully"