#!/bin/bash

# This script prepares the application for deployment on Render

# Make sure we're in the right directory
echo "Current directory: $(pwd)"

# Install dependencies
pip install -r requirements.txt

echo "Build script completed successfully"