#!/bin/bash

# Ensure we're in the project directory
cd "$(dirname "$0")/.."

# Activate the virtual environment
source .venv/bin/activate

# Launch the Streamlit app
streamlit run frontend/app.py
