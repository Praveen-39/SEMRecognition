"""
Entrypoint for the Streamlit app.
Directly runs the main app from results/ph5_streamlit.py
"""

import sys
import os

# Add repo root to path so we can import results module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function
from results.ph5_streamlit import main

main()
