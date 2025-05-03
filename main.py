"""
GrimaceGuide - Main entry point for the application.
Initializes the environment and launches the Kivy application.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the 'grimaceguide' package.
# This ensures the application can find its modules regardless of the launch directory.
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

# Set the Matplotlib backend to 'Agg' before importing Kivy or Matplotlib.
# 'Agg' is a non-interactive backend that renders figures to files,
# preventing potential conflicts with Kivy's graphical backend.
os.environ['MPLBACKEND'] = 'Agg'

# Import the main application class after setting the backend.
from grimaceguide.ui.app import GrimaceGuideApp

# Standard Python entry point check.
if __name__ == "__main__":
    # Create an instance of the main application class.
    app = GrimaceGuideApp()
    # Start the Kivy application event loop. This will build the UI and handle user interactions.
    app.run()