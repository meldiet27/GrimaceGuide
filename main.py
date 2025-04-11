"""
GrimaceGuide - Main entry point for the application
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from grimaceguide package
# This ensures the application can find all modules regardless of how it's launched
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))

# Set environment variable to use non-interactive matplotlib backend
# This prevents matplotlib from trying to use GUI backends that might conflict with Kivy
# Agg is a good choice as it's reliable across platforms and doesn't require a display
os.environ['MPLBACKEND'] = 'Agg'

# Import the main application class
# The app is implemented as a Kivy App subclass in the grimaceguide package
from grimaceguide.ui.app import GrimaceGuideApp

if __name__ == "__main__":
    # Initialize and run the application
    # This is the standard entry point pattern for Kivy applications
    # The run() method will handle the main event loop and UI rendering
    app = GrimaceGuideApp()
    app.run()