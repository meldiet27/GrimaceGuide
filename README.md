# Grimace Guide

An application for evaluating grimace scores in cats using computer vision and landmarks detection.

# Folder Structure

    GrimaceGuide/
    │
    ├── main.py                  # Application entry point
    ├── requirements.txt         # Dependencies
    ├── README.md                # Documentation
    │
    ├── grimaceguide/           # Package directory
    │   ├── __init__.py         # Package initializer
    │   ├── config.py           # Configuration settings
    │   ├── database.py         # Database management
    │   ├── api.py              # API integration
    │   ├── fgsScoreCalc.py     # FGS Score Calculation
    │   ├── utils.py            # Utility functions
    │   │
    │   ├── ui/                 # UI components
    │   │   ├── __init__.py     # UI package initializer
    │   │   ├── app.py          # Main application class
    │   │   ├── widgets.py      # Custom widgets
    │   │   └── popups.py       # Custom popup dialogs
    │   │
    │   └── resources/          # Application resources
    │       └── __init__.py     # Resources package initializer
    │
    └── imagesGUI/              # Guide images directory
    

## Features

- Upload and display images
- Process images with both API and local model
- Display and visualize grimace scores
- Store processing history in a database

## Installation

1. Clone this repository

        gh repo clone ProjectLantier/GrimaceGuide
   
3. Create a virtual environment:

        python -m venv venv

4. Activate the virtual environment:

       .\venv\Scripts\activate
   
5. Install dependencies:

        pip install -r requirements.txt

## Usage

Run the application:

        python main.py


## Directory Structure

- `grimaceguide/` - Main package
  - `ui/` - User interface components
  - `resources/` - Application resources
- `imagesGUI/` - Guide images for help popups
