#Congifuration settings for App
#To be imported in main file

import os
from pathlib import Path


#Establish directory structure for finding resources and storing data

BASE_DIR = Path(__file__).parent.parent
#Root directory of the project
PACKAGE_DIR = Path(__file__).parent
#Directory containing the grimaceguide package
RESOURCES_DIR = PACKAGE_DIR / 'resources'
#Static resources like default images
IMAGES_GUI_DIR = BASE_DIR / 'imagesGUI'
#Directory for explaining scoring in GUI images

#API settings
API_URL = "http://34.165.76.57:6000/landmarks"
#External API endpoint for advanced image processing and landmark detection


#Main GUI window dimensions and title
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
WINDOW_TITLE = 'Grimace Score Evaluator'

#Color scheme for App
#Using RGB values normalized to 0-1 range with alpha channel (opacity)
COLORS = {
    'primary': (0.0, 0.447, 0.741, 1),     #Blue - main application color
    'secondary': (0.8, 0.4, 0.0, 1),       #Orange - accent color for secondary actions
    'success': (0.0, 0.7, 0.0, 1),         #Green - indicates successful operations
    'model': (0.0, 0.0, 0.8, 1),           #Dark Blue - used for model processing button
    'danger': (0.8, 0.2, 0.2, 1),          #Red - warnings and cancellation actions
    'light_gray': (0.95, 0.95, 0.95, 1),   #Light Gray - backgrounds and subtle separators
    'dark_gray': (0.3, 0.3, 0.3, 1),       #Dark Gray - borders and secondary text
    'white': (1, 1, 1, 1),                 #White - text on dark backgrounds
    'black': (0, 0, 0, 1),                 #Black - primary text color
    'background': (0.97, 0.97, 0.97, 1),   #Very Light Gray - panel backgrounds
}


#Reference/Control group images showing scoring examples for each facial feature
#Images depict the facial grimace scale criteria for each category
HELP_IMAGES = {
    'ears': os.path.join(IMAGES_GUI_DIR, 'GrimaceGuideEarsGUI.png'),    #Ear position scoring guide
    'eyes': os.path.join(IMAGES_GUI_DIR, 'GrimaceGuideEyesGUI.png'),    #Eye squeezing scoring guide
    'muzzle': os.path.join(IMAGES_GUI_DIR, 'GrimaceGuideMuzzleGUI.png'),  #Nose/cheek flattening guide
    'whiskers': os.path.join(IMAGES_GUI_DIR, 'GrimaceGuideWhiskersGUI.png'),  #Whisker position guide
    'head': os.path.join(IMAGES_GUI_DIR, 'GrimaceGuideHeadGUI.png')     #Head shape/position guide
}

#Scoring categories
#Each category is scored on a scale of 0-2 AU (Action Units) (0=not present, 1=moderate, 2=severe)
#Higher scores indicate higher pain levels/distress
SCORE_CATEGORIES = ['ears', 'eyes', 'muzzle', 'whiskers', 'head']