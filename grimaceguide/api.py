"""
API integration for the GrimaceGuide application
not original code
reference: https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi?usp=sharing#scrollTo=b1nzhrK-Z6uF

"""

import base64
import requests
import json
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .config import API_URL
from .fgsScoreCalc import calculate_scores_from_landmarks # Import score calculation from models

def convert_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def create_json_payload(image_path, image_base64_string):
    """Create JSON payload for API request"""
    payload = {
        "name": os.path.basename(image_path),  # Extract the filename from the path
        "image": f"data:image/jpeg;base64,{image_base64_string}"
    }
    return json.dumps(payload)

def send_image_for_processing(image_path, url=API_URL):
    """Send image to API for processing, calculate scores, and return results"""
    try:
        image_base64_string = convert_image_to_base64(image_path)
        request = create_json_payload(image_path, image_base64_string)

        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=request, headers=headers)

        if response.status_code == 200:
            print("Image processed successfully by API!")
            api_response_data = response.json()
            
            # Process image with landmarks for visualization
            processed_img_path = None
            try:
                original_img = PILImage.open(image_path)
                processed_img_path = process_image_with_landmarks(original_img, api_response_data, image_path)
            except Exception as e:
                print(f"Error visualizing landmarks: {e}")
                # Continue without visualization if it fails

            # Calculate scores based on landmarks
            scores = {}
            labeled_landmarks = {}
            try:
                labeled_landmarks = get_labeled_landmarks(api_response_data) # Get labeled dict
                if labeled_landmarks: # Check if landmarks were found and labeled
                     # Call the imported function
                     scores = calculate_scores_from_landmarks(labeled_landmarks)
                else:
                     print("No landmarks found or labeled, cannot calculate scores.")
                     # Return default scores or handle error appropriately
                     scores = {'ears': 0, 'eyes': 0, 'muzzle': 0, 'whiskers': 0, 'head': 0, 'total_score': 0}

            except Exception as e:
                print(f"Error calculating scores from landmarks: {e}")
                # Fallback or error handling for score calculation
                scores = {'ears': -1, 'eyes': -1, 'muzzle': -1, 'whiskers': -1, 'head': -1, 'total_score': -1} # Indicate error

            return {
                'success': True,
                'processed_image': processed_img_path, # Path to visualized image (or None)
                'api_response': api_response_data, # Raw API response for storage
                'scores': scores, # Calculated scores
                'labeled_landmarks': labeled_landmarks # Labeled landmarks for potential db storage
            }
            
        else:
            print(f"Failed to process image via API: {response.status_code}")
            print(response.text)
            return {
                'success': False,
                'error': f"API Error: {response.status_code}"
            }
    except Exception as e:
        print(f"Exception during API call or processing: {e}")
        return {
            'success': False,
            'error': f"Exception: {str(e)}"
        }

def process_image_with_landmarks(img, landmarks_result, original_path):
    """Draw landmarks on the image and save it"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a figure without displaying it
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Display the image
    ax.imshow(img)
    
    # Define the number of distinct animals
    num_animals = len(landmarks_result)
    cmap = plt.colormaps['rainbow']

    # Iterate over the result to plot landmarks with different colors
    for i, animal_data in enumerate(landmarks_result):
        for animal, details in animal_data.items():
            color = cmap(i / max(1, num_animals-1))  # Get a color from the colormap
            landmarks = details.get('landmarks', [])
            for landmark in landmarks:
                x = landmark.get('x', 0)
                y = landmark.get('y', 0)
                ax.scatter(x, y, color=color, s=30)  # Plot the point with the color
    
    ax.axis('off')
    
    # Save the processed image to a directory
    output_dir = os.path.join(os.path.dirname(original_path), "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(original_path)
    name, ext = os.path.splitext(base_filename)
    processed_path = os.path.join(output_dir, f"{name}_processed{ext}")
    
    fig.savefig(processed_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return processed_path

def get_labeled_landmarks(landmarks_result):
    """Convert API landmark result list into a dictionary keyed by labels based on the new structure."""
    labeled_landmarks = {}
    # Define the NEW mapping based on user description (48 points)
    # IMPORTANT: Assumes the API returns landmarks in this exact order.
    landmark_labels = (
        # Left Ear (5 points)
        [f'left_ear_{i+1}' for i in range(5)] +
        # Right Ear (5 points)
        [f'right_ear_{i+1}' for i in range(5)] +
        # Right Eye (4 points)
        [f'right_eye_{i+1}' for i in range(4)] +
        # Right Eye Pupil (4 points)
        [f'right_pupil_{i+1}' for i in range(4)] +
        # Left Eye (4 points)
        [f'left_eye_{i+1}' for i in range(4)] +
        # Left Eye Pupil (4 points)
        [f'left_pupil_{i+1}' for i in range(4)] +
        # Nose (5 points)
        [f'nose_{i+1}' for i in range(5)] +
        # Mouth (6 points)
        [f'mouth_{i+1}' for i in range(6)] +
        # Left Whiskers (5 points)
        [f'left_whisker_{i+1}' for i in range(5)] +
        # Right Whiskers (5 points)
        [f'right_whisker_{i+1}' for i in range(5)] +
        # Chin (1 point)
        ['chin_point']
    )

    if len(landmark_labels) != 48:
        print(f"ERROR: Landmark label definition has {len(landmark_labels)} labels, expected 48.")
        return {} # Prevent further errors

    if not landmarks_result:
        print("Warning: Received empty landmarks result from API.")
        return {}
        
    # Process landmarks for the first detected animal only
    first_animal_data = landmarks_result[0] 
    animal_key = list(first_animal_data.keys())[0] 
    landmarks = first_animal_data[animal_key].get('landmarks', [])

    if not landmarks:
        print(f"Warning: No landmarks found for animal '{animal_key}'.")
        return {}
        
    if len(landmarks) != 48:
        print(f"Warning: Received {len(landmarks)} landmarks, but expected 48. Labeling may be incorrect.")
        # Decide how to handle: truncate, error out, or pad?
        # For now, we'll try to label what we have.

    for i, landmark in enumerate(landmarks):
        if i < len(landmark_labels): # Check against the defined labels list length
            label = landmark_labels[i]
            labeled_landmarks[label] = {'x': landmark.get('x'), 'y': landmark.get('y')}
        else:
            # This case means API returned MORE than 48 points
            print(f"Warning: Extra landmark at index {i} beyond the expected 48.")
            labeled_landmarks[f'extra_point_{i}'] = {'x': landmark.get('x'), 'y': landmark.get('y')}
            
    return labeled_landmarks