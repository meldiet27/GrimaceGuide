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
    """Send image to API for processing and return results"""
    try:
        image_base64_string = convert_image_to_base64(image_path)
        request = create_json_payload(image_path, image_base64_string)

        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=request, headers=headers)

        if response.status_code == 200:
            print("Image processed successfully!")
            result = response.json()
            
            # Process image with landmarks
            try:
                original_img = PILImage.open(image_path)
                processed_img_path = process_image_with_landmarks(original_img, result, image_path)
                
                # Calculate scores based on landmarks
                scores = calculate_scores_from_landmarks(result)
                
                return {
                    'success': True,
                    'processed_image': processed_img_path,
                    'api_response': result,
                    'scores': scores
                }
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                return {
                    'success': True,
                    'api_response': result,
                    'error_processing': str(e)
                }
        else:
            print(f"Failed to process image: {response.status_code}")
            print(response.text)
            return {
                'success': False,
                'error': f"API Error: {response.status_code}"
            }
    except Exception as e:
        print(f"Exception during API call: {e}")
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

def calculate_scores_from_landmarks(landmarks_result):
    """Calculate grimace scores based on landmarks"""
    # This is a dummy implementation - in a real app, you'd use actual logic
    # based on the landmarks to calculate the scores
    import random
    scores = {
        'ears': random.randint(0, 2),
        'eyes': random.randint(0, 2),
        'muzzle': random.randint(0, 2),
        'whiskers': random.randint(0, 2),
        'head': random.randint(0, 2)
    }
    
    # Calculate total score
    total_score = sum(scores.values())
    scores['total_score'] = total_score
    
    return scores