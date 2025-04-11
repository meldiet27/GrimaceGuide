#NOT ORIGINAL WRITTEN CODE, use for reference only
#credit to: https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi?usp=sharing#scrollTo=WKZ9yOIBUtVg

import base64
import requests
import json

def convert_image_to_base64(image_path):
  with open(image_path, "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')
  return base64_string

def create_json_payload(image_path, image_base64_string):
  payload = {
    "name": image_path.split("/")[-1],  # Extract the filename from the path
    "image": f"data:image/jpeg;base64,{image_base64_string}"
  }
  return json.dumps(payload)

def send_image_for_processing(image_path, url):
  image_base64_string = convert_image_to_base64(image_path)
  request = create_json_payload(image_path, image_base64_string)

  headers = {'Content-Type': 'application/json'}

  response = requests.post(url, data=request, headers=headers)

  if response.status_code == 200:
    print("Image processed successfully!")
    print("Response:", response.json())
    return response.json()
  else:
    print("Failed to process image")

  # @title get cat image

  cat_url = "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"  # @param {type:"string"}

  # Send a GET request to the URL
  response = requests.get(cat_url)

  # Check if the request was successful (status code 200)
  if response.status_code == 200:
    # Open a file in binary write mode and save the image content
    with open("image.jpg", "wb") as f:
      f.write(response.content)
    print("Cat image downloaded successfully!")
  else:
    print("Failed to download the cat image.")

  url = "http://34.165.76.57:6000/landmarks"
  image_path = "/content/image.jpg"  # Replace with the actual path to your image

  # Send the image for processing
  result = send_image_for_processing(image_path, url)

