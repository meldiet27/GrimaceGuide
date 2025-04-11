#NOT ORIGINAL WRITTEN CODE, use for reference only
#credit to: https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi?usp=sharing#scrollTo=WKZ9yOIBUtVg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.cm as cm
import numpy as np

image = Image.open(image_path)
# Create a figure and axis
fig, ax = plt.subplots()
# Display the image
ax.imshow(image)
# Define the number of distinct animals
num_animals = len(result)
cmap = plt.colormaps['rainbow']

# Iterate over the result to plot landmarks with different colors
for i, animal_data in enumerate(result):
  for animal, details in animal_data.items():
    color = cmap(i / num_animals +2)  # Get a color from the colormap
    landmarks = details['landmarks']
    for landmark in landmarks:
      x = landmark['x']
      y = landmark['y']
      ax.scatter(x, y, color=color,  s=5)  # Plot the point with the color from the colormap

plt.axis('off')
plt.savefig('output_image2.png')
# Show the plot
plt.show()