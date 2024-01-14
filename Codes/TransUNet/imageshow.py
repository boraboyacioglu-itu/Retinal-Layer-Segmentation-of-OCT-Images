import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

# Load the MATLAB data file
mat_data = loadmat('D:/OCTA-500/GT_Layers-20230815T140002Z-001/GT_Layers/10301.mat')  # Replace with your actual file path
matrix_data = mat_data['Layer']
width_idx = 350

# Load your B-scan image (replace with your actual data)
image_filename = f'D:/OCTA-500/OCT/10301-20230827T211856Z-001/10301/{width_idx}.bmp'  # Replace with your image file naming convention
image_data = cv2.imread(image_filename)

# Define cropping coordinates
x_start = 50  # Replace with your desired x-coordinate start
x_end = 250   # Replace with your desired x-coordinate end
y_start = 150  # Replace with your desired y-coordinate start
y_end = 500   # Replace with your desired y-coordinate end

# Crop both the image and label data using the same coordinates
cropped_image = image_data[y_start:y_end, :]
cropped_matrix_data = matrix_data[:, width_idx, :]-150

# Define colors for each layer
colors = ['r', 'g', 'b', 'm', 'c', 'y']

# Create a new figure
plt.figure()

# Display the cropped image
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib

# Loop through the layers and plot the cropped label data
for layer_idx in range(cropped_matrix_data.shape[0]):
    layer = cropped_matrix_data[layer_idx, :]  # Extract the current layer

    # Plot the layer data
    plt.plot(layer, color=colors[layer_idx])

# Add labels and title
plt.xlabel('Width')
plt.ylabel('Depth')
plt.title(f'2D Plot of Layers at Length {width_idx} (Cropped)')

# Show legend
plt.legend(['ILM', 'IPL', 'OPL', 'ISOS', 'RPE', 'BM'])

# Show the plot
plt.show()
