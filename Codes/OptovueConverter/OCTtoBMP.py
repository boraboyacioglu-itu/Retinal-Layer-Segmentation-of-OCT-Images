import os
import numpy as np
from readvueoct import VUEOCT
import cv2
import matplotlib.pyplot as plt


def convert_oct_to_images(oct_file_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Read .OCT file
    vueoct = VUEOCT(oct_file_path)
    oct_volumes = vueoct.read_oct_volume()
    oct_volumes = np.array(oct_volumes[0].transpose((1, 0, 2)))

    # Create a folder for each file
    file_folder = os.path.join(output_folder, "9796_OS")
    os.makedirs(file_folder, exist_ok=True)
        
    # Process each slice in the OCT volume
    for slice_index in range(oct_volumes.shape[1]):
        # Select the slice
        selected_slice = oct_volumes[:, slice_index, :]

        # Normalize and convert the slice to uint8
        normalized_slice = cv2.normalize(selected_slice, None, 0, 255, cv2.NORM_MINMAX)
        normalized_slice = normalized_slice.astype(np.uint8)

        # Save the slice as an image
        slice_file = os.path.join(file_folder, f"{slice_index + 1}.bmp")
        cv2.imwrite(slice_file, normalized_slice)
        
if __name__ == "__main__":
    output_folder = "D:\\YZV-DERSLER\\YZV302E- Deep Learning\\optovue_extract\\Boray Hoca-20240108T123717Z-001\\Boray Hoca\\OCT_DATA"
    file_path = "D:/İndirilenler/OptovueExport/9796/OCT/b9796, y9796 _17446_HD Angio Retina_OS_2023-07-19_15.07.22_1.OCT"
    
    convert_oct_to_images(file_path, output_folder)
