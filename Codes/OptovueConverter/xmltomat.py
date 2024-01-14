import xml.etree.ElementTree as ET
import scipy.io

def parse_xml_to_array(xml_file_path, layer_names, num_images, num_pixels):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Initialize a dictionary to hold the layer data
    layer_data = {layer: [] for layer in layer_names}

    # Extract data for each curve
    for curve in root.iter('Curve'):
        layer_type = curve.find('Type').text if curve.find('Type') is not None else None
        if layer_type in layer_names:
            d_values = [float(d.text.strip()) for d in curve.iter('D')]
            layer_data[layer_type].append(d_values)

    # Reorder and organize the data
    arranged_data = [[[0 for _ in range(num_pixels)] for _ in range(num_images)] for _ in layer_names]
    for i, layer in enumerate(layer_names):
        for j, image_data in enumerate(layer_data.get(layer, [])):
            arranged_data[i][j][:len(image_data)] = image_data

    return arranged_data

def save_to_mat(file_path, data):
    scipy.io.savemat(file_path, {'Layer': data})

def main():
    # Define the parameters
    xml_file_path = "D:/İndirilenler/OptovueExport/9796/curve/b9796, y9796 _OS_HD Angio Retina_9796_17446_5_1.xml"
    mat_file_path = 'D:/YZV-DERSLER/YZV302E- Deep Learning/optovue_extract/Boray Hoca-20240108T123717Z-001/Boray Hoca/LAYER_DATA/9796_OS.mat'      # Replace with your desired output .mat file path
    layer_names = ['ILM', 'IPL', 'OPL', 'ISOS', 'RPE', 'BRM']  # Specify the layer names
    num_images = 400  # Set the number of images
    num_pixels = 400  # Set the number of pixels

    # Process the XML file and get the layer data
    layer_data = parse_xml_to_array(xml_file_path, layer_names, num_images, num_pixels)

    # Save the layer data to a .mat file
    save_to_mat(mat_file_path, layer_data)

    print(f"Data saved to {mat_file_path}")

if __name__ == "__main__":
    main()
