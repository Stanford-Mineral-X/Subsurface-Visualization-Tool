# Subsurface Visualization Tool

## Created by: Heba Alazzeh

### Overview
This tool is designed to load, process, and visualize 3D subsurface data from multiple anisotropic and isotropic CSV files. The tool allows users to toggle between different visualization modes: standard view, interactive slicing, and structure identification.

### Features
- **Standard View**: Displays the full 3D volumetric data.
- **Interactive Slicing**: Allows users to slice through the 3D data along the x, y, and z axes interactively.
- **Structure Identification**: Isolates and visualizes specific structures, such as ore bodies, based on threshold values.

### Requirements
- Python 3.6+
- Required Python packages:
  - `pandas`
  - `numpy`
  - `mayavi`
  - `traits`
  - `traitsui`

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/subsurface-visualization-tool.git
   cd subsurface-visualization-tool


#Install the Required Packages:
pip install -r requirements.txt

#Usage
#Run the Tool:

Ensure you have the necessary CSV files for anisotropic and isotropic data in the specified format.
Run the visualization tool with the following command:

python visualize_subsurface_data.py

#Toggle Between Visualization Modes:

#Once the tool is running, you can toggle between the following modes using the dropdown menu:

#Standard View: The default view showing the entire 3D volume.
#Interactive Slicing: Allows interactive slicing through the data.
#Structure Identification: Displays structures based on a threshold value.

#File Structure
visualize_subsurface_data.py: The main script that runs the visualization tool.
README.md: Documentation for the project.
requirements.txt: List of required Python packages.

Example Data
The tool expects CSV files with the following structure:

x,y,z,value
0,0,0,1.5
0,0,1,1.6
0,0,2,1.7
...

#Contributing
If you would like to contribute to this project, feel free to open an issue or submit a pull request.


Contact
For any questions or issues, please contact Heba Alazzeh via email.
