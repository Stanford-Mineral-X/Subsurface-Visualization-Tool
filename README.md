# Subsurface Visualization Tool 

## Created by: Heba Alazzeh

A comprehensive toolkit for visualizing subsurface geological data using multiple visualization approaches.

## v3 in progress... 

## Features

- **Interactive Web Dashboard**: Explore complex subsurface data through a browser-based interface
- **3D Isosurface Visualization**: Examine subsurface structures with adjustable parameters
- **Volume Slicing**: View cross-sections along any axis to analyze internal features
- **Scientific Colormaps**: Integration with cmcrameri scientific colormaps optimized for geoscience visualization
- **Multiple Data Formats**: Support for various data input formats
- **Log Transformation**: Toggle logarithmic transformation for data with high dynamic range

## Files in this Repository

### Core Visualization
- `subsurface_visualization_plotly.py`: Plotly-based visualization script with 3D isosurfaces
- `subsurface_visualization_v2.py`: Matplotlib-based visualization for simpler 3D contours
- `visualize_subsurface_data.py`: Mayavi-based visualization tool with rich 3D features

### Dashboard Interface
- `enhanced_subsurface_dashboard.py`: Advanced web dashboard with dataset selection

### Utilities
- `colormap_utils.py`: Integration with scientific colormaps from cmcrameri
- `utility_functions.py`: Helper functions for data processing and transformation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Subsurface-Visualization-Tool.git
cd Subsurface-Visualization-Tool

# Install dependencies for Plotly visualization
pip install numpy scipy plotly pandas matplotlib dash dash-bootstrap-components

# For the Hawaii colormap (optional but recommended)
pip install cmcrameri

# For Mayavi visualization (optional)
pip install numpy pandas mayavi vtk traits traitsui
```

## Usage

### Web Dashboard (Recommended)

For the most user-friendly experience with dataset selection and interactive controls:

```bash
python enhanced_subsurface_dashboard.py
```

Then open your browser to http://127.0.0.1:8050/

### Command-Line Visualization

For simple visualization from the command line:

```bash
# Using Plotly (3D isosurfaces)
python subsurface_visualization_plotly.py path/to/your/data.csv

# Using Matplotlib (3D contours)
python subsurface_visualization_v2.py path/to/your/data.csv
```

### Mayavi-based GUI Visualization

For advanced 3D visualization with GUI controls:

```bash
python visualize_subsurface_data.py
```

## Command Line Options

For Plotly visualization:

```bash
python subsurface_visualization_plotly.py your_data.csv --num_contours 15 --opacity 0.7 --colormap hawaii --use_log --show_slices
```

- `--num_contours`: Specify the number of isosurface levels (default: 10)
- `--opacity`: Set the opacity for isosurfaces (default: 0.5)
- `--colormap`: Choose the colormap for visualization (default: hawaii if available, else Viridis)
- `--use_log`: Apply log transformation to values
- `--show_slices`: Show slices through the volume
- `--output`: Optional HTML file to save the visualization

For the web dashboard:

```bash
python enhanced_subsurface_dashboard.py --port 8080 --debug --data-dir ./data
```

## Data Format

The visualization tools accept CSV files in the following formats:

### Basic Format
For `subsurface_visualization_plotly.py` and `subsurface_visualization_v2.py`:
```
x,y,z
1.0,2.0,3.5
1.2,2.3,3.7
...
```

### Enhanced Format
For `enhanced_subsurface_dashboard.py`:
```
x,y,z,value
1.0,2.0,3.5,245.6
1.2,2.3,3.7,247.8
...
```

## Scientific Colormap Integration

The `colormap_utils.py` file provides integration with scientific colormaps from the cmcrameri package, which are particularly well-suited for geoscience visualization.

To preview available colormaps:

```python
from colormap_utils import preview_colormap, list_available_cmcrameri_colormaps

# List all available colormaps
colormaps = list_available_cmcrameri_colormaps()
print(colormaps)

# Preview a specific colormap
preview_colormap('hawaii')
```