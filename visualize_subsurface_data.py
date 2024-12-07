# Subsurface Visualization Tool
# Created by: Heba Alazzeh
# Version: 2.0.0
"""
Function: This script loads, processes, and visualizes 3D subsurface data from multiple anisotropic and isotropic CSV files. 

Features include:
- Multiple visualization modes (Standard, Interactive Slicing, Structure Identification)
- Customizable opacity and threshold settings
- Toggleable cross-sections for each axis
- Adjustable slice counts
- Colorbar toggle

Environment Requirements:
- Python >= 3.8, <= 3.11  # Mayavi has compatibility issues with Python 3.12
- NumPy >= 1.20.0, < 2.0.0
- Pandas >= 1.3.0, < 2.0.0
- Mayavi == 4.7.4  # Specific version for stability
- VTK >= 9.0.0, < 10.0.0
- Traits >= 6.2.0, < 7.0.0
- TraitsUI >= 7.2.0, < 8.0.0

To install requirements:
pip install numpy==1.24.3 pandas==1.5.3 mayavi==4.7.4 vtk==9.2.6 traits==6.3.2 traitsui==7.4.3

Note: For Windows users, it's recommended to install VTK and Mayavi using conda:
conda install -c conda-forge vtk mayavi

Last Updated: 12/06/2024
"""

from numpy import nan
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Instance, Enum, Button, Range, Bool, on_trait_change
from traitsui.api import View, Item, Group, HGroup, VGroup
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

# List of file paths for anisotropic and isotropic data
anisotropic_files = [
    'plot_data_anisotropic_1.csv',
    'plot_data_anisotropic_2.csv',
    'plot_data_anisotropic_3.csv',
    'plot_data_anisotropic_4.csv',
    'plot_data_anisotropic_5.csv',
    'plot_data_anisotropic_6.csv'
]

isotropic_files = [
    'plot_data_isotropic_1.csv',
    'plot_data_isotropic_2.csv',
    'plot_data_isotropic_3.csv',
    'plot_data_isotropic_4.csv',
    'plot_data_isotropic_5.csv',
    'plot_data_isotropic_6.csv'
]


def load_and_combine(files):
    """Load and combine multiple CSV files into a single DataFrame"""
    combined_data = pd.concat([pd.read_csv(file)
                              for file in files], ignore_index=True)
    return combined_data


# Load and combine the data
anisotropic_data = load_and_combine(anisotropic_files)
isotropic_data = load_and_combine(isotropic_files)


def aggregate_data(data):
    """Aggregate values by taking the mean for each (x, y, z) coordinate"""
    aggregated_data = data.groupby(['x', 'y', 'z'], as_index=False).mean()
    return aggregated_data


anisotropic_data = aggregate_data(anisotropic_data)
isotropic_data = aggregate_data(isotropic_data)


def reshape_data(data):
    """Extract unique coordinates and reshape the data into a 3D array"""
    x_unique = np.unique(data['x'])
    y_unique = np.unique(data['y'])
    z_unique = np.unique(data['z'])

    x_count = len(x_unique)
    y_count = len(y_unique)
    z_count = len(z_unique)

    value_3d = data['value'].values.reshape((x_count, y_count, z_count))
    return x_unique, y_unique, z_unique, value_3d


x_unique_a, y_unique_a, z_unique_a, value_3d_a = reshape_data(anisotropic_data)
x_unique_i, y_unique_i, z_unique_i, value_3d_i = reshape_data(isotropic_data)


class VisualizationTool(HasTraits):
    """Main visualization tool class with GUI controls and visualization methods."""

    # Scene and core traits
    scene = Instance(MlabSceneModel, ())

    # Define all Enum traits with default values
    view_mode = Enum(
        "Standard", ["Standard", "Interactive Slicing", "Structure Identification"])
    view_data = Enum("Anisotropic Data", [
                     "Anisotropic Data", "Isotropic Data"])
    update_view = Button("Update View")

    # Visualization control traits
    opacity = Range(0.0, 1.0, 0.5)
    threshold_value = Range(0.0, 10.0, 4.0)
    show_colorbar = Bool(True)

    # Cross-section traits
    show_x_section = Bool(False)
    show_y_section = Bool(False)
    show_z_section = Bool(False)

    # Slice position traits
    x_slice_pos = Range(0.0, 1.0, 0.5)
    y_slice_pos = Range(0.0, 1.0, 0.5)
    z_slice_pos = Range(0.0, 1.0, 0.5)

    # Multiple slice traits
    enable_multi_slice = Bool(False)
    num_x_slices = Range(0, 5, 1)
    num_y_slices = Range(0, 5, 1)
    num_z_slices = Range(0, 5, 1)

    # Structure identification traits
    show_slicing_structure = Bool(False)
    num_slices_structure = Range(0, 5, 1)

    def __init__(self):
        """Initialize with improved error handling."""
        super(VisualizationTool, self).__init__()

        # Initialize instance variables
        self.plot_data = None
        self.x = None
        self.y = None
        self.z = None
        self.data = None
        self.title = ""
        self.colorbar = None
        self.volume = None
        self.x_plane = None
        self.y_plane = None
        self.z_plane = None

        try:
            # Set initial data
            self.set_data(x_unique_a, y_unique_a, z_unique_a,
                          value_3d_a, "Anisotropic Data")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")

    def set_data(self, x, y, z, data, title):
        """Set the data to be visualized with type conversion."""
        try:
            # Convert input arrays to float64 to ensure compatibility
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            z = np.array(z, dtype=np.float64)
            data = np.array(data, dtype=np.float64)

            # Create meshgrid
            self.x, self.y, self.z = np.meshgrid(x, y, z, indexing='ij')
            self.data = data
            self.title = title

            if hasattr(self, 'update_visualization'):
                self.update_visualization()
        except Exception as e:
            print(f"Error in set_data: {str(e)}")

    def setup_scene(self):
        """Set up common scene elements with improved colorbar handling."""
        try:
            # Set white background
            if hasattr(self.scene, 'background'):
                self.scene.background = (1.0, 1.0, 1.0)

            # Clear any existing colorbar
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None

            # Setup axes
            try:
                axes = mlab.axes(self.plot_data, color=(0, 0, 0),
                                 xlabel='X', ylabel='Y', zlabel='Z',
                                 nb_labels=5)
                if axes is not None:
                    axes.label_text_property.color = (0, 0, 0)
                    axes.title_text_property.color = (0, 0, 0)
                    axes.label_text_property.font_size = 12
                    axes.title_text_property.font_size = 12
                    axes.axes.label_format = '%.1f'
            except Exception as e:
                print(f"Warning: Could not setup axes properly: {str(e)}")

            # Enhanced colorbar handling
            if self.show_colorbar and self.plot_data is not None:
                try:
                    # Choose appropriate source for colorbar
                    if self.view_mode == "Structure Identification" and hasattr(self, 'volume') and self.volume is not None:
                        colorbar_source = self.volume
                    elif hasattr(self, 'volume') and self.volume is not None:
                        colorbar_source = self.volume
                    else:
                        colorbar_source = self.plot_data

                    self.colorbar = mlab.colorbar(colorbar_source,
                                                  orientation='vertical',
                                                  title='Value',
                                                  nb_labels=5)

                    # Ensure colorbar text is visible
                    if self.colorbar is not None:
                        self.colorbar.label_text_property.color = (0, 0, 0)
                        self.colorbar.title_text_property.color = (0, 0, 0)
                        self.colorbar.label_text_property.font_size = 10
                        self.colorbar.title_text_property.font_size = 12

                except Exception as e:
                    print(f"Warning: Could not create colorbar: {str(e)}")

        except Exception as e:
            print(f"Warning: Scene setup encountered an error: {str(e)}")

    def add_cross_sections(self):
        """Add cross-sections based on toggle settings."""
        if self.show_x_section:
            x_index = int(self.x_slice_pos * (self.x.shape[0] - 1))
            self.x_plane = mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='x_axes',
                slice_index=x_index
            )
            self.x_plane.ipw.opacity = self.opacity

        if self.show_y_section:
            y_index = int(self.y_slice_pos * (self.y.shape[1] - 1))
            self.y_plane = mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='y_axes',
                slice_index=y_index
            )
            self.y_plane.ipw.opacity = self.opacity

        if self.show_z_section:
            z_index = int(self.z_slice_pos * (self.z.shape[2] - 1))
            self.z_plane = mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='z_axes',
                slice_index=z_index
            )
            self.z_plane.ipw.opacity = self.opacity

    def add_multiple_slices(self):
        """Enhanced multiple slice handling with proper updating."""
        try:
            # Clear existing slices first
            # Clear only the slices, not the whole scene
            self.scene.mlab.clf(figure=False)

            if self.enable_multi_slice:
                for num_slices, axis, max_idx in [
                    (self.num_x_slices, 'x', self.x.shape[0]),
                    (self.num_y_slices, 'y', self.y.shape[1]),
                    (self.num_z_slices, 'z', self.z.shape[2])
                ]:
                    if num_slices > 0:
                        spacing = max_idx // (num_slices + 1)
                        for i in range(1, num_slices + 1):
                            slice_widget = mlab.pipeline.image_plane_widget(
                                self.plot_data,
                                plane_orientation=f'{axis}_axes',
                                slice_index=i * spacing
                            )
                            slice_widget.ipw.opacity = self.opacity
                            # Store slice widget for later reference
                            setattr(self, f'{axis}_slice_{i}', slice_widget)
        except Exception as e:
            print(f"Warning: Could not add multiple slices: {str(e)}")

    def _update_volume_opacity(self):
        """Update volume opacity using VTK's transfer functions."""
        if hasattr(self.volume, 'volume'):
            try:
                # Get the volume property
                vol_prop = self.volume.volume.property
                vol_prop.independent_components = True

                # Create and set the opacity transfer function
                otf = vol_prop.get_scalar_opacity()
                otf.remove_all_points()

                # Add opacity points (linear ramp)
                data_min = float(np.nanmin(self.data))
                data_max = float(np.nanmax(self.data))
                data_range = data_max - data_min

                # Add more points for smoother transition
                num_points = 4
                for i in range(num_points):
                    val = data_min + (i * data_range / (num_points - 1))
                    opacity = self.opacity * (i / (num_points - 1))
                    otf.add_point(val, opacity)

                # Add final point for maximum opacity
                otf.add_point(data_max, self.opacity)

            except Exception as e:
                print(f"Warning: Could not update volume opacity: {str(e)}")
                try:
                    self.volume.volume.property.opacity = self.opacity
                except:
                    pass

    def update_visualization(self):
        """Update visualization with improved error handling."""
        if not hasattr(self, 'scene') or self.scene is None:
            print("Warning: Scene not initialized")
            return

        try:
            self.scene.mlab.clf()
            self.scene.background = (1.0, 1.0, 1.0)

            # Ensure data is in correct format
            if not isinstance(self.data, np.ndarray):
                self.data = np.array(self.data, dtype=np.float64)
            self.data = np.nan_to_num(
                self.data, nan=0.0, posinf=0.0, neginf=0.0)

            # Create scalar field
            self.plot_data = mlab.pipeline.scalar_field(
                self.x, self.y, self.z, self.data)

            if self.view_mode == "Standard":
                self.volume = mlab.pipeline.volume(self.plot_data)
                self._update_volume_opacity()

            elif self.view_mode == "Interactive Slicing":
                self.volume = mlab.pipeline.volume(self.plot_data)
                self._update_volume_opacity()
                if self.enable_multi_slice:
                    self.add_multiple_slices()
                self.add_cross_sections()

            elif self.view_mode == "Structure Identification":
                threshold = mlab.pipeline.threshold(
                    self.plot_data, low=self.threshold_value)
                self.volume = mlab.pipeline.surface(threshold)
                if hasattr(self.volume, 'actor'):
                    self.volume.actor.property.opacity = self.opacity
                if self.show_slicing_structure:
                    self.add_multiple_slices()

            # Setup scene (includes colorbar if enabled)
            self.setup_scene()

            # Set title
            try:
                mlab.title(f"{self.title} - {self.view_mode}",
                           size=0.3, height=0.95, color=(0, 0, 0))
            except Exception as e:
                print(f"Warning: Could not set title: {str(e)}")

        except Exception as e:
            print(f"Error in visualization update: {str(e)}")

    @on_trait_change('opacity, threshold_value, show_colorbar, view_mode')
    def _update_on_trait_change(self):
        """Handle updates when traits change."""
        if hasattr(self, 'update_visualization'):
            self.update_visualization()

    @on_trait_change('view_data')
    def update_data(self):
        """Update the displayed data based on the selected data type."""
        if self.view_data == "Anisotropic Data":
            self.set_data(x_unique_a, y_unique_a, z_unique_a,
                          value_3d_a, "Anisotropic Data")
        else:
            self.set_data(x_unique_i, y_unique_i, z_unique_i,
                          value_3d_i, "Isotropic Data")

    @on_trait_change('opacity')
    def _opacity_changed(self):
        """Handle opacity changes."""
        if hasattr(self, 'volume'):
            if hasattr(self.volume, 'volume'):
                self._update_volume_opacity()
            elif hasattr(self.volume, 'actor'):
                self.volume.actor.property.opacity = self.opacity

    @on_trait_change('num_x_slices, num_y_slices, num_z_slices, enable_multi_slice')
    def _update_slices(self):
        """Handle updates to slice controls."""
        if hasattr(self, 'update_visualization'):
            self.update_visualization()

    @on_trait_change('show_colorbar')
    def _update_colorbar(self):
        """Enhanced colorbar toggle handler."""
        try:
            if not self.show_colorbar and hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None
            self.update_visualization()
        except Exception as e:
            print(f"Warning: Could not update colorbar: {str(e)}")

    # Define the view trait
    view = View(
        VGroup(
            HGroup(
                Item('view_mode', label="Visualization Mode"),
                Item('view_data', label="Data Type"),
                Item('update_view', show_label=False),
            ),
            HGroup(
                Item('opacity', label="Opacity"),
                Item('show_colorbar', label="Show Colorbar"),
            ),
            HGroup(
                Item('threshold_value', label="Threshold",
                     visible_when="view_mode=='Structure Identification'"),
            ),
            Group(
                VGroup(
                    HGroup(
                        Item('show_x_section', label="Show X Section"),
                        Item('x_slice_pos', label="X Position",
                             enabled_when="show_x_section==True"),
                    ),
                    HGroup(
                        Item('show_y_section', label="Show Y Section"),
                        Item('y_slice_pos', label="Y Position",
                             enabled_when="show_y_section==True"),
                    ),
                    HGroup(
                        Item('show_z_section', label="Show Z Section"),
                        Item('z_slice_pos', label="Z Position",
                             enabled_when="show_z_section==True"),
                    ),
                ),
                label="Cross Sections",
                visible_when="view_mode=='Interactive Slicing'",
            ),
            Group(
                VGroup(
                    Item('enable_multi_slice', label="Enable Multiple Slices"),
                    HGroup(
                        Item('num_x_slices', label="X Slices",
                             enabled_when="enable_multi_slice==True"),
                        Item('num_y_slices', label="Y Slices",
                             enabled_when="enable_multi_slice==True"),
                        Item('num_z_slices', label="Z Slices",
                             enabled_when="enable_multi_slice==True"),
                    ),
                ),
                label="Multiple Slices",
                visible_when="view_mode=='Interactive Slicing'",
            ),
            Group(
                Item('show_slicing_structure', label="Enable Slicing"),
                Item('num_slices_structure', label="Number of Slices",
                     enabled_when="show_slicing_structure==True"),
                visible_when="view_mode=='Structure Identification'",
            ),
            Group(
                Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=500,
                     width=700,
                     show_label=False),
            ),
        ),
        resizable=True,
        title="Subsurface Visualization Tool",
    )




if __name__ == "__main__":
    viz_tool = VisualizationTool()
    viz_tool.configure_traits()
