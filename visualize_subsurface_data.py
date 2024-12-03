"""
Subsurface Visualization Tool
Created by: Heba Alazzeh
Function: This script loads, processes, and visualizes 3D subsurface data from multiple anisotropic and isotropic CSV files. 
          The tool allows users to toggle between different visualization modes: standard view, interactive slicing, and structure identification.
Last Updated: 11/08/2024
"""

import pandas as pd
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


def generate_sample_data():
    """Generate sample data if files not found"""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    z = np.linspace(0, 10, 20)

    data = []
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            for k, z_val in enumerate(z):
                value = np.sin(x_val) * np.cos(y_val) * np.exp(-z_val/5)
                data.append({
                    'x': x_val,
                    'y': y_val,
                    'z': z_val,
                    'value': value
                })
    return pd.DataFrame(data)


def load_and_combine(files):
    try:
        combined_data = pd.concat([pd.read_csv(file)
                                  for file in files], ignore_index=True)
        return combined_data
    except FileNotFoundError:
        print("Files not found. Using sample data.")
        return generate_sample_data()


# Load and combine the data
anisotropic_data = load_and_combine(anisotropic_files)
isotropic_data = load_and_combine(isotropic_files)


def aggregate_data(data):
    aggregated_data = data.groupby(['x', 'y', 'z'], as_index=False).mean()
    return aggregated_data


anisotropic_data = aggregate_data(anisotropic_data)
isotropic_data = aggregate_data(isotropic_data)


def reshape_data(data):
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


"""
Subsurface Visualization Tool
Created by: Heba Alazzeh
Function: This script loads, processes, and visualizes 3D subsurface data from multiple anisotropic and isotropic CSV files. 
          The tool allows users to toggle between different visualization modes: standard view, interactive slicing, and structure identification.
Last Updated: 08/22/2024
"""


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


def generate_sample_data():
    """Generate sample data if files not found"""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    z = np.linspace(0, 10, 20)

    data = []
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            for k, z_val in enumerate(z):
                value = np.sin(x_val) * np.cos(y_val) * np.exp(-z_val/5)
                data.append({
                    'x': x_val,
                    'y': y_val,
                    'z': z_val,
                    'value': value
                })
    return pd.DataFrame(data)


def load_and_combine(files):
    try:
        combined_data = pd.concat([pd.read_csv(file)
                                  for file in files], ignore_index=True)
        return combined_data
    except FileNotFoundError:
        print("Files not found. Using sample data.")
        return generate_sample_data()


# Load and combine the data
anisotropic_data = load_and_combine(anisotropic_files)
isotropic_data = load_and_combine(isotropic_files)


def aggregate_data(data):
    aggregated_data = data.groupby(['x', 'y', 'z'], as_index=False).mean()
    return aggregated_data


anisotropic_data = aggregate_data(anisotropic_data)
isotropic_data = aggregate_data(isotropic_data)


def reshape_data(data):
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
    scene = Instance(MlabSceneModel, ())
    view_mode = Enum("Standard", "Interactive Slicing",
                     "Structure Identification")
    view_data = Enum("Anisotropic Data", "Isotropic Data")
    update_view = Button("Update View")

    # Enhanced controls
    opacity = Range(0.1, 1.0, 0.8)
    show_colorbar = Bool(True)
    threshold = Range(0.0, 10.0, 4.0)

    # Slicing controls
    slice_x = Range(0.0, 1.0, 0.5)
    slice_y = Range(0.0, 1.0, 0.5)
    slice_z = Range(0.0, 1.0, 0.5)
    show_x_plane = Bool(True)
    show_y_plane = Bool(True)
    show_z_plane = Bool(True)

    def __init__(self):
        super(VisualizationTool, self).__init__()
        self.plot_data = None
        self.x = None
        self.y = None
        self.z = None
        self.data = None
        self.title = ""

    @on_trait_change('scene.activated')
    def initialize_view(self):
        self.set_data(x_unique_a, y_unique_a, z_unique_a,
                      value_3d_a, "Anisotropic Data")

    @on_trait_change('view_data')
    def update_data(self):
        if self.view_data == "Anisotropic Data":
            self.set_data(x_unique_a, y_unique_a, z_unique_a,
                          value_3d_a, "Anisotropic Data")
        elif self.view_data == "Isotropic Data":
            self.set_data(x_unique_i, y_unique_i, z_unique_i,
                          value_3d_i, "Isotropic Data")

    def _view_mode_changed(self):
        self.update_visualization()

    def _update_view_fired(self):
        self.update_visualization()

    def update_visualization(self):
        if not hasattr(self, 'data') or self.data is None:
            return

        self.scene.mlab.clf()

        # Create the scalar field
        self.plot_data = mlab.pipeline.scalar_field(
            self.x, self.y, self.z, self.data)

        # Set up visualization based on mode
        if self.view_mode == "Standard":
            self.plot_standard()
        elif self.view_mode == "Interactive Slicing":
            self.plot_slicing()
        elif self.view_mode == "Structure Identification":
            self.plot_structures()

        # Add common elements
        axes = mlab.axes(self.plot_data, color=(0, 0, 0),
                         xlabel='X', ylabel='Y', zlabel='Z')
        axes.label_text_property.color = (0, 0, 0)
        axes.title_text_property.color = (0, 0, 0)
        axes.label_text_property.font_size = 12
        axes.title_text_property.font_size = 12
        axes.axes.label_format = '%.1f'

        if self.show_colorbar:
            cb = mlab.colorbar(object=self.plot_data,
                               orientation='vertical', title="Value")
            cb.label_text_property.color = (0, 0, 0)

        mlab.title(f"{self.title} - {self.view_mode}",
                   size=0.3, height=0.95, color=(0, 0, 0))
        self.scene.background = (1, 1, 1)

    def plot_standard(self):
        vol = mlab.pipeline.volume(self.plot_data, opacity=self.opacity)
        vol.volume.mapper.blend_mode = 'composite'

    def plot_slicing(self):
        if self.show_x_plane:
            x_idx = int(self.slice_x * (self.x.shape[0]-1))
            mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='x_axes',
                slice_index=x_idx
            )

        if self.show_y_plane:
            y_idx = int(self.slice_y * (self.y.shape[1]-1))
            mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='y_axes',
                slice_index=y_idx
            )

        if self.show_z_plane:
            z_idx = int(self.slice_z * (self.z.shape[2]-1))
            mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='z_axes',
                slice_index=z_idx
            )

    def plot_structures(self):
        threshold = mlab.pipeline.threshold(self.plot_data, low=self.threshold)
        mlab.pipeline.surface(threshold, opacity=self.opacity)

        # Add a reference plane
        if self.show_z_plane:
            z_idx = int(self.slice_z * (self.z.shape[2]-1))
            mlab.pipeline.image_plane_widget(
                self.plot_data,
                plane_orientation='z_axes',
                slice_index=z_idx
            )

    def set_data(self, x, y, z, data, title):
        self.x, self.y, self.z = np.meshgrid(x, y, z, indexing='ij')
        self.data = data
        self.title = title
        self.update_visualization()

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
                Item('threshold', label="Threshold",
                     visible_when="view_mode=='Structure Identification'"),
            ),
            Group(
                VGroup(
                    HGroup(
                        Item('show_x_plane', label="X Plane"),
                        Item('slice_x', label="X Position"),
                    ),
                    HGroup(
                        Item('show_y_plane', label="Y Plane"),
                        Item('slice_y', label="Y Position"),
                    ),
                    HGroup(
                        Item('show_z_plane', label="Z Plane"),
                        Item('slice_z', label="Z Position"),
                    ),
                    visible_when="view_mode in ['Interactive Slicing', 'Structure Identification']"
                ),
            ),
            Group(
                Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=800, show_label=False),
            ),
        ),
        resizable=True,
        title="Subsurface Visualization Tool",
    )


if __name__ == '__main__':
    viz_tool = VisualizationTool()
    viz_tool.configure_traits()
