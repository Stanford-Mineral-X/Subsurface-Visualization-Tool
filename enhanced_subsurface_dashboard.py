import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from scipy.interpolate import griddata
import argparse
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import os
import glob
import pandas as pd

# Global variables to store loaded data
loaded_data = {}
current_dataset = None

# List of built-in Plotly colormaps that work well for this visualization
COLORMAP_OPTIONS = [
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma', 'value': 'Plasma'},
    {'label': 'Inferno', 'value': 'Inferno'},
    {'label': 'Turbo', 'value': 'Turbo'},
    {'label': 'Jet', 'value': 'Jet'},
    {'label': 'Hot', 'value': 'Hot'},
    {'label': 'Cividis', 'value': 'Cividis'},
    {'label': 'Blues', 'value': 'Blues'},
    {'label': 'Greens', 'value': 'Greens'},
    {'label': 'Reds', 'value': 'Reds'},
    {'label': 'YlOrRd', 'value': 'YlOrRd'},
    {'label': 'YlGnBu', 'value': 'YlGnBu'},
]

# Load data function for your specific CSV format


def load_data_file(filename):
    """
    Load data from a CSV file with x, y, z, value columns.
    """
    try:
        # Using pandas for more robust CSV handling
        df = pd.read_csv(filename)

        # Check if the dataframe has the expected columns
        if not all(col in df.columns for col in ['x', 'y', 'z', 'value']):
            print(
                f"Warning: File {filename} does not have the expected columns (x, y, z, value)")
            # Try to use the first 4 columns if they exist
            if len(df.columns) >= 4:
                df.columns = ['x', 'y', 'z', 'value'] + list(df.columns[4:])
                print(f"Assuming first 4 columns are x, y, z, value")
            else:
                raise ValueError(
                    f"File {filename} does not have enough columns")

        # Extract the data
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        values = df['value'].values

        print(f"Successfully loaded {len(x)} data points from {filename}")
        return x, y, z, values

    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None, None, None, None

# Load all data files in directory


def discover_data_files(directory='.', pattern='*.csv'):
    """
    Discover and categorize all data files in the given directory.
    """
    data_files = {}

    # Get all CSV files
    all_files = glob.glob(os.path.join(directory, pattern))

    for file_path in all_files:
        filename = os.path.basename(file_path)

        # Skip files that don't look like data files
        if not ('plot_data' in filename or 'subsurface' in filename):
            continue

        # Try to categorize the file
        if 'anisotropic' in filename:
            category = 'Anisotropic'
        elif 'isotropic' in filename:
            category = 'Isotropic'
        else:
            category = 'Other'

        # Get the file number if present
        import re
        num_match = re.search(r'(\d+)', filename)
        if num_match:
            file_num = int(num_match.group(1))
            display_name = f"{category} Dataset {file_num}"
        else:
            display_name = filename

        # Add to our data files dictionary
        if category not in data_files:
            data_files[category] = []

        data_files[category].append({
            'path': file_path,
            'filename': filename,
            'display_name': display_name,
            'data': None  # Will be loaded on demand
        })

    # Sort each category
    for category in data_files:
        data_files[category] = sorted(
            data_files[category], key=lambda x: x['display_name'])

    return data_files

# Apply log transformation if needed


def apply_log_transform(values, use_log):
    if use_log:
        # Make sure all values are positive before taking log
        min_val = np.min(values)
        if min_val <= 0:
            # Add a small offset to make all values positive
            offset = abs(min_val) + 1
            values = values + offset
        return np.log(values)
    return values


# Create a Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = html.Div([
    html.H1("Interactive Subsurface Visualization Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30, 'marginTop': 20}),

    dbc.Row([
        # Control panel
        dbc.Col([
            html.Div([
                html.H4("Dataset Selection"),

                # Dataset selector
                html.Div(id='dataset-selector-container'),

                html.Hr(),

                html.H4("Visualization Controls"),

                html.Label("Number of Isosurfaces"),
                dcc.Slider(
                    id='num-contours-slider',
                    min=3,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(3, 21, 3)},
                ),

                html.Label("Opacity", style={'marginTop': 20}),
                dcc.Slider(
                    id='opacity-slider',
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(1, 11, 2)},
                ),

                html.Label("Colormap", style={'marginTop': 20}),
                dcc.Dropdown(
                    id='colormap-dropdown',
                    options=COLORMAP_OPTIONS,
                    value='Viridis',
                ),

                html.Div([
                    dbc.Checkbox(
                        id='log-transform-checkbox',
                        label="Apply Log Transformation",
                        value=False
                    ),
                ], style={'marginTop': 20}),

                html.Div([
                    dbc.Checkbox(
                        id='show-slices-checkbox',
                        label="Show Volume Slices",
                        value=False
                    ),
                ], style={'marginTop': 10}),

                html.Div(id='slice-controls', style={'display': 'none'}, children=[
                    html.Label("X Slice Position", style={'marginTop': 15}),
                    dcc.Slider(
                        id='x-slice-slider',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                    ),

                    html.Label("Y Slice Position", style={'marginTop': 15}),
                    dcc.Slider(
                        id='y-slice-slider',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                    ),

                    html.Label("Z Slice Position", style={'marginTop': 15}),
                    dcc.Slider(
                        id='z-slice-slider',
                        min=0,
                        max=1,
                        step=0.05,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11, 2)},
                    ),
                ]),

                # Status message area
                html.Div(id='status-message',
                         style={'marginTop': 20, 'color': 'blue', 'fontStyle': 'italic'}),

            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], width=3),

        # Visualization panel
        dbc.Col([
            dcc.Graph(
                id='isosurface-plot',
                style={'height': '80vh'},
            )
        ], width=9),
    ]),

    # Store component to save current dataset
    dcc.Store(id='current-dataset-store'),

    # Store discovered data files
    dcc.Store(id='data-files-store'),
])

# Initialize dataset selector


@app.callback(
    Output('dataset-selector-container', 'children'),
    Output('data-files-store', 'data'),
    Input('dataset-selector-container', 'id')  # Dummy input to trigger on load
)
def initialize_dataset_selector(dummy):
    # Discover data files
    data_files = discover_data_files()

    if not data_files:
        return html.Div("No data files found."), {}

    # Create category and dataset dropdowns
    categories = list(data_files.keys())

    selectors = [
        html.Label("Category"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in categories],
            value=categories[0] if categories else None,
            clearable=False
        ),

        html.Label("Dataset", style={'marginTop': 10}),
        dcc.Dropdown(
            id='dataset-dropdown',
            # Options will be set by the callback
            clearable=False
        )
    ]

    return html.Div(selectors), data_files

# Update dataset dropdown when category changes


@app.callback(
    Output('dataset-dropdown', 'options'),
    Output('dataset-dropdown', 'value'),
    Input('category-dropdown', 'value'),
    State('data-files-store', 'data')
)
def update_dataset_dropdown(category, data_files):
    if not category or not data_files:
        return [], None

    dataset_options = [
        {'label': dataset['display_name'], 'value': i}
        for i, dataset in enumerate(data_files[category])
    ]

    return dataset_options, 0 if dataset_options else None

# Load selected dataset


@app.callback(
    Output('current-dataset-store', 'data'),
    Output('status-message', 'children'),
    Input('dataset-dropdown', 'value'),
    Input('category-dropdown', 'value'),
    State('data-files-store', 'data')
)
def load_selected_dataset(dataset_idx, category, data_files):
    if dataset_idx is None or category is None or not data_files:
        return None, "No dataset selected"

    dataset = data_files[category][dataset_idx]
    filepath = dataset['path']

    # Check if we've already loaded this dataset
    global loaded_data, current_dataset
    if filepath in loaded_data:
        current_dataset = loaded_data[filepath]
        return filepath, f"Using cached dataset: {dataset['display_name']}"

    # Load the data
    x, y, z, values = load_data_file(filepath)

    if x is None:
        return None, f"Failed to load dataset: {dataset['display_name']}"

    # Store the loaded data
    loaded_data[filepath] = {
        'x': x,
        'y': y,
        'z': z,
        'values': values,
        'display_name': dataset['display_name']
    }

    current_dataset = loaded_data[filepath]
    return filepath, f"Loaded dataset: {dataset['display_name']} ({len(x)} points)"

# Show/hide slice controls based on checkbox


@app.callback(
    Output('slice-controls', 'style'),
    Input('show-slices-checkbox', 'value')
)
def toggle_slice_controls(show_slices):
    if show_slices:
        return {'display': 'block'}
    return {'display': 'none'}

# Update the plot when controls change


@app.callback(
    Output('isosurface-plot', 'figure'),
    [Input('num-contours-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('colormap-dropdown', 'value'),
     Input('log-transform-checkbox', 'value'),
     Input('show-slices-checkbox', 'value'),
     Input('x-slice-slider', 'value'),
     Input('y-slice-slider', 'value'),
     Input('z-slice-slider', 'value'),
     Input('current-dataset-store', 'data')]
)
def update_isosurface_plot(num_contours, opacity, colormap, use_log, show_slices,
                           x_slice_pos, y_slice_pos, z_slice_pos, dataset_path):
    global current_dataset

    # Check if we have a dataset to visualize
    if dataset_path is None or dataset_path not in loaded_data:
        # Create empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="No dataset selected",
            annotations=[
                dict(
                    text="Please select a dataset to visualize",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ]
        )
        return fig

    # Get the current dataset
    dataset = loaded_data[dataset_path]
    x = dataset['x']
    y = dataset['y']
    z = dataset['z']
    values = dataset['values']

    # Apply log transformation if requested
    transformed_values = apply_log_transform(values.copy(), use_log)

    # Create a grid for interpolation
    # Determine grid size based on data size (balance between detail and performance)
    # Cube root as heuristic
    grid_size = min(50, max(20, int(len(x) ** (1/3))))

    x_range = np.linspace(min(x), max(x), grid_size)
    y_range = np.linspace(min(y), max(y), grid_size)
    z_range = np.linspace(min(z), max(z), grid_size)

    # Create 3D grid for the isosurface
    X_vol, Y_vol, Z_vol = np.meshgrid(x_range, y_range, z_range)

    # We need to interpolate our scattered data onto a regular grid
    # This is computationally intensive, so we'll use a simple approach
    print(
        f"Creating 3D interpolation grid of size {grid_size}x{grid_size}x{grid_size}")

    # Initialize empty volume
    V = np.zeros(X_vol.shape)
    V[:] = np.nan  # Fill with NaN initially

    # Interpolate values onto the 3D grid
    # For each z-slice, perform 2D interpolation
    for i, z_val in enumerate(z_range):
        # Find points close to this z-level
        z_mask = np.abs(z - z_val) < (max(z) - min(z)) / grid_size

        if np.sum(z_mask) > 3:  # Need at least 3 points for interpolation
            # Perform 2D interpolation for this z-slice
            V[:, :, i] = griddata(
                (x[z_mask], y[z_mask]),
                transformed_values[z_mask],
                (X_vol[:, :, i], Y_vol[:, :, i]),
                method='linear',
                fill_value=np.nan
            )

    # Fill NaN values with a default value
    V = np.nan_to_num(V, nan=np.nanmin(V) if not np.all(np.isnan(V)) else 0)

    # Determine isosurface levels
    min_val, max_val = np.nanmin(V), np.nanmax(V)
    levels = np.linspace(min_val, max_val, num_contours)

    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'volume'}]])

    # Add isosurfaces
    for level in levels:
        fig.add_trace(
            go.Isosurface(
                x=X_vol.flatten(),
                y=Y_vol.flatten(),
                z=Z_vol.flatten(),
                value=V.flatten(),
                isomin=level,
                isomax=level,
                opacity=opacity,
                surface_count=1,
                colorscale=colormap,
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        )

    # Add a colorbar
    fig.add_trace(
        go.Isosurface(
            x=X_vol.flatten(),
            y=Y_vol.flatten(),
            z=Z_vol.flatten(),
            value=V.flatten(),
            isomin=min_val,
            isomax=max_val,
            opacity=0,  # Invisible, just for the colorbar
            colorscale=colormap,
            surface_count=1,
            showscale=True,
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
    )

    # Add optional slices if requested
    if show_slices:
        # Calculate slice positions
        x_pos = x_range[int(x_slice_pos * (len(x_range)-1))]
        y_pos = y_range[int(y_slice_pos * (len(y_range)-1))]
        z_pos = z_range[int(z_slice_pos * (len(z_range)-1))]

        # X slice
        fig.add_trace(
            go.Volume(
                x=X_vol.flatten(),
                y=Y_vol.flatten(),
                z=Z_vol.flatten(),
                value=V.flatten(),
                opacity=0.7,
                surface_count=1,
                colorscale=colormap,
                slices_x=dict(show=True, locations=[x_pos]),
                slices_y=dict(show=False),
                slices_z=dict(show=False),
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        )

        # Y slice
        fig.add_trace(
            go.Volume(
                x=X_vol.flatten(),
                y=Y_vol.flatten(),
                z=Z_vol.flatten(),
                value=V.flatten(),
                opacity=0.7,
                surface_count=1,
                colorscale=colormap,
                slices_x=dict(show=False),
                slices_y=dict(show=True, locations=[y_pos]),
                slices_z=dict(show=False),
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        )

        # Z slice
        fig.add_trace(
            go.Volume(
                x=X_vol.flatten(),
                y=Y_vol.flatten(),
                z=Z_vol.flatten(),
                value=V.flatten(),
                opacity=0.7,
                surface_count=1,
                colorscale=colormap,
                slices_x=dict(show=False),
                slices_y=dict(show=False),
                slices_z=dict(show=True, locations=[z_pos]),
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        )

    # Update layout
    fig.update_layout(
        title=f"3D Isosurface Visualization: {dataset['display_name']}",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Create a more interactive display
    fig.update_layout(scene_camera_eye=dict(x=1.25, y=1.25, z=1.25))

    return fig


# Main entry point
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run interactive dashboard for subsurface visualization.')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory containing CSV data files')

    args = parser.parse_args()

    # Run the app
    print(f"Starting dashboard on port {args.port}")
    app.run(debug=args.debug, port=args.port)
