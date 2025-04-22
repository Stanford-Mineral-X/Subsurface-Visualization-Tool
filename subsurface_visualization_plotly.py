import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import argparse
from plotly.subplots import make_subplots

# Load data function


def load_data(filenames):
    all_x, all_y, all_z = [], [], []
    for filename in filenames:
        data = np.loadtxt(filename, delimiter=',',
                          skiprows=1)  # Skip header row
        all_x.extend(data[:, 0])
        all_y.extend(data[:, 1])
        all_z.extend(data[:, 2])
    return np.array(all_x), np.array(all_y), np.array(all_z)

# Apply log transformation if needed


def apply_log_transform(z, use_log):
    if use_log:
        # Make sure all values are positive before taking log
        min_val = np.min(z)
        if min_val <= 0:
            z = z - min_val + 1  # Shift to make all values positive
        return np.log(z)
    return z

# 3D Isosurface visualization function


def plot_3d_isosurface(x, y, z, num_contours=10, opacity=0.5, colormap='Viridis',
                       use_log=False, show_slices=False):
    # Apply log transformation if requested
    z = apply_log_transform(z, use_log)

    # Create a grid for interpolation
    x_range = np.linspace(min(x), max(x), 50)
    y_range = np.linspace(min(y), max(y), 50)
    z_range = np.linspace(min(z), max(z), 50)

    X, Y = np.meshgrid(x_range, y_range)
    Z_grid = griddata((x, y), z, (X, Y), method='cubic')

    # Create a 3D grid for the isosurface
    X_vol, Y_vol, Z_vol = np.meshgrid(x_range, y_range, z_range)

    # Extend the 2D interpolated grid to 3D
    # For each z level, we'll use the same 2D interpolated values
    V = np.zeros(X_vol.shape)
    for i in range(len(z_range)):
        V[:, :, i] = Z_grid

    # Determine isosurface levels
    min_z, max_z = np.nanmin(z), np.nanmax(z)
    levels = np.linspace(min_z, max_z, num_contours)

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
            isomin=min_z,
            isomax=max_z,
            opacity=0,  # Invisible, just for the colorbar
            colorscale=colormap,
            surface_count=1,
            showscale=True,
            caps=dict(x_show=False, y_show=False, z_show=False)
        )
    )

    # Add optional slices if requested
    if show_slices:
        # Add a slice in the middle of each axis
        mid_x = len(x_range) // 2
        mid_y = len(y_range) // 2
        mid_z = len(z_range) // 2

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
                slices_x=dict(show=True, locations=[x_range[mid_x]]),
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
                slices_y=dict(show=True, locations=[y_range[mid_y]]),
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
                slices_z=dict(show=True, locations=[z_range[mid_z]]),
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            )
        )

    # Update layout
    fig.update_layout(
        title="3D Isosurface Visualization of Subsurface Data",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        width=900,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    # Add buttons for interactivity
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"scene.camera": dict(
                            eye=dict(x=1.25, y=1.25, z=1.25))}]
                    ),
                ],
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="left",
                y=0,
                yanchor="top"
            ),
        ]
    )

    # Create a more interactive display
    fig.update_layout(scene_camera_eye=dict(x=1.25, y=1.25, z=1.25))

    return fig

# Main script execution


def main():
    parser = argparse.ArgumentParser(
        description='Visualize subsurface data with 3D isosurface mapping using Plotly.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='CSV files containing X, Y, Z data')
    parser.add_argument('--num_contours', type=int,
                        default=10, help='Number of isosurface levels')
    parser.add_argument('--opacity', type=float,
                        default=0.5, help='Opacity for isosurfaces')
    parser.add_argument('--colormap', type=str,
                        default='hawaii', help='Colormap for visualization')
    parser.add_argument('--use_log', action='store_true',
                        help='Apply log transformation to Z values')
    parser.add_argument('--show_slices', action='store_true',
                        help='Show slices through the volume')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file to save the visualization')

    args = parser.parse_args()
    x, y, z = load_data(args.filenames)

    fig = plot_3d_isosurface(
        x, y, z,
        num_contours=args.num_contours,
        opacity=args.opacity,
        colormap=args.colormap,
        use_log=args.use_log,
        show_slices=args.show_slices
    )

    # Save to HTML file if output is specified
    if args.output:
        fig.write_html(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        # Show interactive plot
        fig.show()


if __name__ == '__main__':
    main()
