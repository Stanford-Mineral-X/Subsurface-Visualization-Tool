import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata
import argparse

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

# 3D Contour visualization function


def plot_3d_contours(x, y, z, num_contours=10, opacity=0.5, colormap='viridis'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid for interpolation
    xi = np.linspace(min(x), max(x), 50)
    yi = np.linspace(min(y), max(y), 50)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Determine contour levels
    min_z, max_z = np.nanmin(Z), np.nanmax(Z)
    levels = np.linspace(min_z, max_z, num_contours)

    # Plot contour surfaces
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=colormap, alpha=opacity)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Contour Visualization of Subsurface Data")

    plt.colorbar(contour, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# Main script execution


def main():
    parser = argparse.ArgumentParser(
        description='Visualize subsurface data with 3D contour mapping.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='CSV files containing X, Y, Z data')
    parser.add_argument('--num_contours', type=int,
                        default=10, help='Number of contour levels')
    parser.add_argument('--opacity', type=float,
                        default=0.5, help='Opacity for contours')
    parser.add_argument('--colormap', type=str,
                        default='hawaii', help='Colormap for visualization')

    args = parser.parse_args()
    x, y, z = load_data(args.filenames)
    plot_3d_contours(x, y, z, num_contours=args.num_contours,
                     opacity=args.opacity, colormap=args.colormap)


if __name__ == '__main__':
    main()
