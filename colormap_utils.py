import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.colors


def get_cmcrameri_colormap(name='hawaii', n_colors=256):
    """
    Load a colormap from the cmcrameri package and convert it to a format usable by Plotly.
    
    Parameters:
    -----------
    name : str
        Name of the colormap to load from cmcrameri package
    n_colors : int
        Number of colors to use in the colormap
        
    Returns:
    --------
    list
        List of colors in the format expected by Plotly
    """
    try:
        import cmcrameri.cm as cmc

        # Get the colormap from cmcrameri
        cmap = getattr(cmc, name)

        # Sample the colormap
        colors = cmap(np.linspace(0, 1, n_colors))

        # Convert to hex format for Plotly
        hex_colors = [plotly.colors.rgb_to_hex(rgb) for rgb in colors[:, :3]]

        # Create a list of colorscale tuples (position, color)
        colorscale = [(i/(n_colors-1), color)
                      for i, color in enumerate(hex_colors)]

        return colorscale

    except ImportError:
        print("cmcrameri package not found. Please install with 'pip install cmcrameri'")
        print("Using default Plotly colormap instead.")
        # Return a default Plotly colormap if cmcrameri is not available
        return px.colors.sequential.Viridis


def list_available_cmcrameri_colormaps():
    """
    List all available colormaps from the cmcrameri package.
    
    Returns:
    --------
    list
        List of colormap names available in cmcrameri
    """
    try:
        import cmcrameri.cm as cmc
        import inspect

        # Get all attributes of the module
        all_attributes = dir(cmc)

        # Filter for colormaps (those that are callable)
        colormaps = [attr for attr in all_attributes
                     if not attr.startswith('_') and callable(getattr(cmc, attr))]

        return sorted(colormaps)

    except ImportError:
        print("cmcrameri package not found. Please install with 'pip install cmcrameri'")
        return []


def register_cmcrameri_colormaps_with_plotly():
    """
    Register all cmcrameri colormaps with Plotly for easy use.
    
    Returns:
    --------
    dict
        Dictionary mapping colormap names to their Plotly colorscales
    """
    try:
        import cmcrameri.cm as cmc

        colormap_names = list_available_cmcrameri_colormaps()
        colorscales = {}

        for name in colormap_names:
            colorscales[name] = get_cmcrameri_colormap(name)

        # Register with plotly
        for name, colorscale in colorscales.items():
            plotly.colors.sequential.__dict__[name] = colorscale

        return colorscales

    except ImportError:
        print("cmcrameri package not found. Please install with 'pip install cmcrameri'")
        return {}


def preview_colormap(name='hawaii', colorscale=None, n_colors=256):
    """
    Create a simple plot to preview a colormap.
    
    Parameters:
    -----------
    name : str
        Name of the colormap to preview
    colorscale : list, optional
        Plotly colorscale to preview (if provided, name is ignored)
    n_colors : int
        Number of colors to display
    """
    import matplotlib.pyplot as plt

    if colorscale is None:
        colorscale = get_cmcrameri_colormap(name, n_colors)

    # Extract colors
    colors = [color for _, color in colorscale]

    # Create a simple display
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(colors):
        plt.axvspan(i, i+1, color=color)
    plt.xlim(0, len(colors))
    plt.ylim(0, 1)
    plt.title(f"Colormap: {name}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Available colormaps in cmcrameri:")
    colormaps = list_available_cmcrameri_colormaps()
    if colormaps:
        for cmap in colormaps:
            print(f"  - {cmap}")

        # Preview the 'hawaii' colormap
        preview_colormap('hawaii')

    print("\nRegistering colormaps with Plotly...")
    register_cmcrameri_colormaps_with_plotly()
    print("Done. Colormaps are now available in plotly.colors.sequential.")
