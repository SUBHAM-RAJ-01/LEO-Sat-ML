# Advanced 3D Visualization Guide

## Overview

The `AdvancedSatellite3DVisualizer` generates publication-ready figures suitable for academic journals and conferences.

## Key Features

### ✅ Realistic Earth Model
- **Land/Ocean differentiation** using pseudo-elevation model
- **Deep ocean blue** (#1a4d6e) and **land green** (#2d5a3d) colors
- **High resolution** (100-120 segments) for smooth appearance
- **Subtle meridians** for geographic context
- **90% opacity** - Earth is prominent without obscuring satellites

### ✅ Professional Satellite Constellation
- Small, consistent markers (5-6 pixels)
- Light gray (#7f8c8d) with 20-25% transparency
- Optional ISL links (very subtle, 3-5% opacity)
- Clear orbital plane visualization

### ✅ Enhanced Routing Path
- **Smooth interpolated curves** (25 points per segment)
- **Bright red** (#e74c3c) high-contrast color
- **3.8 px linewidth** - clearly visible
- **White hop markers** with red edges for intermediate nodes
- **Green circle** (source) and **Orange square** (destination)

### ✅ Route Statistics in Title
Automatically displays:
- Source → Destination cities
- Hop count
- End-to-end delay (ms)
- Total distance (km)

Example: `New York → Singapore | Hops: 7 | Delay: 15.8 ms | Distance: 42,150 km`

### ✅ Publication-Ready Formatting
- **600 DPI** resolution (IEEE Access standard)
- **Optimized camera angles** (azimuth 50°, elevation 25°)
- **Reduced whitespace** with zoom factor 1.15
- **Clean axis-free option** for cleaner figures
- **Professional fonts** (sans-serif, proper sizing)
- **Consistent line widths** and styling

### ✅ Multi-Panel Layouts
Generate comparison figures with (a), (b), (c) panels showing:
- Different routes
- Different routing algorithms
- Different network states
- Consistent camera perspectives

## Usage

### Single Route (Publication Quality)

```bash
C:\Users\srajp\AppData\Local\Programs\Python\Python312\python.exe run_prediction.py --visualize
```

Generates:
- `outputs/3d_viz/route_Doha_London.png`
- `outputs/3d_viz/route_Tokyo_Paris.png`
- `outputs/3d_viz/route_New_York_Singapore.png`

### Multi-Panel Comparison

Automatically generated with `--visualize`:
- `outputs/3d_viz/comparison_multi_panel.png`

Shows 3 routes side-by-side with consistent formatting.

## Programmatic Usage

```python
from src.visualization.satellite_3d_advanced import AdvancedSatellite3DVisualizer
from src.env.topology import SatelliteNetwork
import networkx as nx

# Initialize
topology = SatelliteNetwork()
viz = AdvancedSatellite3DVisualizer(topology, output_dir='outputs/my_figures')

# Compute route
route = nx.shortest_path(topology.graph, source=10, target=150, weight='delay_s')

# Generate publication-quality figure
viz.visualize_route_publication(
    route=route,
    src_label='New York',
    dst_label='London',
    output_filename='ny_london.png',
    figsize=(10, 8),
    dpi=600,  # Publication standard
    show_constellation=True,
    show_isl=False,  # Cleaner look
    show_axes=False  # Publication-ready (no axes)
)

# Multi-panel comparison
routes_dict = {
    'Baseline': {'route': route1, 'src': 'NYC', 'dst': 'LON'},
    'DRL': {'route': route2, 'src': 'NYC', 'dst': 'LON'},
    'LSTM-Aware': {'route': route3, 'src': 'NYC', 'dst': 'LON'}
}

viz.visualize_multi_route_comparison(
    routes_dict=routes_dict,
    output_filename='comparison_3panel.png',
    figsize=(18, 6),
    dpi=600
)
```

## Comparison: Before vs After

### Before (Basic Visualization)
- Simple blue sphere Earth
- High transparency (hard to see)
- No route statistics
- Basic styling
- 300 DPI

### After (Publication Quality)
- Realistic Earth with land/ocean
- 90% opacity (clearly visible)
- Route stats in title (hops, delay, distance)
- Professional color scheme
- 600 DPI
- Multi-panel layouts
- Axis-free option
- Smooth interpolated paths
- Hop markers

## File Outputs

### Format Options
- **PNG** (default) - 600 DPI, publication-ready
- **PDF** - Vector format (modify code: `plt.savefig(..., format='pdf')`)
- **SVG** - Scalable vector (modify code: `plt.savefig(..., format='svg')`)

### File Sizes (Typical)
- Single route: ~5-8 MB (600 DPI PNG)
- Multi-panel: ~12-15 MB (600 DPI PNG)
- PDF: ~2-3 MB (vector)

## Best Practices for Publications

### Figure Preparation Checklist

✅ **Resolution**: 600 DPI minimum  
✅ **Format**: PNG or PDF (not JPEG)  
✅ **File size**: < 20 MB per figure  
✅ **Font size**: 10-12 pt readable  
✅ **Line width**: 2-4 pt for main elements  
✅ **Color**: Colorblind-friendly palette  
✅ **Labels**: All axes, legends, panels clearly labeled  
✅ **Caption**: Descriptive caption in LaTeX  
✅ **Quality**: No pixelation or artifacts  

### LaTeX Integration

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/route_Doha_London.png}
    \caption{LEO satellite routing path from Doha to London. The route
    traverses 8 hops with an end-to-end delay of 22.5 ms over a total
    distance of 5,470 km. Gray dots represent the constellation, red path
    shows the computed route, green circle marks the source, and orange
    square marks the destination.}
    \label{fig:doha_london}
\end{figure}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/comparison_multi_panel.png}
    \caption{Comparison of routing paths under different algorithms: 
    (a) Dijkstra baseline, (b) DRL-based routing, (c) LSTM prediction-aware
    routing. All routes share the same source and destination but differ
    in hop count and latency based on congestion awareness.}
    \label{fig:comparison}
\end{figure*}
```

## Configuration Options

### Earth Styling
```python
# In satellite_3d_ieee.py
resolution = 120  # Higher = smoother (80-150)
alpha = 0.90      # Opacity (0.85-0.95)
```

### Route Styling
```python
linewidth = 3.8   # Route thickness (3.0-4.5)
alpha = 0.98      # Route opacity (0.95-1.0)
```

### Camera Angles
```python
view_azim = 50    # Horizontal rotation (30-70)
view_elev = 25    # Vertical angle (15-35)
zoom = 1.15       # Zoom level (1.0-1.3)
```

### Colors (IEEE Professional Palette)
```python
'earth_ocean': '#1a4d6e',     # Deep blue
'earth_land': '#2d5a3d',      # Forest green
'route_path': '#e74c3c',      # Bright red
'source': '#27ae60',          # Green
'destination': '#f39c12',     # Orange
```

## Troubleshooting

### Issue: Earth too transparent
**Solution**: Increase alpha in `_draw_realistic_earth`:
```python
self._draw_realistic_earth(ax, resolution=120, alpha=0.95)
```

### Issue: Route hard to see
**Solution**: Increase linewidth and contrast:
```python
linewidth=4.5, alpha=1.0
```

### Issue: File size too large
**Solution**: Reduce DPI or resolution:
```python
dpi=300, resolution=80
```

### Issue: Satellites obscure route
**Solution**: Reduce constellation opacity:
```python
self._plot_constellation(ax, alpha=0.15, s=4)
```

## Summary

✅ **Realistic Earth** - Land/ocean differentiation  
✅ **Professional styling** - IEEE color scheme  
✅ **High resolution** - 600 DPI publication-ready  
✅ **Route statistics** - Hops, delay, distance in title  
✅ **Multi-panel** - Comparison layouts (a), (b), (c)  
✅ **Clean formatting** - Axis-free option  
✅ **Smooth paths** - Interpolated curves  
✅ **Hop markers** - Visible intermediate nodes  

**Perfect for IEEE Access, IEEE Trans. on Networking, and satellite communication journals!**
