"""
Publication-quality 3D visualization of LEO satellite constellations and routing paths.
Generates IEEE Access-style figures with Earth, satellites, ISL links, and routes.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import cm
import os


class Satellite3DVisualizer:
    """
    Creates high-resolution 3D visualizations of satellite networks for publications.
    """
    
    def __init__(self, topology, output_dir='outputs/3d_visualizations'):
        self.topology = topology
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.earth_radius_km = 6371.0
        
    def _draw_earth(self, ax, resolution=50, alpha=0.3):
        """Draw a 3D sphere representing Earth with realistic styling."""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = self.earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Enhanced Earth with gradient-like appearance
        ax.plot_surface(x, y, z, color='#4A90E2', alpha=alpha, 
                       linewidth=0, antialiased=True, shade=True, 
                       edgecolor='none', rcount=resolution, ccount=resolution)
        
        # Add subtle grid lines on Earth
        ax.plot_wireframe(x, y, z, color='white', alpha=0.05, 
                         linewidth=0.3, rcount=10, ccount=10)
    
    def _get_satellite_positions(self):
        """Get all satellite positions as numpy array."""
        positions = []
        sat_ids = []
        
        for sat_id, sat in self.topology.satellites.items():
            pos = sat.get_position()
            positions.append(pos)
            sat_ids.append(sat_id)
        
        return np.array(positions), sat_ids
    
    def _plot_constellation(self, ax, color='gray', alpha=0.3, s=1):
        """Plot all satellites in the constellation with enhanced styling."""
        positions, _ = self._get_satellite_positions()
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=color, alpha=alpha, s=s, depthshade=True,
                  edgecolors='darkgray', linewidths=0.3)
    
    def _plot_isl_links(self, ax, color='gray', alpha=0.1, linewidth=0.3):
        """Plot all inter-satellite links."""
        for u, v, data in self.topology.graph.edges(data=True):
            pos_u = self.topology.satellites[u].get_position()
            pos_v = self.topology.satellites[v].get_position()
            
            ax.plot([pos_u[0], pos_v[0]], 
                   [pos_u[1], pos_v[1]], 
                   [pos_u[2], pos_v[2]],
                   color=color, alpha=alpha, linewidth=linewidth)
    
    
    def _plot_route_with_statistics(self, ax, route, color='red', linewidth=3.0, 
                                    alpha=0.95, label=None, show_hops=True):
        """Plot routing path with visible hop transitions and smooth curves."""
        # Draw main path with enhanced visibility
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            pos_u = self.topology.satellites[u].get_position()
            pos_v = self.topology.satellites[v].get_position()
            
            # Interpolate for smoother curves
            num_points = 20
            t = np.linspace(0, 1, num_points)
            x_interp = pos_u[0] * (1 - t) + pos_v[0] * t
            y_interp = pos_u[1] * (1 - t) + pos_v[1] * t
            z_interp = pos_u[2] * (1 - t) + pos_v[2] * t
            
            if i == 0:
                ax.plot(x_interp, y_interp, z_interp, 
                       color=color, linewidth=linewidth, alpha=alpha, 
                       label=label, zorder=100, solid_capstyle='round')
            else:
                ax.plot(x_interp, y_interp, z_interp,
                       color=color, linewidth=linewidth, alpha=alpha, 
                       zorder=100, solid_capstyle='round')
            
            # Show hop nodes
            if show_hops and i > 0 and i < len(route) - 1:
                ax.scatter(pos_v[0], pos_v[1], pos_v[2],
                          c='white', s=30, marker='o', 
                          edgecolors=color, linewidths=2, zorder=101,
                          alpha=0.9)
    
    def _plot_city_marker(self, ax, sat_id, label, color='gold', marker='*', s=200):
        """Mark source/destination satellites."""
        pos = self.topology.satellites[sat_id].get_position()
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker, 
                  s=s, edgecolors='black', linewidths=1.5, label=label, zorder=1000)
    
    def _setup_publication_axes(self, ax, title='', view_azim=45, view_elev=20, 
                                show_axes=True, zoom_factor=1.0):
        """Configure 3D axes for publication-quality figures."""
        if show_axes:
            ax.set_xlabel('X [km]', fontsize=11, labelpad=10, 
                         fontweight='600', color=self.colors['text'])
            ax.set_ylabel('Y [km]', fontsize=11, labelpad=10,
                         fontweight='600', color=self.colors['text'])
            ax.set_zlabel('Z [km]', fontsize=11, labelpad=10,
                         fontweight='600', color=self.colors['text'])
            
            # Professional tick styling
            ax.tick_params(labelsize=9, colors=self.colors['text'], 
                          pad=5, width=0.8)
        else:
            # Hide axes for cleaner publication figures
            ax.set_axis_off()
        
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20, 
                        color=self.colors['text'], family='sans-serif')
        
        ax.view_init(elev=view_elev, azim=view_azim)
        
        # Optimized view bounds (reduce whitespace)
        max_range = int(7500 / zoom_factor)
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Clean professional grid
        ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.6, 
               color=self.colors['grid'])
        
        # Transparent panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(self.colors['grid'])
        ax.yaxis.pane.set_edgecolor(self.colors['grid'])
        ax.zaxis.pane.set_edgecolor(self.colors['grid'])
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        
        # Set proper aspect ratio
        ax.set_box_aspect([1, 1, 1])
    
    def visualize_route(self, route, src_label='Source', dst_label='Destination',
                       title='LEO Satellite Routing Path', output_filename='route_3d.png',
                       figsize=(12, 9), dpi=300, view_azim=45, view_elev=20,
                       show_constellation=True, show_isl=True, route_color='red'):
        """
        Visualize a single routing path in 3D.
        
        Args:
            route: List of satellite IDs forming the path
            src_label: Label for source city/location
            dst_label: Label for destination city/location
            title: Figure title
            output_filename: Output file name
            figsize: Figure size (width, height)
            dpi: Resolution
            view_azim: Camera azimuth angle
            view_elev: Camera elevation angle
            show_constellation: Show all satellites in gray
            show_isl: Show all ISL links
            route_color: Color for the highlighted route
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set figure background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Draw Earth with enhanced styling
        self._draw_earth(ax, alpha=0.35, resolution=60)
        
        # Draw constellation
        if show_constellation:
            self._plot_constellation(ax, color='#7F8C8D', alpha=0.35, s=8)
        
        # Draw ISL links
        if show_isl:
            self._plot_isl_links(ax, color='#BDC3C7', alpha=0.08, linewidth=0.4)
        
        # Draw the routing path with glow effect
        self._plot_route(ax, route, color=route_color, linewidth=3.5, alpha=0.95, label='Routing Path')
        
        # Mark source and destination with better colors
        self._plot_city_marker(ax, route[0], src_label, color='#2ECC71', marker='o', s=300)
        self._plot_city_marker(ax, route[-1], dst_label, color='#F39C12', marker='s', s=300)
        
        # Setup axes
        self._setup_axes(ax, title=title, view_azim=view_azim, view_elev=view_elev)
        
        # Enhanced legend
        legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
                          edgecolor='#BDC3C7', fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                   edgecolor='none', transparent=False)
        plt.close()
        
        print(f"✓ 3D route visualization saved to {filepath}")
        return filepath
    
    def visualize_multiple_routes(self, routes_dict, title='Multi-Route Comparison',
                                  output_filename='multi_route_3d.png', figsize=(14, 10),
                                  dpi=300, view_azim=45, view_elev=20):
        """
        Visualize multiple routes with different colors for comparison.
        
        Args:
            routes_dict: Dictionary {label: {'route': [sat_ids], 'color': 'color'}}
            title: Figure title
            output_filename: Output file name
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Earth
        self._draw_earth(ax, alpha=0.25)
        
        # Draw constellation
        self._plot_constellation(ax, color='gray', alpha=0.3, s=4)
        
        # Draw routes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for idx, (label, info) in enumerate(routes_dict.items()):
            route = info['route']
            color = info.get('color', colors[idx % len(colors)])
            self._plot_route(ax, route, color=color, linewidth=2.5, alpha=0.9, label=label)
            
            # Mark endpoints for first route only (to avoid clutter)
            if idx == 0:
                self._plot_city_marker(ax, route[0], 'Source', color='lime', marker='o', s=200)
                self._plot_city_marker(ax, route[-1], 'Destination', color='gold', marker='s', s=200)
        
        # Setup axes
        self._setup_axes(ax, title=title, view_azim=view_azim, view_elev=view_elev)
        
        # Legend
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Multi-route 3D visualization saved to {filepath}")
        return filepath
    
    def visualize_constellation_snapshot(self, timestamp=None, title='LEO Constellation Snapshot',
                                        output_filename='constellation_3d.png', 
                                        figsize=(12, 9), dpi=300, view_azim=45, view_elev=20,
                                        show_isl=True, highlight_nodes=None):
        """
        Visualize the entire constellation at a specific timestamp.
        
        Args:
            timestamp: Optional timestamp label for the title
            title: Base title
            output_filename: Output file
            show_isl: Show inter-satellite links
            highlight_nodes: List of node IDs to highlight
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Earth
        self._draw_earth(ax, alpha=0.3)
        
        # Set background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Draw constellation
        positions, sat_ids = self._get_satellite_positions()
        
        # Regular satellites with better styling
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='#3498DB', alpha=0.65, s=25, depthshade=True, 
                  label='Satellites', edgecolors='#2C3E50', linewidths=0.6)
        
        # Highlight specific nodes
        if highlight_nodes:
            highlight_pos = np.array([self.topology.satellites[nid].get_position() 
                                     for nid in highlight_nodes])
            ax.scatter(highlight_pos[:, 0], highlight_pos[:, 1], highlight_pos[:, 2],
                      c='#E74C3C', alpha=1.0, s=120, marker='D', edgecolors='#C0392B',
                      linewidths=1.8, label='Highlighted Nodes', zorder=1000)
        
        # Draw ISL links
        if show_isl:
            self._plot_isl_links(ax, color='#95A5A6', alpha=0.12, linewidth=0.5)
        
        # Setup axes
        full_title = title
        if timestamp is not None:
            full_title += f' (t={timestamp:.1f}s)'
        
        self._setup_axes(ax, title=full_title, view_azim=view_azim, view_elev=view_elev)
        
        # Enhanced legend
        legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
                          edgecolor='#BDC3C7', fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                   edgecolor='none', transparent=False)
        plt.close()
        
        print(f"✓ Constellation snapshot saved to {filepath}")
        return filepath
    
    def create_route_comparison_figure(self, baseline_route, drl_route, lstm_route, gru_route,
                                      title='Routing Method Comparison',
                                      output_filename='route_comparison_3d.png',
                                      figsize=(16, 12), dpi=300):
        """
        Create a publication figure comparing routes from different methods.
        
        Args:
            baseline_route: Route from Dijkstra/OSPF
            drl_route: Route from DRL agent
            lstm_route: Route with LSTM prediction
            gru_route: Route with GRU prediction
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        methods = [
            ('Dijkstra Baseline', baseline_route, 'gray', 221),
            ('DRL Agent', drl_route, 'red', 222),
            ('DRL + LSTM Prediction', lstm_route, 'blue', 223),
            ('DRL + GRU Prediction', gru_route, 'green', 224)
        ]
        
        for method_name, route, color, subplot_idx in methods:
            if route is None:
                continue
            
            ax = fig.add_subplot(subplot_idx, projection='3d')
            
            # Draw Earth
            self._draw_earth(ax, alpha=0.2, resolution=30)
            
            # Draw constellation (lighter)
            self._plot_constellation(ax, color='lightgray', alpha=0.2, s=3)
            
            # Draw route
            self._plot_route(ax, route, color=color, linewidth=2.5, alpha=0.95)
            
            # Mark endpoints
            self._plot_city_marker(ax, route[0], '', color='lime', marker='o', s=150)
            self._plot_city_marker(ax, route[-1], '', color='gold', marker='s', s=150)
            
            # Setup axes with smaller labels
            ax.set_xlabel('X [km]', fontsize=9)
            ax.set_ylabel('Y [km]', fontsize=9)
            ax.set_zlabel('Z [km]', fontsize=9)
            ax.set_title(f'{method_name}\n({len(route)} hops)', fontsize=11, fontweight='bold')
            
            ax.view_init(elev=20, azim=45)
            
            max_range = 8000
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            ax.grid(True, alpha=0.15)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Route comparison figure saved to {filepath}")
        return filepath
    
    def export_to_svg(self, route, output_filename='route_3d.svg', **kwargs):
        """Export visualization to SVG format for vector editing."""
        kwargs['output_filename'] = output_filename
        kwargs['dpi'] = 150  # SVG doesn't need high DPI
        
        filepath = self.visualize_route(route, **kwargs)
        return filepath.replace('.png', '.svg')
    
    def export_to_pdf(self, route, output_filename='route_3d.pdf', **kwargs):
        """Export visualization to PDF format for publications."""
        kwargs['output_filename'] = output_filename
        
        filepath = self.visualize_route(route, **kwargs)
        return filepath.replace('.png', '.pdf')
