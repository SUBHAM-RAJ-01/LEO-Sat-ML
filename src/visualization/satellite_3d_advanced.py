"""
Advanced publication-quality 3D visualization for LEO satellite routing.
Features: Realistic Earth, orbital planes, route statistics, multi-panel layouts.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os


class AdvancedSatellite3DVisualizer:
    """Advanced publication-quality satellite network visualization with realistic Earth and professional styling."""
    
    def __init__(self, topology, output_dir='outputs/3d_viz_publication'):
        self.topology = topology
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.earth_radius_km = 6371.0
        
        # Brighter color scheme requested by user
        self.colors = {
            'earth_ocean': '#95a5a6',
            'earth_land': '#8cb38a',
            'constellation': '#d1d8e0',  # Brighter satellite nodes
            'isl_links': '#a4b0be',      # Brighter links
            'route_path': '#ff3838',     # Brighter red for route
            'route_glow': '#ff0000',
            'source': '#32ff7e',         # Bright green source
            'destination': '#18dcff',    # Bright cyan destination
            'hop_marker': '#fff200',     # Bright yellow hop fill
            'hop_edge': '#ff3838',       # Red hop edge
            'grid': '#cccccc',
            'text': '#000000',
            'label_bg': '#ffffff',
        }
    
    def _draw_realistic_earth(self, ax, resolution=100, alpha=0.90):
        """Draw realistic Earth with land/ocean differentiation."""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = self.earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Procedural realistic Earth (lighter ocean/land colors)
        colors_matrix = np.zeros((resolution, resolution, 4))
        for i in range(resolution):
            for j in range(resolution):
                lat = np.arcsin(np.clip(z[i, j] / self.earth_radius_km, -1, 1))
                lon = np.arctan2(y[i, j], x[i, j])
                
                # Complex pseudo-elevation for continent-like blobs
                elev = np.sin(4 * lat) * np.cos(5 * lon) + \
                       0.6 * np.sin(7 * lat) * np.cos(3 * lon) + \
                       0.3 * np.cos(12 * lon)
                
                if elev > 0.2:
                    # Land (realistic earthy green/brown, slightly transparent)
                    colors_matrix[i, j] = [0.45, 0.65, 0.40, alpha]
                else:
                    # Ocean (realistic light blue)
                    colors_matrix[i, j] = [0.35, 0.55, 0.75, alpha]
        
        ax.plot_surface(x, y, z, facecolors=colors_matrix,
                       linewidth=0, antialiased=True, shade=True,
                       rcount=resolution, ccount=resolution, zorder=1)
        
        # Subtle meridians
        ax.plot_wireframe(x, y, z, color='white', alpha=0.02,
                         linewidth=0.3, rcount=20, ccount=40, zorder=2)
    
    def _get_satellite_positions(self):
        """Get all satellite positions."""
        positions = []
        sat_ids = []
        for sat_id, sat in self.topology.satellites.items():
            positions.append(sat.get_position())
            sat_ids.append(sat_id)
        return np.array(positions), sat_ids
    
    def _plot_constellation(self, ax, color='#7f8c8d', alpha=0.3, s=6):
        """Plot satellite constellation."""
        positions, _ = self._get_satellite_positions()
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=color, alpha=alpha, s=s, depthshade=True,
                  edgecolors='darkgray', linewidths=0.3, zorder=10)
    
    def _plot_isl_links(self, ax, color='#a4b0be', alpha=0.15, linewidth=0.5):
        """Plot inter-satellite links with higher visibility."""
        for u, v, data in self.topology.graph.edges(data=True):
            pos_u = self.topology.satellites[u].get_position()
            pos_v = self.topology.satellites[v].get_position()
            ax.plot([pos_u[0], pos_v[0]],
                   [pos_u[1], pos_v[1]],
                   [pos_u[2], pos_v[2]],
                   color=color, alpha=alpha, linewidth=linewidth, zorder=5)
    
    def _plot_route_smooth(self, ax, route, color='#e74c3c', linewidth=3.5, alpha=0.98, label=None):
        """Plot smooth routing path with glow effect, hop markers, and hop labels."""
        glow_color = self.colors['route_glow']
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            pos_u = self.topology.satellites[u].get_position()
            pos_v = self.topology.satellites[v].get_position()
            
            # Smooth interpolation
            t = np.linspace(0, 1, 25)
            x_interp = pos_u[0] * (1 - t) + pos_v[0] * t
            y_interp = pos_u[1] * (1 - t) + pos_v[1] * t
            z_interp = pos_u[2] * (1 - t) + pos_v[2] * t
            
            # Outer glow layer (wider, semi-transparent) for better visibility
            ax.plot(x_interp, y_interp, z_interp,
                   color=glow_color, linewidth=linewidth + 4, alpha=0.25,
                   zorder=98, solid_capstyle='round')
            
            # Middle glow layer
            ax.plot(x_interp, y_interp, z_interp,
                   color=glow_color, linewidth=linewidth + 2, alpha=0.4,
                   zorder=99, solid_capstyle='round')
            
            # Main route line
            if i == 0:
                ax.plot(x_interp, y_interp, z_interp,
                       color=color, linewidth=linewidth, alpha=alpha,
                       label=label, zorder=100, solid_capstyle='round')
            else:
                ax.plot(x_interp, y_interp, z_interp,
                       color=color, linewidth=linewidth, alpha=alpha,
                       zorder=100, solid_capstyle='round')
            
            # Bright inner core to make the link pop
            ax.plot(x_interp, y_interp, z_interp,
                   color='#ffffff', linewidth=linewidth - 1.5, alpha=0.9,
                   zorder=101, solid_capstyle='round')
        
        # Draw hop markers for ALL intermediate nodes (bigger and more visible)
        for i in range(1, len(route) - 1):
            pos = self.topology.satellites[route[i]].get_position()
            
            # Outer glow ring for hop marker
            ax.scatter(pos[0], pos[1], pos[2],
                      c=glow_color, s=120, marker='o',
                      alpha=0.3, zorder=100, edgecolors='none')
            
            # Inner bright circle
            ax.scatter(pos[0], pos[1], pos[2],
                      c=self.colors['hop_marker'], s=100, marker='o',
                      edgecolors=self.colors['hop_edge'], linewidths=3.0,
                      zorder=102, alpha=1.0)
            
            # Hop number label
            ax.text(pos[0], pos[1], pos[2] + 450,
                   f'H{i}', fontsize=8, fontweight='bold',
                   color=self.colors['text'], ha='center', va='bottom',
                   zorder=102,
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor=self.colors['hop_edge'], alpha=0.85,
                            linewidth=0.8))
    
    def _plot_endpoint_marker(self, ax, sat_id, color, marker, s=220, label=None):
        """Plot source/destination marker with enhanced visibility."""
        pos = self.topology.satellites[sat_id].get_position()
        
        # Outer glow for endpoint
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker,
                  s=s * 2.5, alpha=0.2, zorder=998, edgecolors='none')
        
        # Middle glow
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker,
                  s=s * 1.5, alpha=0.4, zorder=999, edgecolors='none')
        
        # Main marker
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker,
                  s=s, edgecolors='white', linewidths=2.5, zorder=1000,
                  alpha=1.0, label=label)
    
    def _add_endpoint_label(self, ax, sat_id, label_text, color, offset_z=600, fontsize=11):
        """Add a text label above the endpoint marker with a styled box."""
        pos = self.topology.satellites[sat_id].get_position()
        
        ax.text(pos[0], pos[1], pos[2] + offset_z,
               label_text, fontsize=fontsize, fontweight='bold',
               color='white', ha='center', va='bottom',
               zorder=1001,
               bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                        edgecolor='white', alpha=0.92, linewidth=1.5))
    
    def _compute_route_stats(self, route):
        """Compute routing statistics."""
        hops = len(route) - 1
        total_delay_ms = 0.0
        total_dist_km = 0.0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if self.topology.graph.has_edge(u, v):
                edge = self.topology.graph[u][v]
                total_delay_ms += edge.get('delay_s', 0.0) * 1000.0
            
            pos_u = self.topology.satellites[u].get_position()
            pos_v = self.topology.satellites[v].get_position()
            total_dist_km += np.linalg.norm(pos_v - pos_u)
        
        return {
            'hops': hops,
            'delay_ms': total_delay_ms,
            'distance_km': total_dist_km
        }
    
    def _compute_optimal_view(self, route):
        """Compute optimal camera angle to best view the route path."""
        positions = []
        for sat_id in route:
            positions.append(self.topology.satellites[sat_id].get_position())
        positions = np.array(positions)
        
        # Find centroid of route
        centroid = positions.mean(axis=0)
        
        # Compute azimuth and elevation to look at centroid from outside
        azim = np.degrees(np.arctan2(centroid[1], centroid[0])) + 30
        elev = np.degrees(np.arcsin(np.clip(
            centroid[2] / (np.linalg.norm(centroid) + 1e-10), -1, 1))) + 10
        
        # Clamp elevation
        elev = np.clip(elev, 10, 60)
        
        return azim, elev
    
    def _setup_publication_axes(self, ax, title='', view_azim=50, view_elev=25,
                        show_axes=False, zoom=1.1):
        """Setup axes for publication-quality figures."""
        ax.set_axis_on()
        ax.set_xlabel('X (km)', fontsize=10, labelpad=8, color=self.colors['text'])
        ax.set_ylabel('Y (km)', fontsize=10, labelpad=8, color=self.colors['text'])
        ax.set_zlabel('Z (km)', fontsize=10, labelpad=8, color=self.colors['text'])
        ax.tick_params(labelsize=8, colors=self.colors['text'], pad=3, width=0.7)
        
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20,
                        color=self.colors['text'], family='sans-serif')
        
        ax.view_init(elev=view_elev, azim=view_azim)
        
        # Optimized bounds
        max_range = int(7200 / zoom)
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Clean grid
        ax.grid(True, alpha=0.08, linestyle=':', linewidth=0.5, color=self.colors['grid'])
        
        # Transparent panes
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(self.colors['grid'])
            pane.set_alpha(0.03)
        
        ax.set_box_aspect([1, 1, 1])
    
    def _add_legend(self, ax, src_label, dst_label):
        """Add a professional legend to the plot."""
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['source'], edgecolor='white',
                          label=f'● {src_label} (Source)'),
            mpatches.Patch(facecolor=self.colors['destination'], edgecolor='white',
                          label=f'■ {dst_label} (Destination)'),
            mpatches.Patch(facecolor=self.colors['route_path'], edgecolor='white',
                          label='━ Routing Path'),
            mpatches.Patch(facecolor='white', edgecolor=self.colors['hop_edge'],
                          label='○ Relay Hop'),
        ]
        
        legend = ax.legend(handles=legend_elements, loc='upper left',
                          fontsize=9, framealpha=0.92, edgecolor='#bdc3c7',
                          fancybox=True, shadow=True, borderpad=0.8,
                          handlelength=1.2, handleheight=1.0)
        legend.get_frame().set_facecolor('white')
    
    def visualize_route_publication(self, route, src_label='Source', dst_label='Destination',
                            output_filename='route_publication.png', figsize=(12, 10), dpi=600,
                            show_constellation=True, show_isl=False, show_axes=False,
                            auto_view=True):
        """
        Publication-quality route visualization with labeled nodes and high visibility.
        
        Args:
            route: List of satellite IDs
            src_label: Source city
            dst_label: Destination city
            output_filename: Output file
            figsize: Figure size
            dpi: Resolution (600 for publications)
            show_constellation: Show all satellites
            show_isl: Show ISL links
            show_axes: Show axis labels
            auto_view: Automatically compute best camera angle for the route
        """
        stats = self._compute_route_stats(route)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        
        # Draw Earth
        self._draw_realistic_earth(ax, resolution=120, alpha=0.92)
        
        # Constellation
        if show_constellation:
            self._plot_constellation(ax, alpha=0.22, s=5)
        
        # ISL links
        if show_isl:
            self._plot_isl_links(ax, alpha=0.03, linewidth=0.25)
        
        # Routing path (with glow and hop labels)
        self._plot_route_smooth(ax, route, color=self.colors['route_path'],
                               linewidth=4.0, label='Route')
        
        # Endpoints (with glow)
        self._plot_endpoint_marker(ax, route[0], self.colors['source'], 'o', s=300,
                                  label=src_label)
        self._plot_endpoint_marker(ax, route[-1], self.colors['destination'], 's', s=300,
                                  label=dst_label)
        
        # Endpoint text labels
        self._add_endpoint_label(ax, route[0], f'  {src_label}  ', self.colors['source'],
                                offset_z=650, fontsize=11)
        self._add_endpoint_label(ax, route[-1], f'  {dst_label}  ', '#e67e22',
                                offset_z=650, fontsize=11)
        
        # Title with stats
        title = f'{src_label} → {dst_label}\n'
        title += f'Hops: {stats["hops"]} | Delay: {stats["delay_ms"]:.1f} ms | '
        title += f'Distance: {stats["distance_km"]:.0f} km'
        
        # Compute optimal view angle
        if auto_view:
            view_azim, view_elev = self._compute_optimal_view(route)
        else:
            view_azim, view_elev = 50, 25
        
        self._setup_publication_axes(ax, title=title, view_azim=view_azim,
                             view_elev=view_elev,
                             show_axes=show_axes, zoom=1.1)
        
        # Add legend
        self._add_legend(ax, src_label, dst_label)
        
        plt.tight_layout(pad=0.3)
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                   transparent=False, pad_inches=0.05)
        plt.close()
        
        print(f"✓ Publication-quality visualization: {filepath}")
        return filepath
    
    def visualize_multi_route_comparison(self, routes_dict, output_filename='comparison_multi_panel.png',
                                         figsize=(20, 8), dpi=600):
        """
        Multi-panel route comparison (a), (b), (c) for publications.
        
        Args:
            routes_dict: {label: {'route': [...], 'src': 'City1', 'dst': 'City2'}}
            output_filename: Output file
            figsize: Figure size
            dpi: Resolution
        """
        num_routes = len(routes_dict)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('white')
        
        labels = list(routes_dict.keys())
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        
        for idx, (label, info) in enumerate(routes_dict.items()):
            route = info['route']
            src = info.get('src', 'Source')
            dst = info.get('dst', 'Dest')
            stats = self._compute_route_stats(route)
            
            ax = fig.add_subplot(1, num_routes, idx + 1, projection='3d')
            ax.set_facecolor('white')
            
            # Earth
            self._draw_realistic_earth(ax, resolution=80, alpha=0.90)
            
            # Constellation (minimal)
            self._plot_constellation(ax, alpha=0.18, s=3)
            
            # Route (with glow and hop labels)
            self._plot_route_smooth(ax, route, linewidth=3.5)
            
            # Endpoints (with glow)
            self._plot_endpoint_marker(ax, route[0], self.colors['source'], 'o', s=220,
                                      label=src)
            self._plot_endpoint_marker(ax, route[-1], self.colors['destination'], 's', s=220,
                                      label=dst)
            
            # Endpoint labels
            self._add_endpoint_label(ax, route[0], f' {src} ', self.colors['source'],
                                    offset_z=550, fontsize=9)
            self._add_endpoint_label(ax, route[-1], f' {dst} ', '#e67e22',
                                    offset_z=550, fontsize=9)
            
            # Title
            title = f'{panel_labels[idx]} {src} → {dst}\n'
            title += f'{stats["hops"]} hops, {stats["delay_ms"]:.1f} ms'
            
            # Auto view angle
            view_azim, view_elev = self._compute_optimal_view(route)
            
            self._setup_publication_axes(ax, title=title, view_azim=view_azim,
                                 view_elev=view_elev,
                                 show_axes=False, zoom=1.15)
            
            # Small legend per panel
            self._add_legend(ax, src, dst)
        
        plt.tight_layout(pad=0.5)
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                   transparent=False, pad_inches=0.1)
        plt.close()
        
        print(f"✓ Multi-panel comparison: {filepath}")
        return filepath
