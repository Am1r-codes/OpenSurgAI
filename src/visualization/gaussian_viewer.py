"""Interactive 3D Gaussian Splatting Viewer for Surgical Scenes.

Provides real-time interactive visualization of EndoGaussian 3D reconstructions
with point picking, annotations, and Nemotron integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData


class GaussianViewer:
    """Interactive viewer for 3D Gaussian splat point clouds."""

    def __init__(self, ply_path: Path):
        """Load Gaussian splat point cloud from .ply file.

        Args:
            ply_path: Path to EndoGaussian output .ply file
        """
        self.ply_path = ply_path
        self.points = None
        self.colors = None
        self.normals = None
        self.loaded = False

        self._load_point_cloud()

    def _load_point_cloud(self):
        """Load point cloud data from .ply file."""
        if not self.ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {self.ply_path}")

        # Load PLY file
        plydata = PlyData.read(str(self.ply_path))
        vertex = plydata['vertex']

        # Extract positions
        self.points = np.stack([
            vertex['x'],
            vertex['y'],
            vertex['z']
        ], axis=1)

        # Extract colors (if available)
        if 'red' in vertex.dtype.names:
            self.colors = np.stack([
                vertex['red'],
                vertex['green'],
                vertex['blue']
            ], axis=1)
        else:
            # Default to white if no colors
            self.colors = np.ones((len(self.points), 3)) * 255

        # Extract normals (if available)
        if 'nx' in vertex.dtype.names:
            self.normals = np.stack([
                vertex['nx'],
                vertex['ny'],
                vertex['nz']
            ], axis=1)

        self.loaded = True
        print(f"✅ Loaded {len(self.points):,} points from {self.ply_path.name}")

    def create_plotly_figure(
        self,
        downsample: int = 10,
        point_size: int = 2,
        camera_position: Optional[dict] = None
    ) -> go.Figure:
        """Create interactive Plotly 3D scatter plot.

        Args:
            downsample: Sample every Nth point (for performance)
            point_size: Marker size
            camera_position: Optional camera position dict

        Returns:
            Plotly Figure object
        """
        if not self.loaded:
            raise RuntimeError("Point cloud not loaded")

        # Downsample for performance
        indices = np.arange(0, len(self.points), downsample)
        points_ds = self.points[indices]
        colors_ds = self.colors[indices]

        # Convert colors to RGB strings
        color_strs = [
            f'rgb({int(r)},{int(g)},{int(b)})'
            for r, g, b in colors_ds
        ]

        # Create 3D scatter plot
        scatter = go.Scatter3d(
            x=points_ds[:, 0],
            y=points_ds[:, 1],
            z=points_ds[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_strs,
                opacity=0.8,
                line=dict(width=0)
            ),
            hovertemplate=(
                '<b>3D Position</b><br>'
                'X: %{x:.3f}<br>'
                'Y: %{y:.3f}<br>'
                'Z: %{z:.3f}<br>'
                '<extra></extra>'
            ),
            name='Gaussian Splat'
        )

        # Create figure
        fig = go.Figure(data=[scatter])

        # Set layout with dark theme matching dashboard
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X',
                    backgroundcolor='#0A1120',
                    gridcolor='rgba(0, 206, 209, 0.2)',
                    showbackground=True,
                ),
                yaxis=dict(
                    title='Y',
                    backgroundcolor='#0A1120',
                    gridcolor='rgba(0, 206, 209, 0.2)',
                    showbackground=True,
                ),
                zaxis=dict(
                    title='Z',
                    backgroundcolor='#0A1120',
                    gridcolor='rgba(0, 206, 209, 0.2)',
                    showbackground=True,
                ),
                bgcolor='#0A1120',
                camera=camera_position if camera_position else dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=1, z=0)
                ),
                aspectmode='cube'
            ),
            paper_bgcolor='#0A1120',
            plot_bgcolor='#0A1120',
            font=dict(color='#E0E0E0'),
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text='🎯 3D Surgical Scene Reconstruction',
                font=dict(size=20, color='#00CED1'),
                x=0.5,
                xanchor='center'
            ),
            height=750,
        )

        return fig

    def get_nearest_point(
        self,
        click_x: float,
        click_y: float,
        click_z: float,
        threshold: float = 0.1
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Find nearest point to clicked location.

        Args:
            click_x: Clicked X coordinate
            click_y: Clicked Y coordinate
            click_z: Clicked Z coordinate
            threshold: Maximum distance to consider

        Returns:
            (point_index, point_coordinates) or (None, None)
        """
        if not self.loaded:
            return None, None

        click_point = np.array([click_x, click_y, click_z])

        # Calculate distances to all points
        distances = np.linalg.norm(self.points - click_point, axis=1)

        # Find nearest point
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist <= threshold:
            return int(min_idx), self.points[min_idx]
        else:
            return None, None

    def get_point_info(self, point_idx: int) -> dict:
        """Get information about a specific point.

        Args:
            point_idx: Index of the point

        Returns:
            Dictionary with point information
        """
        if not self.loaded or point_idx >= len(self.points):
            return {}

        info = {
            'index': point_idx,
            'position': self.points[point_idx].tolist(),
            'color': self.colors[point_idx].tolist() if self.colors is not None else None,
            'normal': self.normals[point_idx].tolist() if self.normals is not None else None
        }

        return info


def load_gaussian_model(model_dir: Path, iteration: int = 7000) -> Optional[GaussianViewer]:
    """Load EndoGaussian model from output directory.

    Args:
        model_dir: Path to EndoGaussian model directory (e.g., output/video49)
        iteration: Training iteration to load

    Returns:
        GaussianViewer instance or None if not found
    """
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    if not ply_path.exists():
        print(f"❌ Model not found: {ply_path}")
        print(f"   Looking for alternatives...")

        # Try other iterations
        for alt_iter in [30000, 15000, 10000, 5000, 3000, 1000]:
            alt_path = model_dir / "point_cloud" / f"iteration_{alt_iter}" / "point_cloud.ply"
            if alt_path.exists():
                print(f"✅ Found model at iteration {alt_iter}")
                ply_path = alt_path
                break
        else:
            return None

    try:
        viewer = GaussianViewer(ply_path)
        return viewer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
