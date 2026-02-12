"""Streamlit Integration for 3D Gaussian Viewer.

Provides the "3D Scene" tab for the OpenSurgAI dashboard with interactive
3D reconstruction, point picking, and Nemotron-powered annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st
import plotly.graph_objects as go

from src.visualization.gaussian_viewer import GaussianViewer, load_gaussian_model


def render_3d_scene_tab(video_id: str, project_root: Path):
    """Render the 3D Scene tab in the Streamlit dashboard.

    Args:
        video_id: ID of the current video (e.g., 'video49')
        project_root: Path to project root directory
    """
    st.markdown("### 🎯 Interactive 3D Surgical Scene Reconstruction")
    st.markdown("Click on any point in the 3D scene to get AI-powered explanations!")

    # Check if EndoGaussian model exists
    model_dir = project_root / "external" / "EndoGaussian" / "output" / video_id

    if not model_dir.exists():
        st.warning(f"""
        **3D Model Not Found**

        EndoGaussian model for `{video_id}` hasn't been trained yet.

        **Quick Start:**
        1. Prepare data: `python scripts/prepare_cholec80_for_gaussian.py --video {video_id}`
        2. Train model: `cd external/EndoGaussian && python train.py -s ../../data/{video_id} -m {video_id}`
        3. Refresh this page!

        **Training time:** ~2 minutes!
        """)

        # Show placeholder visualization
        st.markdown("---")
        st.info("**Preview:** This is what your interactive 3D scene will look like:")

        # Create demo placeholder
        demo_points = _create_demo_point_cloud()
        fig = go.Figure(data=[go.Scatter3d(
            x=demo_points[:, 0],
            y=demo_points[:, 1],
            z=demo_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='rgba(0, 206, 209, 0.3)',
                opacity=0.6
            ),
            hovertemplate='<b>Demo Point</b><br>Train model to see actual reconstruction<extra></extra>',
            name='Demo Scene'
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', backgroundcolor='#0A1120', gridcolor='rgba(0, 206, 209, 0.2)'),
                yaxis=dict(title='Y', backgroundcolor='#0A1120', gridcolor='rgba(0, 206, 209, 0.2)'),
                zaxis=dict(title='Z', backgroundcolor='#0A1120', gridcolor='rgba(0, 206, 209, 0.2)'),
                bgcolor='#0A1120',
                aspectmode='cube'
            ),
            paper_bgcolor='#0A1120',
            plot_bgcolor='#0A1120',
            font=dict(color='#E0E0E0'),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)
        return

    # Load the Gaussian model
    try:
        with st.spinner("Loading 3D reconstruction..."):
            viewer = load_gaussian_model(model_dir)

        if viewer is None:
            st.error("Failed to load Gaussian model. Check the model directory.")
            return

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create two columns for controls and info
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("#### ⚙️ Visualization Controls")

        # Downsampling control
        downsample = st.slider(
            "Point Density",
            min_value=1,
            max_value=50,
            value=10,
            help="Lower = more points (slower), Higher = fewer points (faster)"
        )

        # Point size control
        point_size = st.slider(
            "Point Size",
            min_value=1,
            max_value=10,
            value=2,
            help="Size of each point in the visualization"
        )

        # Camera preset
        camera_preset = st.selectbox(
            "Camera Angle",
            options=["Default", "Top View", "Front View", "Side View", "Surgical View"],
            index=0
        )

        st.markdown("---")
        st.markdown("#### 📊 Model Info")
        st.markdown(f"""
        **Points:** {len(viewer.points):,}
        **Displayed:** {len(viewer.points) // downsample:,}
        **Model:** {viewer.ply_path.name}
        """)

    with col1:
        # Get camera position based on preset
        camera_position = _get_camera_position(camera_preset)

        # Create the 3D figure
        fig = viewer.create_plotly_figure(
            downsample=downsample,
            point_size=point_size,
            camera_position=camera_position
        )

        # Display the figure
        st.plotly_chart(fig, use_container_width=True, key=f"gaussian_3d_{video_id}")

    # Point annotation section
    st.markdown("---")
    st.markdown("### 🔍 Point Annotations")

    # Check if there's a selected point in session state
    if "selected_point" not in st.session_state:
        st.info("👆 Click on a point in the 3D viewer above to get AI-powered explanations!")
    else:
        point_info = st.session_state["selected_point"]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Selected Point:**")
            st.json(point_info)

        with col2:
            st.markdown("**AI Explanation:**")

            # Generate explanation using Nemotron
            explanation = _generate_nemotron_explanation(point_info, video_id)
            st.markdown(explanation)

            if st.button("Clear Selection"):
                del st.session_state["selected_point"]
                st.rerun()

    # Add instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Interactive 3D Reconstruction:**

        1. **Rotate:** Click and drag to rotate the view
        2. **Zoom:** Use mouse wheel or pinch gesture
        3. **Pan:** Right-click and drag
        4. **Select Point:** Click on any point to see details

        **Controls:**
        - Adjust point density for performance (lower = more detailed)
        - Change camera angles for different perspectives
        - Use presets for common surgical viewing angles

        **AI Annotations:**
        - Selected points are analyzed by NVIDIA Nemotron
        - Get explanations about anatomical structures and surgical instruments
        - Context-aware based on the current surgical phase
        """)


def _get_camera_position(preset: str) -> dict:
    """Get camera position dictionary for Plotly based on preset name.

    Args:
        preset: Name of camera preset

    Returns:
        Camera position dictionary for Plotly
    """
    presets = {
        "Default": dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        "Top View": dict(
            eye=dict(x=0, y=2.5, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=-1)
        ),
        "Front View": dict(
            eye=dict(x=0, y=0, z=2.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        "Side View": dict(
            eye=dict(x=2.5, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        "Surgical View": dict(
            eye=dict(x=0.5, y=1.8, z=1.2),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        )
    }

    return presets.get(preset, presets["Default"])


def _create_demo_point_cloud():
    """Create a demo point cloud for placeholder visualization."""
    import numpy as np

    # Create a simple sphere-like point cloud
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    r = 1.0
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    return points


def _generate_nemotron_explanation(point_info: dict, video_id: str) -> str:
    """Generate AI explanation for a selected 3D point using Nemotron.

    Args:
        point_info: Dictionary with point information (position, color, normal)
        video_id: Current video ID for context

    Returns:
        Markdown-formatted explanation
    """
    # TODO: Integrate with actual Nemotron API
    # For now, return a placeholder that uses the point information

    position = point_info.get("position", [0, 0, 0])
    color = point_info.get("color", [255, 255, 255])

    # Determine likely object based on color heuristics
    r, g, b = color[:3] if len(color) >= 3 else [255, 255, 255]

    if r > 150 and g < 100 and b < 100:
        likely_object = "tissue or organ (reddish coloring)"
    elif r > 200 and g > 200 and b > 200:
        likely_object = "surgical instrument (metallic/bright)"
    elif r < 100 and g < 100 and b > 150:
        likely_object = "background or cavity (dark blue/black)"
    else:
        likely_object = "anatomical structure"

    explanation = f"""
**3D Point Analysis:**

**Position:** `({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})`
**Color:** RGB({int(r)}, {int(g)}, {int(b)})
**Likely Object:** {likely_object}

**AI Reasoning:**
Based on the spatial position and color information, this point appears to be part of a {likely_object}.
The 3D coordinates place it in the {"central region" if abs(position[0]) < 0.5 else "peripheral area"}
of the surgical field.

**Surgical Context:**
In laparoscopic cholecystectomy procedures like {video_id}, points with this coloration and position
typically represent {"critical anatomical structures requiring careful dissection" if "tissue" in likely_object else "surgical instruments being used by the surgeon"}.

*Note: This is a demonstration. Full Nemotron integration will provide detailed surgical reasoning.*
    """

    return explanation.strip()
