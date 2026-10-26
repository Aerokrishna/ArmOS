import numpy as np
import matplotlib.pyplot as plt


def plot_scene(vectors=None, frames=None, frame_scale=0.5):
    """
    Plot 3D vectors and coordinate frames together.

    Parameters:
    -----------
    vectors : (N,3) array-like or list of vectors
        Vectors from origin.
    frames : list of (4,4) numpy arrays
        Homogeneous transformation matrices representing frames.
    frame_scale : float
        Length of frame axes arrows.
    """

    # =========================
    # Prepare vectors
    # =========================
    if vectors is not None:
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, 3)
        if vectors.shape[1] != 3:
            raise ValueError("Vectors must have shape (N,3)")
    else:
        vectors = np.zeros((0, 3))

    # =========================
    # Prepare frames
    # =========================
    if frames is None:
        frames = []

    # =========================
    # Determine scaling
    # =========================
    max_val = 1

    if len(vectors) > 0:
        max_val = max(max_val, np.max(np.abs(vectors)))

    for T in frames:
        origin = T[:3, 3]
        max_val = max(max_val, np.max(np.abs(origin)))

    max_val *= 1.5

    # =========================
    # Create 3D Figure
    # =========================
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    # =========================
    # Plot Vectors
    # =========================
    if len(vectors) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
        for vec, color in zip(vectors, colors):
            ax3d.quiver(
                0, 0, 0,
                vec[0], vec[1], vec[2],
                color=color,
                arrow_length_ratio=0.1
            )

    # =========================
    # Plot Frames
    # =========================
    for T in frames:
        R = T[:3, :3]
        origin = T[:3, 3]

        # Frame axes
        x_axis = R[:, 0] * frame_scale
        y_axis = R[:, 1] * frame_scale
        z_axis = R[:, 2] * frame_scale

        # Plot frame origin
        ax3d.scatter(origin[0], origin[1], origin[2], color='k', s=40)

        # Plot axes
        ax3d.quiver(origin[0], origin[1], origin[2],
                    x_axis[0], x_axis[1], x_axis[2],
                    color='r', arrow_length_ratio=0.2)

        ax3d.quiver(origin[0], origin[1], origin[2],
                    y_axis[0], y_axis[1], y_axis[2],
                    color='g', arrow_length_ratio=0.2)

        ax3d.quiver(origin[0], origin[1], origin[2],
                    z_axis[0], z_axis[1], z_axis[2],
                    color='b', arrow_length_ratio=0.2)

    # =========================
    # Global RGB axes
    # =========================
    ax3d.plot([-max_val, max_val], [0, 0], [0, 0], color='r')
    ax3d.plot([0, 0], [-max_val, max_val], [0, 0], color='g')
    ax3d.plot([0, 0], [0, 0], [-max_val, max_val], color='b')

    # Global origin
    ax3d.scatter(0, 0, 0, color='red', s=80)

    ax3d.set_xlim([-max_val, max_val])
    ax3d.set_ylim([-max_val, max_val])
    ax3d.set_zlim([-max_val, max_val])

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title("3D Scene: Vectors and Frames")

    # =========================
    # 2D Projection Window (Vectors Only)
    # =========================
    if len(vectors) > 0:
        fig2d, axs = plt.subplots(1, 3, figsize=(15, 5))

        plane_info = [
            ("XY Plane", 0, 1, "X", "Y"),
            ("YZ Plane", 1, 2, "Y", "Z"),
            ("XZ Plane", 0, 2, "X", "Z"),
        ]

        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))

        for ax, (title, i, j, xlabel, ylabel) in zip(axs, plane_info):

            for vec, color in zip(vectors, colors):
                ax.quiver(
                    0, 0,
                    vec[i], vec[j],
                    angles='xy',
                    scale_units='xy',
                    scale=1,
                    color=color
                )

            ax.scatter(0, 0, color='red', s=60)
            ax.axhline(0, color='black')
            ax.axvline(0, color='black')

            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_aspect('equal')

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.grid(True)

        fig2d.suptitle("Vector Projections")

    plt.show()


# # ======================================================
# # Example Usage
# # ======================================================

# # Vectors
# vectors = [
#     [1, 2, 3],
#     [-2, 1, 1]
# ]

# # Frame 1 (Identity)
# T1 = np.eye(4)

# # Frame 2 (Translated + Rotated)
# theta = np.pi / 4
# Rz = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta),  np.cos(theta), 0],
#     [0, 0, 1]
# ])

# T2 = np.eye(4)
# T2[:3, :3] = Rz
# T2[:3, 3] = [2, 1, 1]

# frames = [T1, T2]

# plot_scene(vectors=None, frames=frames, frame_scale=0.7)