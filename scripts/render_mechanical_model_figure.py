from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_segment(ax, p0, p1, color="#1f4e79", lw=4):
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
    )


def draw_arrow(ax, start, vec, color, label, text_offset=(0, 0, 0)):
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vec[0],
        vec[1],
        vec[2],
        color=color,
        linewidth=2.2,
        arrow_length_ratio=0.15,
    )
    end = (start[0] + vec[0], start[1] + vec[1], start[2] + vec[2])
    ax.text(
        end[0] + text_offset[0],
        end[1] + text_offset[1],
        end[2] + text_offset[2],
        label,
        fontsize=10,
        color=color,
        weight="bold",
    )


def main():
    out_path = Path(r"G:\codxx\paper1-gogo\overleaf_nature_20260606\fig8_mechanical_model.png")

    fig = plt.figure(figsize=(8.4, 5.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    joints = [
        (0.00, 0.00, 0.00),
        (0.10, 0.06, 0.18),
        (0.22, 0.08, 0.38),
        (0.33, 0.02, 0.54),
        (0.44, -0.03, 0.63),
        (0.54, 0.00, 0.70),
        (0.63, 0.05, 0.73),
        (0.73, 0.09, 0.69),
    ]

    for a, b in zip(joints[:-1], joints[1:]):
        draw_segment(ax, a, b)

    xs, ys, zs = zip(*joints)
    ax.scatter(xs, ys, zs, s=28, color="#0b2e4f", depthshade=False)

    ee = joints[-1]
    gripper_left = (ee[0] + 0.05, ee[1] + 0.03, ee[2] - 0.03)
    gripper_right = (ee[0] + 0.05, ee[1] + 0.13, ee[2] - 0.03)
    draw_segment(ax, ee, gripper_left, color="#4d4d4d", lw=3)
    draw_segment(ax, ee, gripper_right, color="#4d4d4d", lw=3)
    draw_segment(ax, gripper_left, (gripper_left[0], gripper_left[1], gripper_left[2] - 0.07), color="#4d4d4d", lw=3)
    draw_segment(ax, gripper_right, (gripper_right[0], gripper_right[1], gripper_right[2] - 0.07), color="#4d4d4d", lw=3)

    obj_center = (0.82, 0.10, 0.55)
    ax.bar3d(
        obj_center[0] - 0.04,
        obj_center[1] - 0.04,
        obj_center[2] - 0.05,
        0.08,
        0.08,
        0.10,
        color="#d9a441",
        alpha=0.85,
        shade=True,
    )

    draw_arrow(ax, obj_center, (0, 0, -0.18), "#b22222", r"$mg$", (0.01, 0.0, -0.01))
    draw_arrow(ax, (0.78, 0.06, 0.55), (-0.08, -0.02, 0.05), "#2e8b57", r"$F_c$", (-0.03, -0.02, 0.01))
    draw_arrow(ax, ee, (-0.10, 0.02, 0.08), "#8b008b", r"$\tau_{\mathrm{res}}$", (-0.05, 0.01, 0.01))
    draw_arrow(ax, joints[3], (0.00, -0.12, 0.00), "#ff8c00", r"$\tau_3$", (0.01, -0.03, 0.00))

    ax.text(joints[1][0] - 0.03, joints[1][1] - 0.02, joints[1][2] + 0.02, r"$L_1$", fontsize=9)
    ax.text(joints[3][0] - 0.01, joints[3][1] - 0.02, joints[3][2] + 0.03, r"$L_3$", fontsize=9)
    ax.text(joints[5][0] + 0.00, joints[5][1] + 0.02, joints[5][2] + 0.02, r"$L_6$", fontsize=9)
    ax.text(ee[0] + 0.01, ee[1] - 0.02, ee[2] + 0.04, r"$T_{\mathrm{ee}}$", fontsize=10, weight="bold")
    ax.text(obj_center[0] - 0.02, obj_center[1] + 0.05, obj_center[2] + 0.08, "target object", fontsize=10)

    ax.plot([0.0, 0.95], [0.0, 0.0], [0.0, 0.0], color="black", linewidth=1.0, alpha=0.4)
    ax.plot([0.0, 0.0], [0.0, 0.30], [0.0, 0.0], color="black", linewidth=1.0, alpha=0.4)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 0.85], color="black", linewidth=1.0, alpha=0.4)
    ax.text(0.98, 0.0, 0.0, "x", fontsize=9)
    ax.text(0.0, 0.32, 0.0, "y", fontsize=9)
    ax.text(0.0, 0.0, 0.88, "z", fontsize=9)

    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(0.0, 0.95)
    ax.set_ylim(-0.10, 0.30)
    ax.set_zlim(0.0, 0.90)
    ax.set_box_aspect((1.2, 0.6, 0.9))
    ax.set_axis_off()
    ax.set_title(
        "Rigid-body and contact model used to motivate PPE feasibility screening\n"
        "and bounded residual execution control",
        fontsize=12,
        pad=18,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
