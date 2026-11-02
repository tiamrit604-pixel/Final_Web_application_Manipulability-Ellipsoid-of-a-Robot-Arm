import numpy as np
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Manipulability Ellipsoid of a Robot Arm", layout="wide")

# Mathematica demonstration constants
D1 = 2.0
R2 = 1.0
R3 = 1.0
STEP = np.pi / 32.0

def jv_rrr_anthro(t1, t2, t3, r2=R2, r3=R3):
    """Linear-velocity Jacobian translated from JvRRRanthro in the Mathematica notebook."""
    a = r2*np.cos(t2) + r3*np.cos(t2+t3)
    return np.array([
        [-a*np.sin(t1),
         -r2*np.sin(t2)*np.cos(t1) - r3*np.cos(t1)*np.sin(t2+t3),
         -r3*np.cos(t1)*np.sin(t2+t3)],
        [ a*np.cos(t1),
         -r2*np.sin(t1)*np.sin(t2) - r3*np.sin(t1)*np.sin(t2+t3),
         -r3*np.sin(t1)*np.sin(t2+t3)],
        [0.0,
         r2*np.cos(t2) + r3*np.cos(t2+t3),
         r3*np.cos(t2+t3)]
    ], dtype=float)

def robot_points(t1, t2, t3):
    o0 = np.array([0.0, 0.0, 0.0])
    o1 = np.array([0.0, 0.0, D1])
    o2 = np.array([
        R2*np.cos(t1)*np.cos(t2),
        R2*np.sin(t1)*np.cos(t2),
        D1 + R2*np.sin(t2)
    ])
    oc = np.array([
        np.cos(t1)*(R2*np.cos(t2) + R3*np.cos(t2+t3)),
        np.sin(t1)*(R2*np.cos(t2) + R3*np.cos(t2+t3)),
        D1 + R2*np.sin(t2) + R3*np.sin(t2+t3)
    ])
    return o0, o1, o2, oc

def add_cylinder(fig, a, b, name, radius=0.085, color="#2878B5"):
    """Draw a true 3D cylindrical link between the original joint centers."""
    direction = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return
    axis = direction / length
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0, 2 * np.pi, 28)
    axial = np.linspace(0, length, 3)
    tt, zz = np.meshgrid(theta, axial)
    xyz = (np.asarray(a)[:, None, None] + axis[:, None, None] * zz[None, :, :]
           + radius * (e1[:, None, None] * np.cos(tt)[None, :, :]
                       + e2[:, None, None] * np.sin(tt)[None, :, :]))
    fig.add_trace(go.Surface(
        x=xyz[0], y=xyz[1], z=xyz[2],
        surfacecolor=np.zeros_like(zz), colorscale=[[0, color], [1, color]],
        showscale=False, name=name, showlegend=True,
        hovertemplate=name + "<extra></extra>",
        lighting=dict(ambient=0.65, diffuse=0.75, specular=0.25, roughness=0.55)
    ))


def add_joint_sphere(fig, center, name, radius=0.12, color="#E86B5A"):
    """Spherical joint marker, sized in model coordinates rather than pixels."""
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(0, np.pi, 14)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, surfacecolor=np.zeros_like(x),
        colorscale=[[0, color], [1, color]], showscale=False,
        showlegend=False, name=name, hovertemplate=name + "<extra></extra>"
    ))

def add_rotation_arrow(fig, center, axis, name, color, radius=0.28):
    """Circular arrow in the plane normal to the actual revolute-joint axis."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    reference = np.array([1., 0., 0.]) if abs(axis[0]) < 0.85 else np.array([0., 1., 0.])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(-0.3, 1.6 * np.pi, 65)
    c = np.asarray(center, dtype=float)
    curve = c[:, None] + radius * (u[:, None] * np.cos(angles) + v[:, None] * np.sin(angles))
    fig.add_trace(go.Scatter3d(x=curve[0], y=curve[1], z=curve[2], mode="lines",
                               line=dict(color=color, width=7), name=name, showlegend=False,
                               hovertemplate=name + "<extra></extra>"))
    tip = curve[:, -1]
    tangent = -u * np.sin(angles[-1]) + v * np.cos(angles[-1])
    fig.add_trace(go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                          u=[tangent[0]], v=[tangent[1]], w=[tangent[2]],
                          anchor="tip", sizemode="absolute", sizeref=0.17,
                          colorscale=[[0, color], [1, color]], showscale=False,
                          name=name + " direction", showlegend=False, hoverinfo="skip"))


def add_joint_housing(fig, center, axis, name, color, radius=0.17, length=0.30):
    """Revolute joint represented by a cylinder aligned to its rotation axis."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    c = np.asarray(center, dtype=float)
    add_cylinder(fig, c - 0.5 * length * axis, c + 0.5 * length * axis,
                 name, radius=radius, color=color)
    # Visible end faces give the cylinder its mechanical-joint appearance.
    for sign in (-1, 1):
        face = c + sign * 0.5 * length * axis
        ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.85 else np.array([0., 1., 0.])
        u = np.cross(axis, ref); u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angles = np.linspace(0, 2*np.pi, 32)
        ring = face[:, None] + radius * (u[:, None]*np.cos(angles) + v[:, None]*np.sin(angles))
        vertices = np.column_stack([face, ring])
        fig.add_trace(go.Mesh3d(x=vertices[0], y=vertices[1], z=vertices[2],
                                i=np.zeros(31, dtype=int), j=np.arange(1,32),
                                k=np.arange(2,33), color=color, opacity=1,
                                flatshading=True, showlegend=False, hoverinfo="skip"))

def add_ellipsoid(fig, center, U, s):
    # Unit sphere -> diag(s) -> U -> translate to end effector.
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 28)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=0).reshape(3, -1)
    pts = (U @ (s[:, None] * sphere)) + center[:, None]
    X = pts[0].reshape(x.shape)
    Y = pts[1].reshape(y.shape)
    Z = pts[2].reshape(z.shape)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=0.45, showscale=False,
        name="Manipulability ellipsoid", hoverinfo="skip"
    ))

def add_principal_axes(fig, center, U, s):
    # The Mathematica notebook suppresses individual principal graphics
    # when the corresponding singular value is <= 0.1.
    for i in range(3):
        if s[i] > 0.1:
            d = U[:, i] * s[i]
            fig.add_trace(go.Scatter3d(
                x=[center[0]-d[0], center[0]+d[0]],
                y=[center[1]-d[1], center[1]+d[1]],
                z=[center[2]-d[2], center[2]+d[2]],
                mode="lines", line=dict(width=5),
                name=f"Principal axis {i+1}"
            ))

st.title("Manipulability Ellipsoid of a Robot Arm")
st.caption("Python/Streamlit translation of the supplied Mathematica Demonstration")

# Joint controls: draggable circular dials and manual inputs.
# The dial is a local Streamlit custom component (no external web dependencies).
DIAL_DIR = Path(__file__).resolve().parent / "joint_angle_dial"
angle_dial = components.declare_component("joint_angle_dial", path=str(DIAL_DIR))
robot_view = components.declare_component("robot_view", path=str(Path(__file__).resolve().parent / "robot_plot"))
st.session_state.setdefault("robot_saved_camera", None)

JOINT_LIMITS = [(-180.0, 180.0, 0.0),
                (-180.0, 180.0, 0.0),
                (-180.0, 180.0, 45.0)]

for i, (_, _, default) in enumerate(JOINT_LIMITS, 1):
    st.session_state.setdefault(f"joint_{i}_deg", default)


def update_joint(i, source, is_degrees):
    value = st.session_state[f"{source}_{i}_{'deg' if is_degrees else 'rad'}"]
    st.session_state[f"joint_{i}_deg"] = (
        float(value) if is_degrees else float(np.degrees(value))
    )


st.markdown("""
<style>

/* Increase left sidebar width */
section[data-testid="stSidebar"] {
    width: 410px !important;
    min-width: 410px !important;
}

/* Hide +/- buttons */
[data-testid="stNumberInput"] button {
    display: none !important;
}

/* Reduce reserved height of circular dial iframe */
section[data-testid="stSidebar"] iframe {
    height: 235px !important;
    min-height: 235px !important;
}

/* Reduce spacing between sidebar elements */
section[data-testid="stSidebar"]
div[data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Joint controls")
    unit = st.radio("Angle unit", ["Degrees (°)", "Radians (rad)"], horizontal=True)
    use_degrees = unit == "Degrees (°)"
    suffix = "deg" if use_degrees else "rad"
    angles_deg = []

    for i, (lower, upper, _) in enumerate(JOINT_LIMITS, 1):
        current_deg = float(np.clip(st.session_state[f"joint_{i}_deg"], -180, 180))
        st.markdown(f"**Joint θ{i}**")

        # Compact row: interactive dial on left, direct input on right.
        dial_col, input_col = st.columns([3, 0.7], gap="small", vertical_alignment="center")
        with dial_col:
            dial_result = angle_dial(
                angle=current_deg, unit=suffix, joint=i,
                key=f"angle_dial_{i}", default=None
            )
        if isinstance(dial_result, dict) and "seq" in dial_result and "angle" in dial_result:
            event_key = f"dial_last_event_{i}"
            if st.session_state.get(event_key) != dial_result["seq"]:
                st.session_state[event_key] = dial_result["seq"]
                st.session_state[f"joint_{i}_deg"] = float(
                    np.clip(dial_result["angle"], -180, 180)
                )

        current_deg = float(st.session_state[f"joint_{i}_deg"])
        number_key = f"number_{i}_{suffix}"
        with input_col:
            if use_degrees:
                st.session_state[number_key] = int(np.clip(round(current_deg), -180, 180))
                st.number_input(
                    f"θ{i} (°)", min_value=-180, max_value=180,
                    step=1, format="%d", key=number_key,
                    on_change=update_joint, args=(i, "number", True),
                )
            else:
                st.session_state[number_key] = float(np.radians(current_deg))
                st.number_input(
                    f"θ{i} (rad)", min_value=float(-np.pi), max_value=float(np.pi),
                    step=0.001, format="%.3f", key=number_key,
                    on_change=update_joint, args=(i, "number", False),
                )

        angles_deg.append(float(st.session_state[f"joint_{i}_deg"]))

    t1, t2, t3 = np.radians(angles_deg)
    st.caption("Joint angles")
    for i, angle in enumerate(angles_deg, 1):
        st.write(f"θ{i} = {angle:.0f}° = {np.radians(angle):.3f} rad")


J = jv_rrr_anthro(t1, t2, t3)
U, s, Vt = np.linalg.svd(J)
Sigma = np.diag(s)
mu = float(np.linalg.det(Sigma))
rank = int(np.linalg.matrix_rank(Sigma))
o0, o1, o2, oc = robot_points(t1, t2, t3)

# On narrow desktop windows, stack the visualization and results instead of squeezing them.
st.markdown("""
<style>
@media (max-width: 1100px) {
  div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]) {
    flex-wrap: wrap !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: min(100%, 360px) !important;
    flex: 1 1 100% !important;
    width: 100% !important;
  }
}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1.7, 1], gap="medium")

with left:
    st.subheader(f"Manipulability μ = {mu:.6g}")
    fig = go.Figure()

    # Solid geometry is for visualization only; joint centers and all calculations are unchanged.
    add_cylinder(fig, o0, o1, "Base / link 1", radius=0.10, color="#2463A6")
    add_cylinder(fig, o1, o2, "Link 2", radius=0.085, color="#4CA5D9")
    add_cylinder(fig, o2, oc, "Link 3", radius=0.075, color="#E66B4E")
    # Joint axes derived from the same kinematic convention as the Jacobian.
    base_axis = np.array([0., 0., 1.])
    pitch_axis = np.array([-np.sin(t1), np.cos(t1), 0.])
    add_joint_housing(fig, o0, base_axis, "Joint 1 (yaw)", "#D94B3D", radius=0.19, length=0.38)
    add_joint_housing(fig, o1, pitch_axis, "Joint 2 (shoulder pitch)", "#2455D6")
    add_joint_housing(fig, o2, pitch_axis, "Joint 3 (elbow pitch)", "#47A832")
    add_joint_sphere(fig, oc, "End effector", radius=0.095, color="#F2BD46")
    add_rotation_arrow(fig, o0, base_axis, "θ₁ rotation", "#D94B3D", radius=0.32)
    add_rotation_arrow(fig, o1, pitch_axis, "θ₂ rotation", "#2455D6", radius=0.29)
    add_rotation_arrow(fig, o2, pitch_axis, "θ₃ rotation", "#47A832", radius=0.29)

    add_principal_axes(fig, oc, U, s)

    # Original Mathematica draws the filled 3-D ellipsoid only if MatrixRank[Σ] > 2.
    if rank > 2:
        add_ellipsoid(fig, oc, U, s)

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        # Keep Plotly's user-controlled camera/zoom when Streamlit reruns
        # after a joint-angle change.
        uirevision="robot-camera-v1",
        scene_uirevision="robot-scene-camera-v1",
        scene=dict(
            uirevision="robot-scene-camera-v1",
            dragmode="orbit",
            xaxis=dict(range=[-3.5, 3.5], title="x"),
            yaxis=dict(range=[-4, 4], title="y"),
            zaxis=dict(range=[-1, 5], title="z"),
            aspectmode="manual",
            aspectratio=dict(x=7, y=8, z=6),
            # A slightly farther initial view so the complete robot and
            # manipulability graphics are easier to see on first load.
            camera=dict(
                eye=dict(x=0, y=-6, z=2.5),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0)
            )
        ),
        legend=dict(orientation="h")
    )
    # Custom Plotly component captures camera changes and retains them across
    # Streamlit reruns (including joint-angle dial updates).
    if st.session_state["robot_saved_camera"] is not None:
        fig.update_layout(scene_camera=st.session_state["robot_saved_camera"])

    camera_event = robot_view(
        figure=fig.to_json(),
        camera=st.session_state["robot_saved_camera"],
        key="persistent_robot_view",
        default=None,
    )
    if isinstance(camera_event, dict) and isinstance(camera_event.get("camera"), dict):
        st.session_state["robot_saved_camera"] = camera_event["camera"]

with right:
    st.subheader("Numerical values")
    st.write("End-effector position")
    st.dataframe({"Axis": ["x", "y", "z"], "Position": np.round(oc, 6)}, hide_index=True, use_container_width=True)
    st.write("Singular values")
    st.dataframe({"Axis": ["σ₁", "σ₂", "σ₃"], "Value": np.round(s, 6)}, hide_index=True, use_container_width=True)
    st.write(f"rank(Σ) = {rank}")
    with st.expander("Jacobian Jᵥ"):
        st.dataframe(np.round(J, 6), use_container_width=True)
    with st.expander("SVD matrices"):
        st.write("U")
        st.dataframe(np.round(U, 6), use_container_width=True)
        st.write("Σ")
        st.dataframe(np.round(Sigma, 6), use_container_width=True)
        st.write("Vᵀ")
        st.dataframe(np.round(Vt, 6), use_container_width=True)

if rank <= 2:
    st.warning("Singular configuration: the original Mathematica code suppresses the filled 3-D ellipsoid when rank(Σ) ≤ 2.")
