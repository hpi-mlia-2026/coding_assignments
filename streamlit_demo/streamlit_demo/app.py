import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Linear Regression GD Demo", layout="wide")

# ----------------------------
# Data and model
# ----------------------------
np.random.seed(7)

x = np.linspace(-2.5, 2.5, 25)
true_w = 1.7
true_b = -0.8
noise = 0.35
y = true_w * x + true_b + noise * np.random.randn(len(x))

val_x = np.array([0.22, 0.37, 0.41, 0.57, 0.63])
val_y = true_w * val_x + true_b + 2 * noise * np.random.randn(len(val_x))


def mse_loss(w, b):
    pred = w * x + b
    return float(np.mean((pred - y) ** 2))


def grad_mse(w, b):
    pred = w * x + b
    err = pred - y
    dw = 2.0 * np.mean(err * x)
    db = 2.0 * np.mean(err)
    return float(dw), float(db)


def val_mse_loss(w, b):
    pred = w * val_x + b
    return float(np.mean((pred - val_y) ** 2))


def init_state():
    defaults = {
        "w": -0.5,
        "b": 1.5,
        "path_w": [],
        "path_b": [],
        "loss_history": [],
        "val_loss_history": [],
        "running": False,
        "last_tick": -1,
        "initialized": False,
        "show_validation": False,
        "learning_rate": 0.05,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def build_surface():
    w_vals = np.linspace(-1.0, 4.0, 160)
    b_vals = np.linspace(-3.0, 2.0, 160)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = ((W[..., None] * x + B[..., None] - y) ** 2).mean(axis=-1)
    return W, B, Z


W, B, Z = build_surface()


def record_current_state():
    w = float(st.session_state.w)
    b = float(st.session_state.b)
    loss = mse_loss(w, b)
    val_loss = val_mse_loss(w, b)

    st.session_state.path_w.append(w)
    st.session_state.path_b.append(b)
    st.session_state.loss_history.append(loss)
    st.session_state.val_loss_history.append(val_loss)


def step_once():
    w = float(st.session_state.w)
    b = float(st.session_state.b)
    lr = float(st.session_state.learning_rate)
    dw, db = grad_mse(w, b)

    st.session_state.w = w - lr * dw
    st.session_state.b = b - lr * db
    record_current_state()


def reset_state():
    st.session_state.w = float(np.random.uniform(-1.0, 4.0))
    st.session_state.b = float(
        np.sqrt(2.5 * 2.5 + 0.01 - (st.session_state.w - 1.5) ** 2)
        * np.random.choice([-1, 1])
    )
    st.session_state.path_w = []
    st.session_state.path_b = []
    st.session_state.loss_history = []
    st.session_state.val_loss_history = []
    st.session_state.running = False
    st.session_state.last_tick = -1
    record_current_state()


init_state()

# First load should show the initial state immediately.
if not st.session_state.initialized:
    record_current_state()
    st.session_state.initialized = True

# Auto-run: one gradient step per refresh tick.
if st.session_state.running:
    tick = st_autorefresh(interval=180, limit=None, key="gd_tick")
    if tick != st.session_state.last_tick:
        st.session_state.last_tick = tick
        step_once()

st.title("Gradient Descent on a Linear Model")
st.caption("Streamlit version of the interactive matplotlib demo.")

# ----------------------------
# Figure layout
# ----------------------------
fig = plt.figure(figsize=(15, 6))

ax_surface = fig.add_axes([0.04, 0.18, 0.28, 0.75])
ax_loss = fig.add_axes([0.37, 0.18, 0.25, 0.75])
ax_fit = fig.add_axes([0.67, 0.18, 0.30, 0.75])

# Left panel: loss surface
cont = ax_surface.contourf(W, B, Z, levels=35, cmap="viridis")
fig.colorbar(cont, ax=ax_surface, fraction=0.046, pad=0.04).set_label("MSE loss")
ax_surface.set_xlabel("w")
ax_surface.set_ylabel("b")
ax_surface.set_title("Loss surface")

path_w = st.session_state.path_w
path_b = st.session_state.path_b
traj_line, = ax_surface.plot(path_w, path_b, "w-", lw=2, alpha=0.95)
current_pt, = ax_surface.plot([st.session_state.w], [st.session_state.b], "ro", ms=8)

# Gradient arrow
w_now = float(st.session_state.w)
b_now = float(st.session_state.b)
dw_now, db_now = grad_mse(w_now, b_now)
scale = 0.15
ax_surface.quiver(
    [w_now],
    [b_now],
    [-dw_now * scale],
    [-db_now * scale],
    color="white",
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.006,
)

# Middle panel: loss per step
steps = np.arange(len(st.session_state.loss_history))
loss_line, = ax_loss.plot(steps, st.session_state.loss_history, color="blue", lw=2)
val_loss_line, = ax_loss.plot(steps, st.session_state.val_loss_history, color="orange", lw=2)
show_val = bool(st.session_state.show_validation)
val_loss_line.set_visible(show_val)
ax_loss.set_title("Training & Validation loss")
ax_loss.set_xlabel("step")
ax_loss.set_ylabel("MSE")
ax_loss.grid(True, alpha=0.25)

# Right panel: fit to data
ax_fit.scatter(x, y, label="training data", s=35)
val_scatter = ax_fit.scatter(val_x, val_y, label="validation data", s=35)
val_scatter.set_visible(show_val)
fit_line, = ax_fit.plot([], [], "r-", lw=2, label="model")
xx = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
yy = st.session_state.w * xx + st.session_state.b
fit_line.set_data(xx, yy)
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("y")
ax_fit.set_title("Current fit")
ax_fit.legend(loc="upper left")
ax_fit.grid(True, alpha=0.25)

# Info box
ax_info = fig.add_axes([0.82, 0.04, 0.16, 0.08])
ax_info.axis("off")
loss_now = mse_loss(st.session_state.w, st.session_state.b)
ax_info.text(
    0.0,
    0.5,
    f"w={st.session_state.w:.3f}\nb={st.session_state.b:.3f}\nloss={loss_now:.4f}",
    fontsize=10,
    va="center",
)

# ----------------------------
# Controls
# ----------------------------
control_cols = st.columns([3, 1, 1, 1, 2])
with control_cols[0]:
    st.session_state.learning_rate = st.slider(
        "learning rate",
        0.001,
        0.2,
        float(st.session_state.learning_rate),
        0.001,
    )
with control_cols[1]:
    if st.button("Step", use_container_width=True):
        step_once()
with control_cols[2]:
    if st.button("Run/Pause", use_container_width=True):
        st.session_state.running = not st.session_state.running
        if st.session_state.running:
            st.session_state.last_tick = -1
with control_cols[3]:
    if st.button("Reset", use_container_width=True):
        reset_state()
with control_cols[4]:
    st.session_state.show_validation = st.checkbox(
        "Show Validation Loss", value=bool(st.session_state.show_validation)
    )

st.pyplot(fig, use_container_width=True)

st.info(
    "The app matches the original workflow: Step for one update, Run/Pause for continuous updates, Reset for a new start, and a checkbox to show validation loss.",
    icon="ℹ️",
)
