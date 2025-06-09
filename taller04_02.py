import numpy as np
import matplotlib.pyplot as plt

# --- Datos originales ---
xs = np.array([
    0.0003, 0.0822, 0.2770, 0.4212, 0.4403, 0.5588, 0.5943, 0.6134, 0.9070,
    1.0367, 1.1903, 1.2511, 1.2519, 1.2576, 1.6165, 1.6761, 2.0114, 2.0557,
    2.1610, 2.6344
])

ys = np.array([
    1.1017, 1.5021, 0.3844, 1.3251, 1.7206, 1.9453, 0.3894, 0.3328, 1.2887,
    3.1239, 2.1778, 3.1078, 4.1856, 3.3640, 6.0330, 5.8088, 10.5890, 11.5865,
    11.8221, 26.5077
])

# --- Selecciona un punto para ser movido (el del medio) ---
idx_movable = len(xs) // 2
movable_point = [xs[idx_movable], ys[idx_movable]]

# --- Figura interactiva ---
fig, ax = plt.subplots()
ax.set_xlim(min(xs)-0.5, max(xs)+0.5)
ax.set_ylim(min(ys)-1, max(ys)+5)
ax.set_title("Ajuste Cuadrático (Mínimos Cuadrados) con un Punto Movible")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Plots
scatter_plot = ax.scatter(xs, ys, color='red', label='Datos')
movable_plot, = ax.plot(*movable_point, 'go', markersize=10, label='Punto movible')
fit_line, = ax.plot([], [], 'b-', linewidth=2, label='Ajuste cuadrático')

dragging = False

def ajustar_parabola():
    xs_copy = xs.copy()
    ys_copy = ys.copy()
    ys_copy[idx_movable] = movable_point[1]  # usar punto movido

    A = np.vstack([xs_copy**2, xs_copy, np.ones_like(xs_copy)]).T
    coef, _, _, _ = np.linalg.lstsq(A, ys_copy, rcond=None)

    x_fit = np.linspace(min(xs)-0.5, max(xs)+0.5, 300)
    y_fit = coef[0]*x_fit**2 + coef[1]*x_fit + coef[2]

    fit_line.set_data(x_fit, y_fit)
    scatter_plot.set_offsets(np.column_stack((xs, ys_copy)))
    movable_plot.set_data(movable_point[0], movable_point[1])
    fig.canvas.draw_idle()

def on_press(event):
    global dragging
    if event.inaxes != ax:
        return
    if np.hypot(event.xdata - movable_point[0], event.ydata - movable_point[1]) < 0.3:
        dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    if dragging and event.inaxes == ax:
        movable_point[1] = event.ydata  # solo movemos la coordenada Y
        ajustar_parabola()

# Conectar eventos
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

ajustar_parabola()
plt.legend()
plt.show()

