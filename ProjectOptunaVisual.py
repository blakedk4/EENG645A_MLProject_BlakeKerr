import optuna
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import os

# Load your study
study = optuna.load_study(
    study_name="cnn_detector",
    storage="sqlite:///optuna_study.db"
)

# Output folder
fig_path = "./optuna_figures"
os.makedirs(fig_path, exist_ok=True)

# --- Optimization History ---
print("Generating optimization history plot...")
ax = plot_optimization_history(study)
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "optimization_history.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Hyperparameter Importances ---
print("Generating hyperparameter importance plot...")
ax = plot_param_importances(study)
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "param_importances.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Parallel Coordinate Plot ---
print("Generating parallel coordinate plot...")
try:
    ax = plot_parallel_coordinate(study)
    fig = ax.get_figure()
    fig.savefig(os.path.join(fig_path, "parallel_coordinate.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Failed to save parallel_coordinate:", e)

# --- Slice Plot ---
print("Generating slice plot...")
try:
    ax = plot_slice(study)
    fig = ax.get_figure()
    fig.savefig(os.path.join(fig_path, "slice.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Failed to save slice plot:", e)

print("All plots saved as PNGs in:", fig_path)