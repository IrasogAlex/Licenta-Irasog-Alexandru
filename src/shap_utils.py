from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import shap


def generate_shap_waterfall(model, X_scaled_row, output_path: Path | str | None = None) -> plt.Figure:
    """Generate a SHAP waterfall plot figure for a single transaction row."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled_row)

    fig = plt.figure(figsize=(8, 4.5))
    FigureCanvasAgg(fig)
    shap.plots.waterfall(shap_values[0], show=False)
    fig.tight_layout()
    fig.canvas.draw()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    # Return the matplotlib Figure object for display (do not close it here).
    return fig
