"""
Plotting module for AutoML Mini Lab.

Implements all visual charts, heatmaps, decision boundaries, and performance graphs
using Plotly for responsive, modern, interactive visualizations.
Strictly receives precomputed data as arguments without hidden ML operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Brand Color Palette
PRIMARY_TEAL = "#1B8A93"
DARK_NAVY = "#1B3A57"
ACCENT_ORANGE = "#E0A040"
LIGHT_BG = "#F8FAFC"
BORDER_GRAY = "#E2E8F0"
TEXT_COLOR = "#0F172A"

# Discrete Class Palette for multi-class targets
CLASS_COLORS = [
    "#1B8A93",  # Teal
    "#E0A040",  # Amber/Orange
    "#1B3A57",  # Navy
    "#43AA8B",  # Sage Green
    "#F94144",  # Coral Red
    "#577590",  # Slate Blue
    "#90BE6D",  # Olive Green
    "#F8961E",  # Warm Orange
    "#277DA1",  # Ocean Blue
    "#9D4EDD",  # Violet
]


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling across all Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 17, "color": DARK_NAVY, "family": "Inter, -apple-system, sans-serif"},
        },
        paper_bgcolor="white",
        plot_bgcolor="#FAFCFE",
        font={"family": "Inter, -apple-system, sans-serif", "color": TEXT_COLOR, "size": 12},
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        hoverlabel={"bgcolor": DARK_NAVY, "font_color": "white", "font_size": 12},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EDF2F7",
        linecolor="#CBD5E1",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EDF2F7",
        linecolor="#CBD5E1",
        zeroline=False,
    )
    return fig


def plot_accuracy_vs_k(df_k: pd.DataFrame, best_k: Optional[int] = None) -> go.Figure:
    """
    Plot 1: Mean Cross-Validation Accuracy vs K with a standard-deviation confidence band
    and a distinct highlight marker for the optimal K.

    Args:
        df_k: DataFrame containing ['k', 'cv_mean', 'cv_std', 'train_acc'].
        best_k: Best K value discovered via optimization or CV maximum.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Upper and Lower SD band
    upper_band = (df_k["cv_mean"] + df_k["cv_std"]).clip(upper=1.0)
    lower_band = (df_k["cv_mean"] - df_k["cv_std"]).clip(lower=0.0)

    # Shaded SD region
    fig.add_trace(
        go.Scatter(
            x=list(df_k["k"]) + list(df_k["k"][::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill="toself",
            fillcolor="rgba(27, 138, 147, 0.14)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=True,
            name="CV Std Dev Band (±1σ)",
        )
    )

    # Main CV curve
    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["cv_mean"],
            mode="lines+markers",
            name="Mean CV Accuracy",
            line={"color": PRIMARY_TEAL, "width": 3},
            marker={"size": 8, "color": PRIMARY_TEAL, "symbol": "circle"},
            hovertemplate="<b>K</b> = %{x}<br><b>Mean CV Acc</b> = %{y:.4f}<extra></extra>",
        )
    )

    # Determine best K if not explicitly supplied
    if best_k is None or best_k not in df_k["k"].values:
        best_row = df_k.loc[df_k["cv_mean"].idxmax()]
        target_k = int(best_row["k"])
        target_acc = float(best_row["cv_mean"])
    else:
        best_row = df_k.loc[df_k["k"] == best_k].iloc[0]
        target_k = int(best_k)
        target_acc = float(best_row["cv_mean"])

    # Highlight Best K marker
    fig.add_trace(
        go.Scatter(
            x=[target_k],
            y=[target_acc],
            mode="markers+text",
            name=f"Optimal K = {target_k}",
            marker={"size": 15, "color": ACCENT_ORANGE, "symbol": "star", "line": {"width": 1.5, "color": DARK_NAVY}},
            text=[f"Best K = {target_k}"],
            textposition="top center",
            textfont={"size": 13, "color": DARK_NAVY, "family": "Inter, sans-serif"},
            hovertemplate="<b>Optimal K</b>: %{x}<br><b>CV Score</b>: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "Number of Neighbors (K)", "dtick": 1, "range": [0.5, df_k["k"].max() + 0.5]},
        yaxis={"title": "Cross-Validation Accuracy", "tickformat": ".2f"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
    )

    return _apply_theme(fig, "Cross-Validation Accuracy vs. Number of Neighbors (K)")


def plot_training_vs_k(df_k: pd.DataFrame) -> go.Figure:
    """
    Plot 2: Training Accuracy vs Cross-Validation Accuracy across K.
    Demonstrates model variance/overfitting at low K and underfitting/smoothing at high K.

    Args:
        df_k: DataFrame containing ['k', 'train_acc', 'cv_mean'].

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Training Accuracy curve
    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["train_acc"],
            mode="lines+markers",
            name="Training Accuracy (Fit)",
            line={"color": DARK_NAVY, "width": 2.5, "dash": "dash"},
            marker={"size": 7, "color": DARK_NAVY, "symbol": "square"},
            hovertemplate="<b>K</b> = %{x}<br><b>Train Acc</b> = %{y:.4f}<extra></extra>",
        )
    )

    # CV Accuracy curve
    fig.add_trace(
        go.Scatter(
            x=df_k["k"],
            y=df_k["cv_mean"],
            mode="lines+markers",
            name="Validation Accuracy (Generalization)",
            line={"color": PRIMARY_TEAL, "width": 3},
            marker={"size": 8, "color": PRIMARY_TEAL, "symbol": "circle"},
            hovertemplate="<b>K</b> = %{x}<br><b>CV Acc</b> = %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "Number of Neighbors (K)", "dtick": 1, "range": [0.5, df_k["k"].max() + 0.5]},
        yaxis={"title": "Accuracy", "tickformat": ".2f", "range": [min(df_k["cv_mean"].min(), df_k["train_acc"].min()) - 0.05, 1.02]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
    )

    return _apply_theme(fig, "Model Complexity Analysis: Training vs. Validation Accuracy")


def plot_decision_boundary(
    model_2d: Any,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_test_2d: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
) -> go.Figure:
    """
    Render 2D KNN decision boundaries with smooth meshgrid contour regions,
    clearly distinguishing training samples from held-out test samples.

    Args:
        model_2d: Fitted 2D pipeline/model on the two selected features.
        X_train_2d: (N_train, 2) numpy array.
        y_train: (N_train,) array of true class integers.
        X_test_2d: (N_test, 2) numpy array.
        y_test: (N_test,) array of true test class integers.
        feature_names: List of the 2 selected feature names [feat_x, feat_y].
        class_names: List of all target class labels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Combine data to determine axis ranges with a 6% boundary margin
    all_x = np.concatenate([X_train_2d[:, 0], X_test_2d[:, 0]])
    all_y = np.concatenate([X_train_2d[:, 1], X_test_2d[:, 1]])

    x_span = all_x.max() - all_x.min()
    y_span = all_y.max() - all_y.min()
    x_pad = x_span * 0.08 if x_span > 0 else 1.0
    y_pad = y_span * 0.08 if y_span > 0 else 1.0

    x_min, x_max = all_x.min() - x_pad, all_x.max() + x_pad
    y_min, y_max = all_y.min() - y_pad, all_y.max() + y_pad

    # Generate 120x120 meshgrid for fluid boundary contours
    grid_res = 120
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict class indices over meshgrid
    try:
        grid_preds = model_2d.predict(grid_points).reshape(xx.shape)
    except Exception:
        # Fallback if DataFrame format is expected by pipeline
        grid_df = pd.DataFrame(grid_points, columns=feature_names)
        grid_preds = model_2d.predict(grid_df).reshape(xx.shape)

    num_classes = len(class_names)
    colors = CLASS_COLORS[:num_classes]

    # Map discrete colorscale for contour heatmap
    custom_colorscale = []
    for i in range(num_classes):
        val_start = i / num_classes
        val_end = (i + 1) / num_classes
        # Use pastel alpha for background regions
        hex_c = colors[i]
        custom_colorscale.append([val_start, hex_c])
        custom_colorscale.append([val_end, hex_c])

    # Add Decision Boundary surface
    fig.add_trace(
        go.Heatmap(
            x=np.linspace(x_min, x_max, grid_res),
            y=np.linspace(y_min, y_max, grid_res),
            z=grid_preds,
            colorscale=custom_colorscale,
            showscale=False,
            opacity=0.32,
            hoverinfo="skip",
        )
    )

    # Scatter points for each class: Training (circles) and Test (diamonds)
    for c_idx, c_name in enumerate(class_names):
        c_color = colors[c_idx % len(colors)]

        # Training points
        mask_tr = (y_train == c_idx)
        if np.any(mask_tr):
            fig.add_trace(
                go.Scatter(
                    x=X_train_2d[mask_tr, 0],
                    y=X_train_2d[mask_tr, 1],
                    mode="markers",
                    name=f"{c_name} (Train)",
                    marker={
                        "size": 8,
                        "color": c_color,
                        "symbol": "circle",
                        "line": {"color": "#FFFFFF", "width": 1.2},
                    },
                    hovertemplate=f"<b>Train</b> | {c_name}<br>{feature_names[0]}: %{{x:.3f}}<br>{feature_names[1]}: %{{y:.3f}}<extra></extra>",
                )
            )

        # Test points
        mask_te = (y_test == c_idx)
        if np.any(mask_te):
            fig.add_trace(
                go.Scatter(
                    x=X_test_2d[mask_te, 0],
                    y=X_test_2d[mask_te, 1],
                    mode="markers",
                    name=f"{c_name} (Test)",
                    marker={
                        "size": 11,
                        "color": c_color,
                        "symbol": "diamond",
                        "line": {"color": DARK_NAVY, "width": 2},
                    },
                    hovertemplate=f"<b>Test</b> | {c_name}<br>{feature_names[0]}: %{{x:.3f}}<br>{feature_names[1]}: %{{y:.3f}}<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis={"title": feature_names[0], "range": [x_min, x_max]},
        yaxis={"title": feature_names[1], "range": [y_min, y_max]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.28, "xanchor": "center", "x": 0.5},
    )

    return _apply_theme(fig, f"2D KNN Decision Boundary: {feature_names[0]} vs. {feature_names[1]}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Annotated confusion matrix heatmap with counts and row percentages.

    Args:
        cm: Confusion matrix numpy array (rows = actual, cols = predicted).
        class_names: Names of classes corresponding to indices.

    Returns:
        Plotly Figure.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid zero division
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_pct = (cm / row_sums_safe) * 100

    # Text annotations inside each cell
    annotations_text = []
    for r in range(cm.shape[0]):
        row_annot = []
        for c in range(cm.shape[1]):
            count = cm[r, c]
            pct = cm_pct[r, c]
            row_annot.append(f"<b>{count}</b><br><span style='font-size:10px;'>({pct:.1f}%)</span>")
        annotations_text.append(row_annot)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred: {name}" for name in class_names],
            y=[f"True: {name}" for name in class_names],
            text=annotations_text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white", "family": "Inter, sans-serif"},
            colorscale=[
                [0.0, "#F1F5F9"],
                [0.3, "#94D2BD"],
                [0.7, PRIMARY_TEAL],
                [1.0, DARK_NAVY],
            ],
            showscale=True,
            colorbar={"title": "Samples", "thickness": 14},
            hovertemplate="<b>Actual</b>: %{y}<br><b>Predicted</b>: %{x}<br><b>Count</b>: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "<b>Predicted Label</b>", "side": "bottom"},
        yaxis={"title": "<b>Actual Label</b>", "autorange": "reversed"},
    )

    return _apply_theme(fig, "Confusion Matrix Heatmap (Held-Out Test Set)")


def plot_before_after(
    metrics_default: Dict[str, float],
    metrics_optimized: Dict[str, float],
) -> go.Figure:
    """
    Grouped bar chart comparing Baseline/Default model against Optimized model
    on the exact same held-out test dataset.

    Args:
        metrics_default: Dictionary with 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1'.
        metrics_optimized: Dictionary with corresponding metrics from the optimized pipeline.

    Returns:
        Plotly Figure.
    """
    categories = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]

    vals_default = [round(metrics_default.get(k, 0.0), 4) for k in keys]
    vals_opt = [round(metrics_optimized.get(k, 0.0), 4) for k in keys]

    fig = go.Figure()

    # Default Baseline Bars
    fig.add_trace(
        go.Bar(
            name="Baseline Model (Default)",
            x=categories,
            y=vals_default,
            text=[f"{v:.3f}" for v in vals_default],
            textposition="outside",
            marker_color=DARK_NAVY,
            marker_line={"color": DARK_NAVY, "width": 1},
        )
    )

    # Optimized Model Bars
    fig.add_trace(
        go.Bar(
            name="Optimized Model (Tuned)",
            x=categories,
            y=vals_opt,
            text=[f"{v:.3f}" for v in vals_opt],
            textposition="outside",
            marker_color=PRIMARY_TEAL,
            marker_line={"color": PRIMARY_TEAL, "width": 1},
        )
    )

    max_val = max(max(vals_default), max(vals_opt))
    fig.update_layout(
        barmode="group",
        yaxis={"title": "Score (0.0 – 1.0)", "range": [0.0, min(1.15, max_val + 0.15)]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
    )

    return _apply_theme(fig, "Performance Comparison: Baseline vs. Tuned Pipeline")


def plot_dataset_comparison(df_comp: pd.DataFrame) -> go.Figure:
    """
    Comparison bar chart illustrating Cross-Validation and Test Accuracy across all 4 datasets.

    Args:
        df_comp: DataFrame containing ['Dataset', 'CV Accuracy', 'Test Accuracy', 'Best K', 'Best Metric', 'Standardization'].

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="5-Fold CV Accuracy",
            x=df_comp["Dataset"],
            y=df_comp["CV Accuracy"],
            text=[f"{v:.3f}" for v in df_comp["CV Accuracy"]],
            textposition="outside",
            marker_color=DARK_NAVY,
        )
    )

    fig.add_trace(
        go.Bar(
            name="Held-Out Test Accuracy",
            x=df_comp["Dataset"],
            y=df_comp["Test Accuracy"],
            text=[f"{v:.3f}" for v in df_comp["Test Accuracy"]],
            textposition="outside",
            marker_color=PRIMARY_TEAL,
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis={"title": "Accuracy", "range": [0.0, 1.15]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
    )

    return _apply_theme(fig, "Cross-Dataset KNN Benchmark")


def plot_prediction_probabilities(
    probabilities: np.ndarray,
    class_names: List[str],
    predicted_idx: int,
) -> go.Figure:
    """
    Horizontal probability bar chart highlighting the predicted winning class.

    Args:
        probabilities: 1D array of class probabilities summing to 1.0.
        class_names: List of all target class labels.
        predicted_idx: Winning index predicted by model.

    Returns:
        Plotly Figure.
    """
    # Color winning class with Accent Orange, others with muted navy
    bar_colors = [ACCENT_ORANGE if i == predicted_idx else "#94A3B8" for i in range(len(class_names))]

    fig = go.Figure(
        go.Bar(
            x=probabilities,
            y=class_names,
            orientation="h",
            text=[f"{p * 100:.1f}%" for p in probabilities],
            textposition="outside",
            marker={"color": bar_colors, "line": {"color": DARK_NAVY, "width": 1}},
            hovertemplate="<b>Class</b>: %{y}<br><b>Probability</b>: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "Probability", "range": [0.0, 1.15], "tickformat": ".0%"},
        yaxis={"title": "Class", "autorange": "reversed"},
    )

    return _apply_theme(fig, "Predicted Class Probability Distribution")


def plot_digit_preview(image_matrix: np.ndarray, title: str = "Digit Visualizer") -> go.Figure:
    """
    Render 8x8 handwritten digit grayscale pixel matrix.

    Args:
        image_matrix: 8x8 numpy array of pixel values.
        title: Title of figure.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        go.Heatmap(
            z=image_matrix,
            colorscale="Blues",
            showscale=False,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Intensity: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        width=260,
        height=260,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False, "autorange": "reversed"},
    )

    return _apply_theme(fig, title)
