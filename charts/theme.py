import plotly.graph_objects as go

# 12-Color Palette
PALETTE = [
    "#2962FF", "#FF6D00", "#00897B", "#D50000", "#AA00FF",
    "#FFD600", "#00BFA5", "#FF3D00", "#2E7D32", "#C51162",
    "#6200EA", "#0091EA"
]

BASE_VOL_COLOR = "#616161"
CANDLE_INCREASING = "#26A69A"
CANDLE_DECREASING = "#EF5350"
BACKGROUND_COLOR = "#FFFFFF"
PLOT_BACKGROUND_COLOR = "#F8F9FA"
GRID_COLOR = "#E9ECEF"
TEXT_COLOR = "#212529"

def apply_light_professional_theme(fig):
    """Applies the light professional theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=PLOT_BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5
        )
    )

    # Update all x and y axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=GRID_COLOR,
        tickcolor=GRID_COLOR
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=TEXT_COLOR,
        showline=True,
        linecolor=GRID_COLOR,
        tickcolor=GRID_COLOR
    )
    return fig
