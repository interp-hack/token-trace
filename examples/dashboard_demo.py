import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st


def dashboard_demo():
    # Define grid size and number of blocks
    grid_size = 3
    values = [[(i + j) % 3 for j in range(grid_size)] for i in range(grid_size)]
    colors = ["red", "green", "blue"]  # colors corresponding to values 0, 1, 2
    row_names = ["Row A", "Row B", "Row C"]
    col_names = ["Col 1", "Col 2", "Col 3"]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")  # Ensure squares are displayed as squares

    # Plot settings
    ax.axis("off")  # Turn off axes
    spacing = 0.1  # Spacing between blocks

    # Calculate total width and height with spacing for the background block
    total_width = grid_size + (grid_size) * spacing
    total_height = grid_size + (grid_size) * spacing

    # Draw large background rounded rectangle
    background = patches.FancyBboxPatch(
        (0, -0.5),
        total_width,
        total_height,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        edgecolor="black",
        facecolor="grey",
        zorder=0,
    )
    ax.add_patch(background)

    for i in range(grid_size):
        for j in range(grid_size):
            # Compute rectangle parameters
            x = j + (spacing * j) + spacing
            # Increased vertical spacing in the calculation
            y = grid_size - 1 - i - (spacing * i) - (spacing * (grid_size - 1 - i))
            width = 1 - spacing
            height = 1 - spacing
            value = values[i][j]
            color = colors[value]

            # Create a rounded rectangle
            rect = patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                edgecolor="black",
                facecolor=color,
            )
            ax.add_patch(rect)
            # Optionally add text inside each rectangle
            ax.text(
                x + width / 2,
                y + height / 2,
                str(value),
                va="center",
                ha="center",
                color="white",
            )

    # Add row names
    for i, name in enumerate(row_names):
        ax.text(
            -0.3,
            grid_size - 1 - i - (spacing * i) - spacing / 2,
            name,
            va="center",
            ha="right",
            color="black",
        )

    # Add column names
    for j, name in enumerate(col_names):
        ax.text(
            j + spacing * j + spacing + 0.5,
            grid_size + spacing,
            name,
            va="bottom",
            ha="center",
            color="black",
        )

    # Set the limits of the plot
    ax.set_xlim(0, grid_size + (grid_size + 1) * spacing)
    ax.set_ylim(-0.5, grid_size + (grid_size + 1) * spacing)

    # Show the plot in Streamlit
    st.pyplot(fig)


if __name__ == "__main__":
    st.write(
        """
    # My first app
    Hello *world!*
    """
    )

    dashboard_demo()
