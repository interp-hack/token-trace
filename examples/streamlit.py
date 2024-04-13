import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap


def visualize_matrix():
    # Create a 2D matrix using numpy
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Display the matrix
    st.dataframe(matrix)


def visualize_blocks_2d():
    st.write("Visualizing a 2D matrix with custom colors")

    # Define the matrix values
    matrix = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 3]])

    # Define colors for each value
    colors = ["red", "green", "blue"]  # Red for 1, green for 2, blue for 3
    cmap = ListedColormap(colors)

    # Create a figure and a subplot
    fig, ax = plt.subplots()

    # Use matshow to display the matrix with the custom colors
    cax = ax.matshow(matrix, cmap=cmap)

    # Add colorbar to map values to colors
    plt.colorbar(
        cax,
        ticks=np.arange(min(np.unique(matrix)), max(np.unique(matrix)) + 1, 1),
        spacing="proportional",
    )

    # Optionally add numbers on the blocks
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{val}", ha="center", va="center", color="white")

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_blocks_rounded():
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
    st.write("""
    # My first app
    Hello *world!*
    """)

    visualize_matrix()
    visualize_blocks_2d()
    visualize_blocks_rounded()
