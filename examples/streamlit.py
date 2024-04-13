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


if __name__ == "__main__":
    st.write("""
    # My first app
    Hello *world!*
    """)

    visualize_matrix()
    visualize_blocks_2d()
