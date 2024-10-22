import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

from GhostBuster import gb_game_state, gb_game_rep, GhostBuster  # Ensure your GhostBuster logic is imported correctly
from GhostBuster import GBDashData  # Import your GBDashData class
from GhostBuster import GBDashViz  # Import your GBDashViz class

# Streamlit app
st.title("GhostBuster Game Simulator")

# Sidebar inputs for game parameters
num_campers = st.sidebar.number_input('Number of Campers', min_value=1, value=10, step=1)
num_pairs = st.sidebar.number_input('Number of Pairs', min_value=1, value=5, step=1)

# Initialize GBDashData instance for database connection and querying
dashboard_data = GBDashData()

# Function to generate game plot
def generate_plot(game_rep, title="Game Representation"):
    """Helper function to generate the game plot and return the image for Streamlit display."""
    plt.figure(figsize=(5, 5))
    game_rep.create_game_rep()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return img

# Simulate the game when the button is clicked
if st.sidebar.button("Simulate Game"):
    try:
        # Initialize gb_game_state and gb_game_rep
        game_state = gb_game_state(num_campers, num_pairs, "random")
        game_state.create_ghostbusters()
        game_state.set_ghostbuster_positions_rand()

        # Initialize gb_game_rep
        game_rep = gb_game_rep(game_state, verbose=True)

        # Create and evaluate the game representation
        game_rep.create_game_rep()
        game_rep.evaluate_game()
        game_rep.write_game_result()

        # Display the result
        st.write(f"Game Result: {game_rep.game_result}")

        # Generate and display the plot
        st.image(generate_plot(game_rep), use_column_width=True)

    except ValueError as e:
        st.error(f"Error: {str(e)}")

# Add button to refresh the data using GBDashData and GBDashViz
if st.sidebar.button("Refresh Data"):
    try:
        # Refresh the data from the database
        dashboard_data.refresh_data()

        # Create a GBDashViz instance to visualize the refreshed data
        dashboard_viz = GBDashViz(dashboard_data)

        # First row with 2 columns
        col1, col2 = st.columns(2)

        # First column: Actual vs Target
        with col1:
            st.subheader("Actual vs Target")
            dashboard_viz.plot_actual_pct()

        # Second column: Success Rate by Number of Campers
        with col2:
            st.subheader("Success Rate by Number of Campers")
            # Increase the size of the plot to make it larger
            plt.figure(figsize=(8, 6))  # Adjust this figure size as needed
            dashboard_viz.plot_success_rate_by_campers()

        # Second row with 2 columns
        col3, col4 = st.columns(2)

        # Third column: Success Rate by Number of Pairs
        with col3:
            st.subheader("Success Rate by Number of Pairs")
            # Increase the size of the plot to make it larger
            plt.figure(figsize=(8, 6))  # Adjust this figure size as needed
            dashboard_viz.plot_success_rate_by_pairs()

        # Fourth column: Simulations Run
        with col4:
            st.subheader("Simulations Run")
            dashboard_viz.plot_sims_run()

    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")



if __name__ == '__main__':
    pass