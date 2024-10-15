from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from GhostBuster import gb_game_state, gb_game_rep, GhostBuster

# Import your GhostBuster classes and logic here
# from your_module import gb_game_state, gb_game_rep

app = Flask(__name__)

# Home route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the game simulation
@app.route('/simulate', methods=['POST'])
def simulate_game():
    num_campers = int(request.form['num_campers'])
    num_pairs = int(request.form['num_pairs'])

    # Initialize gb_game_state and gb_game_rep
    try:
        game_state = gb_game_state(num_campers, num_pairs, "random")
        game_state.create_ghostbusters()
        game_state.set_ghostbuster_positions_rand()

        # Initialize gb_game_rep
        game_rep = gb_game_rep(game_state, verbose=True)  # Set verbose to False to suppress plots

        # Create and evaluate the game representation
        game_rep.create_game_rep()
        game_rep.evaluate_game()
        game_rep.write_game_result()

        # Generate plot if game is verbose
        plot_url = generate_plot(game_rep)

        # Return the result and the image to the frontend
        return jsonify({
            'result': game_rep.game_result,
            'plot_url': plot_url
        })

    except ValueError as e:
        return jsonify({'error': str(e)})


def generate_plot(game_rep):
    """Helper function to generate the game plot and return it as a base64-encoded image."""
    # Create the plot
    plt.figure(figsize=(5, 5))
    game_rep.create_game_rep()  # Will create the plot
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode image to base64 and return it
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free memory
    return f"data:image/png;base64,{plot_url}"


if __name__ == '__main__':
    app.run(debug=True)
