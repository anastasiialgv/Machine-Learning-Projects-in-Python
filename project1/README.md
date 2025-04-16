This Python project simulates the classic game of Rock, Paper, Scissors, where the computer follows a predefined or adaptive strategy based on transition probabilities. 
The goal is to analyze how a player can perform against different computer strategies and track the player's cash balance over time.

Project Description
The computer uses a transition matrix to choose its next move based on its previous choice:
transition_matrix_computer = {
    "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
    "Rock": {"Paper": 0/3, "Rock": 2/3, "Scissors": 1/3},
    "Scissors": {"Paper": 2/3, "Rock": 0/3, "Scissors": 1/3}
}
The player has two possible strategies:

Version 1: The player predicts the next computer move based on the stationary vector of the transition matrix.
Version 2: The player continuously updates the computer's transition matrix during the game based on observed moves.

Game Rules
Win: +1 point
Loss: -1 point
Draw: 0 points

The player’s cash balance is updated accordingly after each game.

Simulation
The simulation runs 10,000 rounds of Rock, Paper, Scissors.
A plot shows how the player's cash balance changes over time.

Technologies Used:
matplotlib for plotting
random for move selection
numpy (optional for matrix operations)
