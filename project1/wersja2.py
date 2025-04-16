import random
import numpy as np
import matplotlib.pyplot as plt

# Opis: Program symuluje grę w "Kamień, Papier, Nożyce" pomiędzy graczem a komputerem,
#       aktualizując macierz przejść na podstawie wyników i uczenia się w czasie rzeczywistym.
##### GRACZ #####
# Definicja ruchów gracza:
#   wersja 1: na podstawie wektora stacjonarnego transition_matrix_computer,
#   wersja 2: w trakcie gry(iteracji) nauczenie gracza taktyki w postaci jego macierzy przejść
#             (inicjujemy macierz przejść gracza wypełnioną np. 1/3, a w trakcie gry po każdej rundzie aktualizujemy ją). 
# Należy napisać kod dla obu wersji (w osobnych plikach, albo w jednym pliku z możliwością zmiany taktyki jakimś parametrem)

# Inicjalizacja stanu gotówki gracza
cash = 0
cash_history = [cash]

##### KOMPUTER #####
# Definicja ruchów/taktyki komputera
states_computer = ["Paper", "Rock", "Scissors"]
transition_matrix_computer = {
    "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
    "Rock": {"Paper": 0/3, "Rock": 2/3, "Scissors": 1/3},
    "Scissors": {"Paper": 2/3, "Rock": 0/3, "Scissors": 1/3}
}
# Przekształcenie macierzy przejść transition_matrix_computer do postaci tablicy numpy
T = np.array([
    [2/3, 0, 2/3],
    [1/3, 2/3, 0],
    [0, 1/3, 1/3]
])
# Funkcja wybierająca ruch komputera na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru
def choose_move(player_previous_move):
    if player_previous_move is None:
        return random.choice(states_computer)
    return np.random.choice(states_computer, p=list(transition_matrix_computer[player_previous_move].values()))



transition_matrix_player = {
    "Paper": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3},
    "Rock": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3},
    "Scissors": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3}
}

# Funkcja aktualizująca macierz przejść gracza (wersja 2 taktyki gracza)
def update_transition_matrix(previous_move, current_move):
    if previous_move is not None:
        transition_matrix_player[previous_move][current_move] += 1
        total = sum(transition_matrix_player[previous_move].values())
        for move in transition_matrix_player[previous_move]:
            transition_matrix_player[previous_move][move] /= total
def update_transition_matrixminus(previous_move, current_move):
    if previous_move is not None and transition_matrix_player[previous_move][current_move] > 1:
        transition_matrix_player[previous_move][current_move] -= 1
        total = sum(transition_matrix_player[previous_move].values())
        for move in transition_matrix_player[previous_move]:
            transition_matrix_player[previous_move][move] /= total
# Funkcja wybierająca ruch gracza na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru (wersja 2 taktyki gracza)
player_previous_move = None
def choose_player_move():
    global player_previous_move
    if player_previous_move is None:
        return random.choice(states_computer)
    return np.random.choice(states_computer, p=list(transition_matrix_player[player_previous_move].values()))

def evaluate_game(player_move, computer_move):
    global cash
    if player_move == computer_move:
        return 0  
    elif (player_move == "Paper" and computer_move == "Rock") or (player_move == "Rock" and computer_move == "Scissors") or (player_move == "Scissors" and computer_move == "Paper"):
        cash += 1  
        return 1
    else:
        cash -= 1  
        return -1
# Główna pętla gry
computer_previous_move = None
for _ in range(10000):
    player_move = choose_player_move()
    computer_move = choose_move(computer_previous_move)

    result = evaluate_game(player_move, computer_move)

    if result == 1:
        update_transition_matrix(player_previous_move, player_move)
    elif result == -1:
        update_transition_matrixminus(player_previous_move, player_move)
    cash_history.append(cash)
    player_previous_move = player_move
    computer_previous_move = computer_move


# Wykres zmiany stanu gotówki w każdej kolejnej grze
plt.plot(range(10001), cash_history)
plt.xlabel('Numer Gry')
plt.ylabel('Stan Gotówki')
plt.title('Zmiana Stanu Gotówki w Grze "Kamień, Papier, Nożyce"')
plt.show()