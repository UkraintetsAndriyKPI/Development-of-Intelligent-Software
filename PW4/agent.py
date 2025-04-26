import random
import numpy as np


class QLearningAgent:
    def __init__(self, R, goal_state, gamma=0.5, alpha=0.7):
        """
        :param R: Матриця винагород середовища (2D numpy масив).
        :param goal_state: Номер цільового стану.
        :param gamma: Коефіцієнт майбутніх нагород (0 < gamma ≤ 1).
        :param alpha: Швидкість навчання агента (0 < alpha ≤ 1).
        """
        self.R = R
        self.Q = np.zeros_like(R, dtype=float)
        self.goal_state = goal_state
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = R.shape[0]

    def available_actions(self, state):
        return np.where(self.R[state] >= 0)[0]

    # def choose_action(self, state):
    #     actions = self.available_actions(state)
    #     return random.choice(actions)

    def choose_action(self, state):
        actions = self.available_actions(state)
        if len(actions) == 0:
            return None
        rewards = [self.R[state, a] for a in actions]
        best_action = actions[np.argmax(rewards)]
        return best_action

    def update_Q(self, current_state, action):
        max_Q = np.max(self.Q[action])
        self.Q[current_state, action] = (1 - self.alpha) * self.Q[current_state, action] + \
                                         self.alpha * (self.R[current_state, action] + self.gamma * max_Q)

    def train_episode(self, start_state=None, verbose=False, states=None):
        if start_state is None:
            state = random.randint(0, self.n_states - 1)
        else:
            state = start_state
        path = [state]

        while state != self.goal_state:
            action = self.choose_action(state)
            self.update_Q(state, action)
            if verbose:
                if states:
                    print(f"Зі стану {state} {states[str(state)]} переходимо до {action} - {states[str(action)]}")
                else:
                    print(f"Зі стану {state} переходимо до {action}")

            state = action
            path.append(state)

        return path

    def get_optimal_path(self, start_state):
        state = start_state
        path = [state]

        while state != self.goal_state:
            actions = self.available_actions(state)
            best_action = actions[np.argmax([self.Q[state, a] for a in actions])]
            state = best_action
            path.append(state)

        return path

    def print_matrices(self):
        print("Матриця R (нагороди):")
        print(self.R)
        print("\nМатриця Q (знання агента після навчання):")
        print(np.round(self.Q, 2))


def draw_path_with_states(path, states, board_size=4):
    """
    :param path: список станів (номерів комірок як int або str)
    :param states: словник номер -> (рядок, стовпець)
    :param board_size: розмір дошки (4 для 4x4)
    """
    board = [["." for _ in range(board_size)] for _ in range(board_size)]

    print(path)

    for idx, state in enumerate(path):
        state_str = str(state)
        if state_str not in states:
            print(f"Попередження: стан {state} не описаний у states, пропускаю.")
            continue

        row, col = states[state_str]

        if idx == 0:
            board[row][col] = "S"  # Start
        elif idx == len(path) - 1:
            board[row][col] = "G"  # Goal
        else:
            board[row][col] = str(idx)  # Шлях

    print("\nШлях агента на дошці:")
    for row in board:
        print(" ".join(row))
