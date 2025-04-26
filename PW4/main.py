import numpy as np

from agent import QLearningAgent, draw_path_with_states


# Індекс    Координати
#      0    (0,0)
#      1    (0,2)
#      2    (1,1)
#      3    (1,3)
#      4    (2,0)
#      5    (2,2)
#      6    (3,1)
#      7    (3,3)

STATES = {
    '0': (0,0),
    '1': (0,2),
    '2': (1,1),
    '3': (1,3),
    '4': (2,0),
    '5': (2,2),
    '6': (3,1),
    '7': (3,3),
}


# Standard R matrix
# R = np.array([
#     [-1, -1, 0, -1, -1, 0, -1, 0],  # 0 - (0,0)
#     [-1, -1, 0, 0, 0, -1, -1, -1],  # 1 - (0,2)
#     [100, 0, -1, -1, 0, 0, -1, 0],    # 2 - (1,1)
#     [-1, 0, -1, -1, -1, 0, 0, -1],  # 3 - (1,3)
#     [-1, 0, 0, -1, -1, -1, 0, -1],  # 4 - (2,0)
#     [100, -1, 0, 0, -1, -1, 0, 0],    # 5 - (2,2)
#     [-1, -1, -1, 0, 0, 0, -1, -1],  # 6 - (3,1)
#     [100, -1, 0, -1, -1, 0, -1, -1],  # 7 - (3,3)
# ])

# R matrix with more rewards states
R = np.array([
    [-1, -1, 20, -1, -1, 20, -1, 20],  # 0 - (0,0)
    [-1, -1, 20, 0, 0, -1, -1, -1],  # 1 - (0,2)
    [100, 0, -1, -1, 0, 0, -1, 0],    # 2 - (1,1)
    [-1, 0, -1, -1, -1, 20, 0, -1],  # 3 - (1,3)
    [-1, 0, 20, -1, -1, -1, 0, -1],  # 4 - (2,0)
    [100, -1, 0, 0, -1, -1, 0, 0],    # 5 - (2,2)
    [-1, -1, -1, 0, 0, 20, -1, -1],  # 6 - (3,1)
    [100, -1, 0, -1, -1, 0, -1, -1],  # 7 - (3,3)
])

agent = QLearningAgent(R=R, goal_state=0)

print("Перша спроба (навчання довільно):")
path1 = agent.train_episode(start_state=6, verbose=True, states=STATES)  # start state 6 - (3,1)
print(f"\nШлях агента (1 спроба): {path1}\n")

# print("Перша спроба (навчання довільно):")
# path2 = agent.train_episode(start_state=6, verbose=True, states=STATES)  # start state 6 - (3,1)
# print(f"\nШлях агента (2 спроба): {path2}\n")

print("Друга спроба (оптимальний шлях на базі Q):")
path3 = agent.get_optimal_path(start_state=6)
print(f"Оптимальний шлях: {path3}\n")

agent.print_matrices()


draw_path_with_states(path3, STATES)
