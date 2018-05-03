from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from maze_animation import MazeAnimation

def softmax_convert_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros([m, n])

    exp_theta = np.exp(beta * theta)

    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

    return np.nan_to_num(pi)

def get_action_and_next_s(s, Q, epsilon, pi):
    direction = ['up', 'right', 'down', 'left']

    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]

    if next_direction == 'up':
       action = 0
       s_next = s - 3
    elif next_direction == 'right':
       action = 1
       s_next = s + 1
    elif next_direction == 'down':
       action = 2
       s_next = s + 3
    elif next_direction ==  'left':
       action = 3
       s_next = s - 1

    return [action, s_next]

def sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
    
    return Q

def run_to_goal(Q, epsilon, eta, gamma, pi):
    s = 0
    s_history = [[0, np.nan]]

    while (1):
        [action, s_next] = get_action_and_next_s(s, Q, epsilon, pi)
        s_history[-1][1] = action
        
        s_history.append([s_next, np.nan])

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            [a_next, _] = get_action_and_next_s(s_next, Q, epsilon, pi)

        Q = sarsa(s, action, r, s_next, a_next, Q, eta, gamma)

        if s_next == 8:
            break
        else:
            s = s_next

    return [s_history, Q]

if __name__ == '__main__':
    theta0 = np.array([
       [np.nan, 1, 1, np.nan],
       [np.nan, 1, np.nan, 1],
       [np.nan, np.nan, 1, 1],
       [1, 1, 1, np.nan],
       [np.nan, np.nan, 1, 1],
       [1, np.nan, np.nan, np.nan],
       [1, np.nan, np.nan, np.nan],
       [1, 1, np.nan, np.nan],
    ])

    [a, b] = theta0.shape
    Q = np.random.rand(a, b) * theta0

    pi0 = softmax_convert_pi_from_theta(theta0)

    eta = 0.1
    gamma = 0.9
    epsilon = 0.5
    v = np.nanmax(Q, axis=1)
    is_continue = True
    episode = 1

    while is_continue:
        print("episode: " + str(episode))
        epsilon = epsilon / 2
        [s_history, Q] = run_to_goal(Q, epsilon, eta, gamma, pi0)

        new_v = np.nanmax(Q, axis=1)
        print(np.sum(np.abs(new_v - v)))
        v = new_v

        print('step: ' + str(len(s_history)))

        episode = episode + 1
        if episode > 100:
            break

    np.set_printoptions(precision=3, suppress=True)
    print(Q)
    # maze_animation = MazeAnimation()
    # ani = maze_animation.set_data(s_history)
    # maze_animation.show()
