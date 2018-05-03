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

def get_action_and_next_s(pi, s):
    direction = ['up', 'right', 'down', 'left']
    next_direction = np.random.choice(direction, p=pi[s, :])
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

def run_to_goal(pi):
    s = 0
    s_history = [[s, np.nan]]

    while (1):
       [action, s_next] = get_action_and_next_s(pi, s)
       s_history[-1][1] = action
       s_history.append([s_next, np.nan])

       if s_next == 8:
           break
       else:
           s = s_next

    return s_history

def update_theta(theta, pi, s_history):
    eta = 0.1
    T = len(s_history)

    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(0, m):
        SA_i = [SA for SA in s_history if SA[0] == i]
        N_i = len(SA_i)
        for j in range(0, 4):
            if not(np.isnan(theta[i, j])):
                SA_ij = [SA for SA in s_history if SA == [i, j]]
                N_ij = len(SA_ij)
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T


    return theta + eta * delta_theta
    

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
    pi0 = softmax_convert_pi_from_theta(theta0)

    stop_epsilon = 10**-8
    theta = theta0
    pi = pi0

    is_continue = True
    count = -1
    while is_continue:
        s_history = run_to_goal(pi)
        new_theta = update_theta(theta, pi, s_history)
        new_pi = softmax_convert_pi_from_theta(new_theta)
        
        eps = np.sum(np.abs(new_pi - pi))
        print(eps)
        print('step: ' + str(len(s_history)))

        if eps < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi

    np.set_printoptions(precision=3, suppress=True)
    print(pi)
    # maze_animation = MazeAnimation()
    # ani = maze_animation.set_data(s_history)
    # maze_animation.show()
