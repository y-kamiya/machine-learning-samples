from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from maze_animation import MazeAnimation

def simple_convert_pi_from_theta(theta):
    [m,n] = theta.shape
    pi = np.zeros([m,n])
    for i in range(0, m):
        pi[i,:] = theta[i,:] / np.nansum(theta[i,:])

    pi = np.nan_to_num(pi)
    return pi

def get_next_s(pi, s):
    direction = ['up', 'right', 'down', 'left']
    next_direction = np.random.choice(direction, p=pi[s, :])
    if next_direction == 'up':
       s_next = s - 3
    elif next_direction == 'right':
       s_next = s + 1
    elif next_direction == 'down':
       s_next = s + 3
    elif next_direction ==  'left':
       s_next = s - 1

    return s_next

def run_to_goal(pi):
    s = 0
    s_history = [s]

    while (1):
       s_next = get_next_s(pi, s)
       s_history.append(s_next)

       if s_next == 8:
           break
       else:
           s = s_next

    return s_history


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

    pi0 = simple_convert_pi_from_theta(theta0)

    s_history = run_to_goal(pi0)
    print(s_history)
    print('step: ' + str(len(s_history)))

    maze_animation = MazeAnimation()
    ani = maze_animation.set_data(s_history)
    maze_animation.show()
