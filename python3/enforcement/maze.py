from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

class MazeAnimation():
    def __init__(self):
        # 図を描く大きさと、図の変数名を宣言
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
         
        # 赤い壁を描く
        plt.plot([1, 1], [0, 1], color='red', linewidth=2)
        plt.plot([1, 2], [2, 2], color='red', linewidth=2)
        plt.plot([2, 2], [2, 1], color='red', linewidth=2)
        plt.plot([2, 3], [1, 1], color='red', linewidth=2)
         
        # 状態を示す文字S0～S8を描く
        plt.text(0.5, 2.5, 'S0', size=14, ha='center')
        plt.text(1.5, 2.5, 'S1', size=14, ha='center')
        plt.text(2.5, 2.5, 'S2', size=14, ha='center')
        plt.text(0.5, 1.5, 'S3', size=14, ha='center')
        plt.text(1.5, 1.5, 'S4', size=14, ha='center')
        plt.text(2.5, 1.5, 'S5', size=14, ha='center')
        plt.text(0.5, 0.5, 'S6', size=14, ha='center')
        plt.text(1.5, 0.5, 'S7', size=14, ha='center')
        plt.text(2.5, 0.5, 'S8', size=14, ha='center')
        plt.text(0.5, 2.3, 'START', ha='center')
        plt.text(2.5, 0.3, 'GOAL', ha='center')
         
        # 描画範囲の設定と目盛りを消す設定
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False, labelleft=False)
         
        self.fig = fig
        self.ax = ax

    def __animation_init(self):
        self.line, = self.ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
        return (self.line,)

    def __animation_update(self, i):
        s = self.state_history[i]
        x = (s % 3) + 0.5
        y = 2.5 - int(s / 3)
        self.line.set_data(x, y)
        return (self.line,)

    def set_data(self, state_history):
        self.state_history = state_history
        return animation.FuncAnimation(self.fig, self.__animation_update, frames=len(state_history), interval=100, init_func=self.__animation_init, repeat=False)

    def show(self):
        plt.show()


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
