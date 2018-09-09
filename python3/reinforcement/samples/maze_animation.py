from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


