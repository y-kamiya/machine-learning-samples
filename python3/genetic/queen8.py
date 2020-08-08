import sys
import random
import copy
import numpy as np
import argparse

class Solver():
    def __init__(self, dim, n_populations, n_epochs, cross_rate):
        self.dim = dim
        self.n_populations = n_populations
        self.n_epochs = n_epochs
        self.cross_rate = cross_rate

        self.populations = [Population(dim) for _ in range(n_populations)]

    def find_elite(self):
        scores = [p.score() for p in self.populations]
        index = scores.index(max(scores))
        return self.populations[index]

    def execute(self):
        for epoch in range(self.n_epochs):
            elite = self.find_elite()
            # print([p.genes for p in self.populations])
            self.output(epoch, elite)

            if elite.score() == Population.SCORE_MAX:
                print('finish evolution on epoch {}'.format(epoch))
                print(elite.arrangement())
                return

            next_populations = self.create_next_populations()

            self.populations = [elite] + self.select(self.n_populations - 1, self.populations + next_populations)

        self.output(self.n_epochs, self.find_elite())

    def output(self, epoch, population):
        print('epoch {}: {}'.format(epoch, list(map(lambda p: int(1 / p.score()), self.populations))))
        print(population.genes)

    def create_next_populations(self):
        populations = []
        for i in range(self.n_populations):
            parent1, parent2 = random.choices(self.populations, k=2)

            if random.random() < self.cross_rate:
                child = parent1.cross(parent2)
            else:
                child = parent1.mutate()

            populations.append(child)

        return populations

    def select(self, n_select, populations):
        selection = []
        scores = [p.score() for p in populations]
        score_total = sum(scores)

        for _ in range(n_select):
            threshold = int(random.random() * score_total)
            score_sum = 0
            for i, score in enumerate(scores):
                score_sum += score
                if threshold < score_sum:
                    selection.append(populations[i])
                    break

        return selection


class Population():
    SCORE_MAX = 1.0

    def __init__(self, dim, genes=None):
        self.dim = dim
        self.genes = genes
        if genes is None:
            self.genes = random.choices(range(dim), k=dim)

    def gene(self, start, end):
        return self.genes[start:end]

    def arrangement(self):
        board = np.zeros((self.dim, self.dim), dtype=int)
        for i, queen in enumerate(self.genes):
            board[i][queen] = 1
        return board

    def score(self):
        est = 0
        board = self.arrangement()
        dim = self.dim
        for i, queen in enumerate(self.genes):
            q_x = queen
            q_y = i
            for j in range(dim):
                #縦にqueenがあるか？
                if board[j][q_x] == 1 and j != q_y:
                    est += 1
                
                #斜めにqueenがあるか？
                vertical = i+j
                horizontal = q_x + j
                if vertical < dim and horizontal < dim:
                    if board[vertical][horizontal] == 1 and j != 0:
                        est += 1
                        
                vertical = i-j
                horizontal = q_x - j
                if vertical > -1 and horizontal > -1:
                    if board[vertical][horizontal] == 1 and j != 0:
                        est += 1
                        
                vertical = i-j
                horizontal = q_x + j
                if vertical > -1 and horizontal < dim:
                    if board[vertical][horizontal] == 1 and j != 0:
                        est += 1
                        
                vertical = i+j
                horizontal = q_x - j
                if vertical < dim and horizontal > -1:
                    if board[vertical][horizontal] == 1 and j != 0:
                        est += 1
        if est == 0:
            return Population.SCORE_MAX

        return 1 / est

    def cross(self, population):
        indexes = random.sample(range(self.dim), 2)
        r1 = min(indexes[0], indexes[1])
        r2 = max(indexes[0], indexes[1])

        cloned = copy.deepcopy(self.genes)
        cloned[r1:r2] = population.gene(r1, r2)

        return Population(self.dim, cloned)

    def mutate(self):
        index = random.choice(range(self.dim))

        cloned = copy.deepcopy(self.genes)
        cloned[index] = random.choice(range(self.dim))

        return Population(self.dim, cloned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dim', type=int, default=8, help='epoch count')
    parser.add_argument('--epochs', type=int, default=1000, help='epoch count')
    parser.add_argument('--populations', type=int, default=50, help='epoch count')
    parser.add_argument('--cross_rate', type=float, default=0.5, help='epoch count')
    args = parser.parse_args()
    print(args)

    solver = Solver(args.dim, args.populations, args.epochs, args.cross_rate)
    solver.execute()
