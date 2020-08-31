import sys
import random
import copy
import argparse

class Solver():
    def __init__(self, dim, n_individuals, n_epochs, cross_rate):
        self.dim = dim
        self.n_individuals = n_individuals
        self.n_epochs = n_epochs
        self.cross_rate = cross_rate

        self.population = [Individual(dim) for _ in range(n_individuals)]

    def find_elite(self):
        scores = [p.score() for p in self.population]
        index = scores.index(max(scores))
        return self.population[index]
    
    def execute(self):
        for epoch in range(self.n_epochs):
            self.output(epoch)

            elite = self.find_elite()
            if elite.score() == self.dim:
                print('finish evolution on generation {}'.format(epoch))
                return

            next_populations = self.create_next_populations()

            self.population = [elite] + self.select(self.n_individuals - 1, self.population + next_populations)

        self.output(self.n_epochs)

    def output(self, epoch):
        print('epoch {}: {}'.format(epoch, list(map(lambda p: p.score(), self.population))))

    def create_next_populations(self):
        population = []
        for i in range(self.n_individuals):
            parent1 = self.population[i]
            parent2 = self.population[(i + 1) % self.n_individuals]

            if random.random() < self.cross_rate:
                child = parent1.cross(parent2)
            else:
                child = parent1.mutate()

            population.append(child)

        return population

    def select(self, n_select, population):
        selection = []
        scores = [p.score() for p in population]
        score_total = sum(scores)

        for _ in range(n_select):
            threshold = int(random.random() * score_total)
            score_sum = 0
            for i, score in enumerate(scores):
                score_sum += score
                if threshold < score_sum:
                    selection.append(population[i])
                    break

        return selection

class Individual():
    def __init__(self, dim, genes=None):
        self.dim = dim
        self.genes = genes
        if genes is None:
            self.genes = [random.randint(0, 1) for _ in range(dim)]

    def gene(self, start, end):
        return self.genes[start:end]

    def score(self):
        return sum(self.genes)

    def cross(self, individual):
        indexes = random.sample(range(self.dim), 2)
        r1 = min(indexes[0], indexes[1])
        r2 = max(indexes[0], indexes[1])

        cloned = copy.deepcopy(self.genes)
        cloned[r1:r2] = individual.gene(r1, r2)

        return Individual(self.dim, cloned)

    def mutate(self):
        index = random.choice(range(self.dim))

        cloned = copy.deepcopy(self.genes)
        cloned[index] = (self.genes[index] + 1) % 2

        return Individual(self.dim, cloned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--individuals', type=int, default=50)
    parser.add_argument('--cross_rate', type=float, default=0.95)
    args = parser.parse_args()
    print(args)

    solver = Solver(args.dim, args.individuals, args.epochs, args.cross_rate)
    solver.execute()
