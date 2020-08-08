import sys
import random
import copy

class OneMax():
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
    
    def evolve(self):
        for epoch in range(self.n_epochs):
            self.output(epoch)

            elite = self.find_elite()
            if elite.score() == self.dim:
                print('finish evolution on generation {}'.format(epoch))
                break

            next_populations = self.create_next_populations()

            self.populations = [elite] + self.select(self.n_populations - 1, self.populations + next_populations)

        self.output(self.n_epochs)

    def output(self, epoch):
        print('epoch {}: {}'.format(epoch, list(map(lambda p: p.score(), self.populations))))

    def create_next_populations(self):
        populations = []
        for i in range(self.n_populations):
            parent1 = self.populations[i]
            parent2 = self.populations[(i + 1) % self.n_populations]

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
    def __init__(self, dim, genes=None):
        self.dim = dim
        self.genes = genes
        if genes is None:
            self.genes = [random.randint(0, 1) for _ in range(dim)]

    def gene(self, start, end):
        return self.genes[start:end]

    def score(self):
        return sum(self.genes)

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
        cloned[index] = (self.genes[index] + 1) % 2

        return Population(self.dim, cloned)

if __name__ == '__main__':
    one_max = OneMax(10, 20, 50, 0.95)
    one_max.evolve()
