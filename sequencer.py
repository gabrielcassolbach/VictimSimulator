import numpy as np

class Sequencer:
    def __init__(self):
        self.name = "Sequencer"

    def euclidean_distance(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def total_distance(self, route):
        return sum(self.euclidean_distance(route[i], route[i+1])
                   for i in range(len(route)-1))

    def bfs_like_sequence(self, victims):
        unvisited = victims.copy()
        current = (0, 0)
        sequence = []

        while unvisited:
            nearest = min(unvisited, key=lambda v: self.euclidean_distance(current, v))
            sequence.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return sequence

    #Algoritmo de busca local opt-2
    def two_opt(self, route):
        best = route
        
        improved = True

        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1: continue  #
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if self.total_distance(new_route) < self.total_distance(best):
                        best = new_route
                        improved = True
        
        return best

