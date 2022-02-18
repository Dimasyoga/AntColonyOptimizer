import numpy as np
import time
from matplotlib import pyplot as plt

# add test comment
# add test comment 2
class ACO:

    def __init__(self, graph, num=15, alpha=0.6, beta=0.4, evaporation_coef=0.3):
        
        self._num_ants = num
        self._alpha = alpha
        self._beta = beta
        self._pheromone_evaporation_coeff = evaporation_coef
        self._choose_best = .0
        self._node_list = None

        self._graph_matrix = np.array(graph)
        self._pheromone_matrix = None
        self._visibility_matrix = None
        self._probability_matrix = None

        self._ants_path = []
        self._ants_path_cost = []
        self._ants_available_node = []
        self._best_path = []
    
    def initialization(self):

        self._node_list = np.arange(len(self._graph_matrix))

        for i in range(self._num_ants):
            self._ants_path.append([np.random.choice(self._node_list)])
            self._ants_path_cost.append(0)

        for i in range(self._num_ants):
            self._ants_available_node.append([x for x in self._node_list if x != self._ants_path[i][0]])

        self._pheromone_matrix = np.ones(shape=[len(self._node_list), len(self._node_list)])
        self._visibility_matrix = np.empty(shape=[len(self._node_list), len(self._node_list)])
        
        for m in range(len(self._node_list)):
            for n in range(len(self._node_list)):
                if m != n:
                    self._visibility_matrix[m][n] = 1 / self._graph_matrix[m][n]
        
        self._probability_matrix = (self._pheromone_matrix ** self._alpha) * (self._visibility_matrix ** self._beta)
    
    def update(self):
        self._probability_matrix = (self._pheromone_matrix ** self._alpha) * (self._visibility_matrix ** self._beta)

        self._ants_path = []
        self._ants_path_cost = []
        self._ants_available_node = []

        for i in range(self._num_ants):
            self._ants_path.append([np.random.choice(self._node_list)])
            self._ants_path_cost.append(0)

        for i in range(self._num_ants):
            self._ants_available_node.append([x for x in self._node_list if x != self._ants_path[i][0]])
    
    def update_pheromone_matrix(self):
        self._pheromone_matrix = ((1 - self._pheromone_evaporation_coeff) ** self._pheromone_matrix)
        for i in range(self._num_ants):
            for j in range(len(self._ants_path[i]) - 1):
                self._pheromone_matrix[self._ants_path[i][j]][self._ants_path[i][j+1]] += 1 - (self._ants_path_cost[i] / sum(self._ants_path_cost))

    def calculate_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self._graph_matrix[path[i]][path[i+1]]
        return cost
    
    def next_node(self, src, ant):

        numerator = self._probability_matrix[src, self._ants_available_node[ant]]
        if np.random.random() < self._choose_best:
            next_node = self._ants_available_node[ant][np.argmax(numerator)]
        else:
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            next_node = np.random.choice(self._ants_available_node[ant], p=probabilities)
        return next_node

    def fit(self, iteration):
        fg = plt.figure("Probability Matrix")
        ax = fg.gca()

        start = time.time()
        self.initialization()
        h = ax.imshow(self._probability_matrix, interpolation='nearest')
        for _ in range(iteration):
            h.set_data(self._probability_matrix)
            plt.draw(), plt.pause(1e-3)
            
            for i in range(self._num_ants):
                while len(self._ants_available_node[i]) > 0:
                    next_node = self.next_node(self._ants_path[i][-1], i)
                    self._ants_path[i].append(next_node)
                    self._ants_available_node[i].remove(next_node)
                self._ants_path[i].append(self._ants_path[i][0])
                self._ants_path_cost[i] = self.calculate_cost(self._ants_path[i])
            self._best_path = self._ants_path[np.argmin(self._ants_path_cost)]
            self.update_pheromone_matrix()
            self.update()
            self._choose_best += 1 / iteration

        print(f"\ntime: {time.time() - start} s  min cost: {self.calculate_cost(self._best_path)}\n")
        print(self._best_path)
        
