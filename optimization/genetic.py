import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from random import randrange, randint
from tqdm.auto import tqdm

def average(lst):
    return sum(lst) / len(lst)

class Individual:
    def __init__(self, GA, route, timepoints):
        self.route = route
        self.timepoints = timepoints
        self._fitness = 0
        self._preference_value = 0
        self._travel_cost = 0
        self._GA = GA

    @property
    def travel_cost(self):  # Get total travelling cost
        if self._travel_cost == 0:
            for i in range(len(self.route)):
                origin = self.route[i]
                dest = self.route[0]
                if i != len(self.route) - 1:
                    dest = self.route[i+1]

                t = self.timepoints[i]

                self._travel_cost += self._GA._A[origin, dest, t]

            self._fitness = 0

        return self._travel_cost

    @property
    def preference_value(self):
        if self._preference_value == 0:
            for i in self.route:
                self._preference_value += self._GA._P[i]

        return self._preference_value

    @property
    def fitness(self):
        if self._fitness == 0:

            c = self.travel_cost

            self._fitness = 2

            if c > self._GA._budget:
                self._fitness -= 1

            if self.timepoints[-1] > self._GA._A.shape[2]:
                self._fitness -= 0.5

            scaled = ((self.preference_value - self._GA._min_pref) / (self._GA._max_pref - self._GA._min_pref)) / self._GA._n
            self._fitness += scaled

        return self._fitness

    def __str__(self):
        res = ""

        for i in range(len(self.route) - 1):
            res += "edge " + str(self.route[i]) + "-" + str(self.route[i+1]) + " at time " + str(self.timepoints[i]) + "\n"

        res += "edge " + str(self.route[-1]) + "-" + str(self.route[0]) + " at time " + str(self.timepoints[-1])

        return res

    def __reset_params(self):
        self._travel_cost = 0
        self._fitness = 0
        self._preference_value = 0


class GA:

    @staticmethod
    def default_params():
        def_params = {}
        def_params['population size'] = 1000
        def_params['crossover rate'] = 0.3
        def_params['dying rate'] = 0.6
        def_params['survival rate'] = 0.1
        def_params['nIterations'] = 300
        def_params['nFittest'] = 40

        return def_params

    def __init__(self, A, P, transfer_time, budget):
        self.population = []
        self.params = GA.default_params()
        self.stats = {}
        self.current_run = None
        self._A = A
        self._P = P
        self._min_pref = min(P)
        self._max_pref = max(P)
        self._T = transfer_time
        self._budget = budget
        self._tmax = A.shape[2]-1
        self._n = A.shape[0]

    def set_parameter(self,k,p):
        self.params[k] = p

    def compute_unvisited(self, ind):
        return list(set(range(self._n)) - set(ind.route))

    def add_stats_to_current_run(self, key, value):

        curr_key = self.current_run

        if not curr_key in self.stats:
            self.stats[curr_key] = {}

        if key == 'ev_fitness':
            if not key in self.stats[curr_key]:
                self.stats[curr_key][key] = []

            self.stats[curr_key][key].append(value)

    def generate_individual(self, max_nodes):

        max_tour_size = int(min(self._n / self._T, self._tmax))

        route = [0] + random.sample(range(1,max_tour_size), max_nodes-1)
        time = [i * self._T for i in range(len(route))]

        return Individual(self, route, time)

    def crossover(self, ind1, ind2):

        strategies = ['merge', 'shuffle-merge']
        strategy = random.sample(strategies,1)[0]

        route1 = ind1.route[:]
        route2 = ind2.route[:]
        timepoints1 = ind1.timepoints[:]
        timepoints2 = ind2.timepoints[:]

        new_route = ind1.route[:]
        new_timepoints = ind1.timepoints[:]

        if strategy == 'merge':

            first_half = int(len(route1)/2)
            second_half = int(len(route2)/2)

            new_route = route1[:first_half] + route2[second_half:]

            # remove duplicates
            new_route = list(dict.fromkeys(new_route))
            new_timepoints = [i * self._T for i in range(len(new_route))]

        if strategy == 'shuffle-merge':
            all_nodes = set(route1 + route2)
            new_route = list(all_nodes)

            random.shuffle(new_route)
            new_route = new_route[:len(route1)]

            # fix 0 as start
            new_route = [0] + [i for i in new_route if i!=0]

            new_timepoints = [i * self._T for i in range(len(new_route))]

        return Individual(self, new_route, new_timepoints)

    def mutate_individual(self, ind):

        new_route = ind.route[:]
        new_timepoints = ind.timepoints[:]

        strategies = ['swap_intern', 'swap_extern', 'add', 'remove', 'rotate', 'time_change', 'tight_time_schedule']

        if len(new_route) <= 2:
            strategies = ['add']

        strategy = random.sample(strategies,1)[0]

        if strategy == 'swap_extern':
            unvisited = self.compute_unvisited(ind)
            new_node = random.sample(unvisited,1)[0]
            index = random.sample(range(1,len(new_route)),1)[0]
            new_route[index] = new_node

        if strategy == 'swap_intern':
            idx = range(1,len(new_route))
            i1, i2 = random.sample(idx, 2)
            new_route[i1], new_route[i2] = new_route[i2], new_route[i1]

        elif strategy == 'add':
            unvisited = self.compute_unvisited(ind)

            new_route.append(random.sample(unvisited,1)[0])
            new_timepoints.append(new_timepoints[-1] + self._T)

        elif strategy == 'remove':
            index = random.sample(range(1,len(new_route)),1)[0]
            new_route.pop(index)
            new_timepoints.pop(index)

        elif strategy == 'rotate':
            idx = range(1,len(new_route))
            i1, i2 = random.sample(idx, 2)
            lb, ub = min(i1,i2), max(i1,i2)

            first = new_route[lb]

            for el in range(lb,ub):
                new_route[el] = new_route[el+1]

            new_route[ub] = first

        elif strategy == 'inver_over':

            idx = range(1,len(new_route)+1)
            i1, i2 = random.sample(idx, 2)
            lb, ub = min(i1,i2), max(i1,i2)

            new_route[lb:ub] = new_route[lb:ub][::-1]


        elif strategy == 'time_change':
            index = random.sample(range(1,len(new_route)),1)[0]
            extra_time = randrange(1, self._T + 1)
            new_timepoints[index:] = [tp + extra_time for tp in new_timepoints[index:]]

        elif strategy == 'tight_time_schedule':
            new_timepoints = [i * self._T for i in range(len(new_route))]

        return Individual(self, new_route, new_timepoints)


    def create_initial_population(self):
        # this might fail if there are too many nodes

        list_of_individuals = []

        for i in range(self.params['population size']):
            nNodes = randint(1, max(2, int(0.8 * (self._tmax / self._T))))
            ind = self.generate_individual(nNodes)
            list_of_individuals.append(ind)

        return list_of_individuals

    def evolve_population(self, population):

        pop_sorted = sorted(population, key=lambda x: x.fitness, reverse=True)

        best_fitness = pop_sorted[0].fitness
        self.add_stats_to_current_run('ev_fitness', best_fitness)

        new_pop = pop_sorted[: int(len(pop_sorted) * self.params['dying rate'])]

        for p in pop_sorted[int(len(new_pop) * self.params['dying rate']):]:
            if random.random() <= self.params['survival rate']:
                new_pop.append(p)


        while len(new_pop) < self.params['population size']:

            if random.random() <= self.params['crossover rate']:
                old_inds = random.sample(new_pop, 2)
                new_ind = self.crossover(old_inds[0], old_inds[1])

                new_pop.append(new_ind)

            else:

                old_ind = random.sample(new_pop, 1)[0]
                new_ind = self.mutate_individual(old_ind)

                new_pop.append(new_ind)

        return new_pop

    def get_average_fitness(pop):
        return average([i.fitness for i in pop])

    def run(self):

        final_pop = []

        for i in tqdm(range(self.params['nFittest'])):

            self.current_run = "Iteration " + str(i)

            pop = self.create_initial_population()

            for i in range(self.params['nIterations']):
                pop = self.evolve_population(pop)

            fittest = self.get_fittest(pop)
            final_pop.append(fittest)

        self.current_run = "Final Genetic Algorithm"
        for i in range(self.params['nIterations']):
            final_pop = self.evolve_population(final_pop)

        return final_pop[0]


    def plot(self):
        for k in self.stats:

            print(k)

            plt.title(k)
            plt.plot(range(self.params['nIterations']), self.stats[k]['ev_fitness'])
            plt.xlabel("Evolution steps")
            plt.ylabel("Fitness")
            plt.tight_layout()
            plt.show()

        return 0


    def get_fittest(self, pop):
        fittest = pop[0]
        for ind in pop:
            if ind.fitness > fittest.fitness:
                fittest = ind

        return fittest
