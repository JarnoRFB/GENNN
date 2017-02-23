import random
import copy
from time import gmtime, strftime
import os

import matplotlib.pyplot as plt


class GA:

    def __init__(self, parms):
        #
        self._parms = parms
        self._population_size = parms['population']
        self._rate_mutation = parms['mutation']['rate']
        self._rate_crossover = parms['crossover']['rate']
        self._candidate_class = parms['candidate_class']

        self._start_time = strftime("%Y.%m.%d-%H.%M.%S", gmtime())
        self._candidate_id = 0
        # Create Random start population
        self._population = list(
            self._candidate_class(candidate_id=i, start_time_str=self._start_time, runtime_spec=parms['RUNTIME_SPEC'])
            for i in range(self._population_size)
        )
        self._candidate_id = self._population_size

        self.generation = 0
        self.best_candidate = None
        self.best_candidate_forever = None
        self.fitness_avg = None
        self.diversity = None

        # set base_logdir
        self._base_logdir = os.path.join(self._parms['RUNTIME_SPEC']['logdir'], str(self._start_time))
        os.makedirs(self._base_logdir, exist_ok=True)

        # Create running file
        file_loc = os.path.join(self._base_logdir, "_running")
        with open(file_loc, 'w') as fd:
            fd.write("running")

        # Save json

        file_loc = os.path.join(self._base_logdir, "ga.json")
        with open(file_loc, 'w') as fp:
            fp.write(str(self._parms))

        self._all_fitness_avg = list()
        self._all_fitness_best = list()
        self._all_diversity = list()

    def mutate(self):
        self.best_candidate = None
        self.fitness_avg = None
        self.diversity = None

        for candidate in self._population:
            candidate.mutation(self._rate_mutation)

    def crossover(self, strategy):

        if self._rate_crossover == 0:
            return
        self.best_candidate = None
        self.fitness_avg = None
        self.diversity = None
        # Number of Crossover operations
        crossovers = int((self._population_size * self._rate_crossover) / 2)
        for i in range(crossovers):
            candidate1 = random.randint(0, len(self._population) - 1)
            candidate2 = random.randint(0, len(self._population) - 1)
            # Get new Candidate 2 until Candidates are different
            while candidate1 == candidate2:
                candidate2 = random.randint(0, len(self._population) - 1)

            self._population[candidate1].crossover(other_candidate=self._population[candidate2],
                                                   crossover_parms=strategy)

    def evaluate(self, calc_diversity):
        self.diversity = 0
        self.fitness_avg = 0
        self.generation += 1
        # Here we can make Multi computing
        # Set: best_candidate and fitness_avg
        for candidate in self._population:
            candidate.to_next_generation(self.generation)
            self.fitness_avg += candidate.get_fitness()
            if self.best_candidate is None or candidate.get_fitness() > self.best_candidate.get_fitness():
                self.best_candidate = copy.deepcopy(candidate)
        self.fitness_avg /= len(self._population)
        if (self.best_candidate_forever is None or
            self.best_candidate_forever.get_fitness() < self.best_candidate.get_fitness()):
            # Copy best candidate.
            self.best_candidate_forever = copy.deepcopy(self.best_candidate)
        # Compute Diversity if wanted
        if calc_diversity:
            self._calc_diversity()

        self._all_fitness_avg.append(copy.copy(self.fitness_avg))
        self._all_fitness_best.append(copy.copy(self.best_candidate.get_fitness()))
        self._all_diversity.append(copy.copy(self.diversity))

    def write_stats(self):

        file = os.path.join(self._base_logdir, 'graph.png')
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.plot(self._all_fitness_avg, label='Fitness avg')
        ax.plot(self._all_fitness_best, label='Fitness best')
        ax.plot(self._all_diversity, label='diversity')
        ax.set_xlabel('Generation')
        ax.legend()
        fig.savefig(file, format='png')
        plt.clf()
        plt.close(fig)

        # Calc best Candidate more
        print("BestID: " + str(self.best_candidate_forever._candidate_id) + "- Fitness: " +
              str(round(self.best_candidate_forever.get_fitness(),3)))
        file_loc = os.path.join(self._base_logdir, "besetID")
        with open(file_loc, 'w') as fd:
            fd.write(str(self.best_candidate_forever._candidate_id))
        # Remove running file
        file_loc = os.path.join(self._base_logdir, "_running")
        os.remove(file_loc)

    def _calc_diversity(self):
        divs = 0
        for idx_from, candidate_from in enumerate(self._population):
            for candidate_to in self._population[idx_from + 1:]:
                self.diversity += candidate_from.get_diversity(candidate_to)
                divs += 1
        self.diversity /= divs

    def selection(self, strategy="Tournament", tournament_win_rate=0.75, tournament_size=10):
        if strategy == "Tournament":
            self._selection_tournament(tournament_win_rate, tournament_size)

    def _selection_tournament(self, win_rate=0.75, tournement_size=10):
        new_population = list()
        sorted_candidates = sorted(self._population, key=lambda x: x.get_fitness())

        for tournement in range(self._population_size):
            # Create tournement candidates
            idx_candidates = [i for i in range(self._population_size)]
            random.shuffle(idx_candidates)

            best_candidate_idx = max(idx_candidates[0:tournement_size])
            worst_candidate_idx = min(idx_candidates[0:tournement_size])

            if random.random() <= win_rate:
                # new_population.append(copy.deepcopy(sorted_candidates[best_candidate_idx]))
                network_spec_copy = copy.deepcopy(sorted_candidates[best_candidate_idx].network_spec)
                new_population.append(self._candidate_class(candidate_id=self._candidate_id,
                                                            start_time_str=self._start_time,
                                                            network_spec=network_spec_copy,
                                                            runtime_spec=self._parms['RUNTIME_SPEC']))
                self._candidate_id += 1
            else:
                # new_population.append(copy.deepcopy(sorted_candidates[worst_candidate_idx]))
                network_spec_copy = copy.deepcopy(sorted_candidates[worst_candidate_idx].network_spec)
                new_population.append(self._candidate_class(candidate_id=self._candidate_id,
                                                            start_time_str=self._start_time,
                                                            network_spec=network_spec_copy,
                                                            runtime_spec=self._parms['RUNTIME_SPEC']))
                self._candidate_id += 1
        self._population = new_population
