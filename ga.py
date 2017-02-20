import random
import copy
from time import gmtime, strftime

class GA:
    def __init__(self, population_cnt: int, rate_mutation: float, rate_crossover: float, candidate_class):
        #
        self._population_size = population_cnt
        self._rate_mutation = rate_mutation
        self._rate_crossover = rate_crossover
        self._candidate_class= candidate_class
        self._start_time = strftime("%Y.%m.%d-%H:%M:%S",gmtime())
        self._candidate_id = 0
        # Create Random start population
        self._population = list(
            self._candidate_class(candidate_id=i, start_time_str=self._start_time) for i in range(self._population_size))
        self._candidate_id = self._population_size

        self.generation = 0
        self.best_candidate = None
        self.fitness_avg = None
        self.diversity = None

    def mutate(self):
        self.best_candidate = None
        self.fitness_avg = None
        self.diversity = None

        for candidate in self._population:
            candidate.mutation(self._rate_mutation)

    def crossover(self,strategy = "onePointSwap"):

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

            self._population[candidate1].crossover(crossover_rate=self._rate_crossover,other_candidate=self._population[candidate2],strategy=strategy)

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
                self.best_candidate = candidate
        self.fitness_avg /= len(self._population)

        # Compute Diversity if wanted
        if calc_diversity:
            self._calc_diversity()

    def _calc_diversity(self):
        divs = 0
        for idx_from, candidate_from in enumerate(self._population):
            for candidate_to in self._population[idx_from:]:
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
                new_population.append(copy.deepcopy(sorted_candidates[best_candidate_idx]))
            else:
                new_population.append(copy.deepcopy(sorted_candidates[worst_candidate_idx]))
        self._population = new_population