import random
import numpy as np


class GA:
    def __init__(self, population_cnt, rate_mutation, rate_crossover, sampleCandidate):
        # INt
        self._population_size = population_cnt
        # double
        self._rate_mutation = rate_mutation
        self._rate_crossover = rate_crossover
        # Candidate Class
        self._sampleCandidate = sampleCandidate
        # Candidaten einer Population
        self._population = list(self._sampleCandidate() for i in range(self._population_size))

        self.generation = 0
        self.best_candidate = None
        self.fitness_avg = 0
        self.diversity = 0

    def mutate(self):
        for candidate in self._population:
            candidate.mutation(self._rate_mutation)

    def crossover(self):
        if self._rate_crossover == 0:
            return
        # Number of Crossover operations
        crossovers = int((self._population_size * self._rate_crossover) / 2)
        for i in range(crossovers):
            candidate1 = random.randint(0, len(self._population) - 1)
            candidate2 = random.randint(0, len(self._population) - 1)
            # Get new Candidate 2 until Candidates are different
            while candidate1 == candidate2:
                candidate2 = random.randint(0, len(self._population) - 1)

        self._population[candidate1].crossover(self._population[candidate2])

    def evaluate(self, calc_diversity):
        self.diversity = 0
        self.fitness_avg = 0
        # Here we can make Multi computing
        # Compute eacht Fitness
        for candidate in self._population:
            candidate.evaluate()
            self.fitness_avg += candidate.get_fitness()
        self.fitness_avg /= len(self._population)
        # Compute Diversity if wanted

        divs = 0  # Lazy
        if calc_diversity:
            for idx_from, candidate_from in enumerate(self._population):
                for candidate_to in self._population[idx_from:]:
                    self.diversity = candidate_from.get_diversity(candidate_to)
                    divs += 1
        self.diversity /= divs

    def selection(self, strategy="Tournament", tournament_win_rate=0.75, tournament_size=10):
        if strategy == "Tournament":
            self._selection_tournement(tournament_win_rate, tournament_size)

    #IMPROVMENT: Create a List sorted by get_fitness(), create a 2 list with tournement_size random indizes
    # and append the higest or lowest iodize
    def _selection_tournement(self, win_rate=0.75, tournement_size=10):
        new_population = list()
        for tournement in range(self._population_size):
            # Create tournement candidates
            idx_candidates = [i for i in range(self._population_size)]
            random.shuffle(idx_candidates)
            tournament_candidates = list()
            for candidate_idx in idx_candidates:
                tournament_candidates.append(self._population[candidate_idx])
            #

            best_candidate = None
            worst_candidate = None
            for candidate in tournament_candidates:
                # For best
                if best_candidate is None or best_candidate.get_fitness() < candidate.get_fitness():
                    best_candidate = candidate
                # For badest
                if worst_candidate is None or worst_candidate.get_fitness() > candidate.get_fitness():
                    worst_candidate = candidate
                if random.random() <= win_rate:
                    new_population.append(best_candidate)
                else:
                    new_population.append(worst_candidate)
        self._population = new_population
