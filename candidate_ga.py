from utils import RangedInt, RangedNum
from candidate_nn import CandidateNN
import random
from ga import GA
from copy import deepcopy
import os

class Candidate_GA:

    def __init__(self, candidate_id, start_time_str, runtime_spec, network_spec=None):
        self._candidate_id = candidate_id
        self._base_logdir = os.path.join(self.runtime_spec['logdir'], str(start_time_str))

        self._problem_class = runtime_spec
        if network_spec is None:
            self.network_spec = self._generate_random()
        self._fitness = None
        self._best_candidate = None
        self._best_candidate_forever = None
        self._diversity = None


    def crossover(self, crossover_parms, other_candidate):
        self._fitness = None
        self._diversity = None
        if crossover_parms['strategy'] == 'uniform_crossover':
            self._crossover_uniform(crossover_rate=crossover_parms['rate'],
                                    other_candidate=other_candidate,
                                    uniform_method=crossover_parms['uniform_method'])
        else:
            raise ValueError('not implemented crossover strategy')

    def _crossover_uniform(self, crossover_rate, other_candidate, uniform_method):
        if random.uniform(0, 1) <= crossover_rate:
            self.network_spec['crossover']['rate'] = other_candidate._hyper_parms['crossover']['rate']

        if random.uniform(0, 1) <= crossover_rate:
            self.network_spec['mutation']['rate'] = other_candidate._hyper_parms['mutation']['rate']

        if random.uniform(0, 1) <= crossover_rate:
            self.network_spec['selection_strategy']['best_win_rate'] = \
            other_candidate._hyper_parms['selection_strategy']['best_win_rate']

        if random.uniform(0, 1) <= crossover_rate:
            self.network_spec['selection_strategy']['tournament_size'] = \
            other_candidate._hyper_parms['selection_strategy']['tournament_size']

    def mutation(self, mutation_rate):
        self._fitness = None
        self._diversity = None

        if random.uniform(0, 1) <= mutation_rate:
            self.network_spec['crossover']['rate'] = random.uniform(0.1, 0.9)

        if random.uniform(0, 1) <= mutation_rate:
            self.network_spec['mutation']['rate'] = random.uniform(0.01, 0.5)

        if random.uniform(0, 1) <= mutation_rate:
            self.network_spec['selection_strategy']['best_win_rate'] = random.uniform(0.5, 0.9)

        if random.uniform(0, 1) <= mutation_rate:
            self.network_spec['selection_strategy']['tournament_size'] = random.randint(1, 10)

    def get_diversity(self, other_candidate):
        if self._diversity is None:
            self._diversity = 0
            if self.network_spec['crossover']['rate'] == other_candidate._hyper_parms['crossover']['rate']:
                self._diversity += 1

            if self.network_spec['mutation']['rate'] == other_candidate._hyper_parms['mutation']['rate']:
                self._diversity += 1

            if self.network_spec['selection_strategy']['best_win_rate'] == \
                other_candidate._hyper_parms['selection_strategy']['best_win_rate']:
                self._diversity += 1

            if self.network_spec['selection_strategy']['tournament_size'] == \
                other_candidate._hyper_parms['selection_strategy']['tournament_size']:
                self._diversity += 1
            self._diversity /= 4.0
        return self._diversity

    def get_fitness(self):
        if self._fitness is None:
            gen = GA(self.network_spec)
            for i in range(self.network_spec['number_of_generation']):
                gen.mutate()
                gen.crossover(strategy=self.network_spec['crossover'])
                gen.evaluate(calc_diversity=self.network_spec['calc_diversity'])
                gen.selection()
                if gen.best_candidate_forever.get_fitness() >= self._best_candidate.get_fitness():
                    self._best_candidate = deepcopy(gen.best_candidate_forever)
                print("-Gen: "+str(gen.generation)+"-Diversity: "+str(round(gen.diversity))+"- Fitness_avg: "+str(round(gen.fitness_avg,3))+"- Fitness_best: "+str(round(gen.best_candidate.get_fitness(),3)))
                self._fitness = 0.6*gen.best_candidate + 0.4*gen.fitness_avg

        file_loc = os.path.join(self.network_spec['RUNTIME_SPEC']['logdir'], "ga.json")
        with open(file_loc, 'w') as fp:
            fp.write(str(self._parms))
        return self._fitness

    def to_next_generation(self, generation):
        self.outer_gen = generation
        generation_dir = 'generation_{}/'.format(generation)
        id_dir = '{}/'.format(self._candidate_id)
        self.runtime_spec['logdir'] = os.path.join(self._base_logdir, generation_dir, id_dir)
        self.network_spec.update(self.runtime_spec)

    outer_gen = 0
    def _generate_random(self):
        hyper_parms = genetic_hyperparamter = {
            'population': 10,
            'calc_diversity': False,
            'crossover': {
                'strategy': 'uniform_crossover',
                'uniform_method': 'swap',
                'rate': RangedNum(0.1, 0.9).value
            },
            'mutation': {
                'rate': RangedNum(0.01, 0.5).value,

            },
            'candidate_class': CandidateNN,
            'number_of_generation': 10,
            'selection_strategy': {
                'type': 'Tournament',
                'best_win_rate': RangedNum(0.5, 0.9).value,
                'tournament_size': RangedInt(1, 10).value
            },
            'RUNTIME_SPEC': {'id': 1,
                            'datadir': 'MNIST_data',
                            'logdir': 'log/'+str(self.outer_gen)+'/',
                            'validate_each_n_steps': 10,
                            'max_number_of_iterations': 600,
                            'max_runtime': 10,
                            'max_layer': 3,
                            'fitness_strategy': 'accuracy',
                            'fitness_power': 3}

        }
        self.outer_gen += 1
        return hyper_parms

