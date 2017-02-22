from utils import RangedInt, RangedNum
from candidate_nn import CandidateNN
import random
from ga import GA

class candidate_ga:

    def __init__(self, candidate_id, start_time_str, runtime_spec, network_spec=None):
        self._problem_class = runtime_spec
        if network_spec is None:
            self._hyper_parms = self._generate_random()


    def crossover(self,crossover_parms, other_candidate):
        self._fitness = None

        if crossover_parms['strategy'] == 'uniform_crossover':
            self._crossover_uniform(crossover_rate=crossover_parms['rate'],
                                    other_candidate=other_candidate,
                                    uniform_method=crossover_parms['uniform_method'])
        else:
            raise ValueError('not implemented crossover strategy')
    def _crossover_uniform(self, crossover_rate, other_candidate, uniform_method):
        if random.uniform(0, 1) <= crossover_rate:
            self._hyper_parms['crossover']['rate'] = other_candidate._hyper_parms['crossover']['rate']

        if random.uniform(0, 1) <= crossover_rate:
            self._hyper_parms['mutation']['rate'] = other_candidate._hyper_parms['mutation']['rate']

        if random.uniform(0, 1) <= crossover_rate:
            self._hyper_parms['selection_strategy']['best_win_rate'] = \
            other_candidate._hyper_parms['selection_strategy']['best_win_rate']

        if random.uniform(0, 1) <= crossover_rate:
            self._hyper_parms['selection_strategy']['tournament_size'] = \
            other_candidate._hyper_parms['selection_strategy']['tournament_size']

    def mutation(self, mutation_rate):

        if random.uniform(0, 1) <= mutation_rate:
            self._hyper_parms['crossover']['rate'] = random.uniform(0.1,0.9)

        if random.uniform(0, 1) <= mutation_rate:
            self._hyper_parms['mutation']['rate'] = random.uniform(0.01,0.5)

        if random.uniform(0, 1) <= mutation_rate:
            self._hyper_parms._hyper_parms['selection_strategy']['best_win_rate'] =random.uniform(0.5, 0.9)

        if random.uniform(0, 1) <= mutation_rate:
            self._hyper_parms['selection_strategy']['tournament_size'] = random.randint(1,10)

    def get_diversity(self, other_candidate):

        div = 0
        if self._hyper_parms['crossover']['rate'] == other_candidate._hyper_parms['crossover']['rate']:
            div += 1

        if self._hyper_parms['mutation']['rate'] == other_candidate._hyper_parms['mutation']['rate']:
            div += 1

        if self._hyper_parms['selection_strategy']['best_win_rate'] == \
            other_candidate._hyper_parms['selection_strategy']['best_win_rate']:
            div += 1

        if self._hyper_parms['selection_strategy']['tournament_size'] == \
            other_candidate._hyper_parms['selection_strategy']['tournament_size']:
            div += 1
        return div /4.0

    def get_fitness(self):
        gen = GA(self._hyper_parms)
        for i in range(self._hyper_parms['number_of_generation']):
            gen.mutate()
            gen.crossover(strategy=self._hyper_parms['crossover'])
            gen.evaluate(calc_diversity=self._hyper_parms['calc_diversity'])
            gen.selection()

            print("Gen: "+str(gen.generation)+"-Diversity: "+str(round(gen.diversity))+"- Fitness_avg: "+str(round(gen.fitness_avg,3))+"- Fitness_best: "+str(round(gen.best_candidate.get_fitness(),3)))


    def to_next_generation(self, generation):
        print ("a")

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
            'number_of_generation': 15,
            'selection_strategy': {
                'type': 'Tournament',
                'best_win_rate': RangedNum(0.5, 0.9).value,
                'tournament_size': RangedInt(1, 10).value
            },
            'RUNTIME_SPEC': {'id': 1,
                            'datadir': 'MNIST_data',
                            'logdir': 'log/',
                            'validate_each_n_steps': 10,
                            'max_number_of_iterations': 600,
                            'max_runtime': 10,
                            'max_layer': 3,
                            'fitness_strategy': 'accuracy',
                            'fitness_power': 3}

        }
        return hyper_parms

