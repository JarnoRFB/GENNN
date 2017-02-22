from ga import GA
from candidate_nn import CandidateNN
from candidate_ga import Candidate_GA
from utils import RangedNum
import time
import os
#Graphen gerniene

genetic_hyperparamter = {
    'population': 10,
    'calc_diversity': False,
    'crossover': {
        'strategy': 'uniform_crossover',
        'uniform_method': 'swap',
        'rate': 0.5
    },
    'mutation': {
        'rate': 0.1,

    },
    'candidate_class': Candidate_GA,
    'number_of_generation': 2,
    'selection_strategy': {
        'type': 'Tournament',
        'best_win_rate': 0.8,
        'tournament_size': 3
    },
    'RUNTIME_SPEC': {'id': 1,
                     'datadir': 'MNIST_data',
                     'logdir': 'log_opti/',
                     'validate_each_n_steps': 10,
                     'max_number_of_iterations': 600,
                     'max_runtime': 10,
                     'max_layer': 1,
                     'fitness_strategy': 'accuracy',
                     'fitness_power': 3}

}
gen = GA(genetic_hyperparamter)

for i in range(genetic_hyperparamter['number_of_generation']):
    gen.mutate()
    gen.crossover(strategy=genetic_hyperparamter['crossover'])
    gen.evaluate(calc_diversity=genetic_hyperparamter['calc_diversity'])
    gen.selection()

    print("Gen: " + str(gen.generation) + "- Fitness_avg: " + str(round(gen.fitness_avg, 3)) + "- Fitness_best: " + str(
        round(gen.best_candidate.get_fitness(), 3)))
gen.write_stats()
