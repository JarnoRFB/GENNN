from ga import GA
from candidate_nn import CandidateNN
from utils import RangedNum
import time
import os

while(True):

    try:
        genetic_hyperparamter = {
            'base_log_dir': 'log/',
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
            'number_of_generation': 20,
            'selection_strategy': {
                'type': 'Tournament',
                'best_win_rate': RangedNum(0.1, 0.9).value,
                'tournament_size': 10
            },
            'RUNTIME_SPEC': {'id': 1,
                            'datadir': 'MNIST_data',
                            'logdir': 'log/',
                            'validate_each_n_steps': 10,
                            'max_number_of_iterations': 600,
                            'max_runtime': 10}

        }
        gen = GA(genetic_hyperparamter)

        for i in range(genetic_hyperparamter['number_of_generation']):
            gen.mutate()
            gen.crossover(strategy=genetic_hyperparamter['crossover'])
            gen.evaluate(calc_diversity=genetic_hyperparamter['calc_diversity'])
            gen.selection()

            print("Gen: "+str(gen.generation)+"- Fitness_avg: "+str(gen.fitness_avg)+"- Fitness_best: "+str(gen.best_candidate.get_fitness()))

    except Exception as e:
        print(e)
        os.makedirs(genetic_hyperparamter['base_log_dir'], exist_ok=True)
        with open(os.path.join(genetic_hyperparamter['base_log_dir'], 'error.log'), mode='a') as fp:
            fp.write(time.strftime("%Y.%m.%d-%H.%M.%S", time.gmtime()) + ' error: ' + str(e) + '\n')
        continue