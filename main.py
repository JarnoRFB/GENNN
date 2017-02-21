from ga import GA
from candidate_nn import CandidateNN
from utils import RangedNum
import time
import os
import traceback
while(True):

    try:
        genetic_hyperparamter = {
            'population': 10,
            'calc_diversity': True,
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
                            'max_runtime': 10,
                            'max_layer': 3}

        }
        gen = GA(genetic_hyperparamter)

        for i in range(genetic_hyperparamter['number_of_generation']):
            gen.mutate()
            gen.crossover(strategy=genetic_hyperparamter['crossover'])
            gen.evaluate(calc_diversity=genetic_hyperparamter['calc_diversity'])
            gen.selection()

            print("Gen: "+str(gen.generation)+"- Fitness_avg: "+str(round(gen.fitness_avg,3))+"- Fitness_best: "+str(round(gen.best_candidate.get_fitness(),3)))
        gen.write_stats()
    except Exception as e:
        print(e)
        os.makedirs(genetic_hyperparamter['RUNTIME_SPEC']['logdir'], exist_ok=True)
        with open(os.path.join(genetic_hyperparamter['RUNTIME_SPEC']['logdir'], 'error.log'), mode='a') as fp:
            fp.write(time.strftime("%Y.%m.%d-%H.%M.%S", time.gmtime()) + ' error: ' + str(traceback.format_exc()) + '\n')
        continue