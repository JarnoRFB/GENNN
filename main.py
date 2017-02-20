from ga import GA
from candidate_nn import CandidateNN

genetic_hyperparamter = {
    'base_log_dir' : 'log',
    'population' : 2,
    'calc_diversity' : True,
    'crossover' : {
        'strategy' : 'uniform_crossover',
        'uniform_method' : 'swap',
        'rate' : 0.7
    },
    'mutation' : {
        'rate' : 0.8,

    },
    'candidate_class' : CandidateNN,
    'number_of_generation' : 10,
    'selection_strategy' : {
        'type' : 'Tournament',
        'best_win_rate' : 0.75,
        'tournament_size' : 10
    },
    'RUNTIME_SPEC' : {'id': 1,
                    'datadir': 'MNIST_data',
                    'logdir': 'log/',
                    'validate_each_n_steps': 10,
                    'max_number_of_iterations': 200,
                    'max_runtime': 10}

}
gen = GA(genetic_hyperparamter)

for i in range(genetic_hyperparamter['number_of_generation']):
    gen.mutate()
    gen.crossover(strategy=genetic_hyperparamter['crossover'])
    gen.evaluate(calc_diversity=genetic_hyperparamter['calc_diversity'])
    gen.selection()

    print("Gen: "+str(gen.generation)+"- Fitness_avg: "+str(gen.fitness_avg)+"- Fitness_best: "+str(gen.best_candidate.get_fitness()))