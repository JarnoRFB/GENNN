from candidate_nn import CandidateNN
import os
# clear logging dir
os.system('rm -R /tmp/gennn/')


c = CandidateNN(1,"a",
                runtime_spec={'id': 1,
                     'datadir': 'MNIST_data',
                     'logdir': 'log/',
                     'validate_each_n_steps': 10,
                     'max_number_of_iterations': 600,
                     'max_runtime': 10,
                     'max_layer': 1})
c2 = CandidateNN(2,"a",
                runtime_spec={'id': 1,
                     'datadir': 'MNIST_data',
                     'logdir': 'log/',
                     'validate_each_n_steps': 10,
                     'max_number_of_iterations': 600,
                     'max_runtime': 10,
                     'max_layer': 3})
print(c.network_spec['layers'])
print(c2.network_spec['layers'])

print('\n')
c.crossover( crossover_parms={
        'strategy': 'uniform_crossover',
        'uniform_method': 'swap',
        'rate': 0.7
    },other_candidate=c2)
print(c.network_spec['layers'])
print(c2.network_spec['layers'])

