from candidate_nn import CandidateNN
import os
# clear logging dir
os.system('rm -R /tmp/gennn/')

runtime_spec =  {'id': 1,
            'datadir': 'dir',
            'logdir': '/tmp/gennn/test/',
            'validate_each_n_steps': 10,
            'max_number_of_iterations': 200,
            'max_runtime': 10,}

c = CandidateNN(runtime_spec)
print(c.network_spec)
print('\n')
print(c.get_fitness())
