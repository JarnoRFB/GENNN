from candidate_nn import CandidateNN
import os
# clear logging dir
os.system('rm -R /tmp/gennn/')


c = CandidateNN()
print(c.network_spec)
print('\n')
print(c.get_fitness())
