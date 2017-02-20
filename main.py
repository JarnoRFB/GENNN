from ga import GA
from candidate_nn import CandidateNN

gen = GA(2,0.7,0.8,CandidateNN)

for i in range(10):
    gen.mutate()
    gen.crossover(strategy='uniform_crossover')
    gen.evaluate(calc_diversity=False)
    gen.selection()
    print("Gen: "+str(gen.generation)+"- Fitness_avg: "+str(gen.fitness_avg)+"- Fitness_best: "+str(gen.best_candidate.get_fitness()))