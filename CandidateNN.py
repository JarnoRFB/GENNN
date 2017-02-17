import json
import random
class CandidateNN:
    _fitness = None

    def __init__(self, network_json):
        self.data = json.loads(network_json)

    def crossover(self, other_candidate, strategy="onePointSwap"):
        self._fitness = None
        if(strategy == "onePointSwap"):
            self._crossing_one_point_swap(other_candidate)

    def mutation(self, mutation_rate):
        self._fitness = None
        #print("mutation")

    def get_diversity(self, otherCandidate):
        #print("get_div")
        return random.random()

    def get_fitness(self):
        if(self._fitness is None):
            self._fitness = random.random()
            #print("get_fitness")
        return self._fitness

    def _crossing_one_point_swap(self, other_candidate):
        print("")

def create_random_CandidateNN():
    fd = open("network_spec.json")
    return CandidateNN(fd.read())