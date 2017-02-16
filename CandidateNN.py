import json
import random
class CandidateNN:
    def __init__(self,network_json):
        self.data = json.loads(network_json)
    def crossover(self,other_candidate,strategy="onePointSwap"):
        if(strategy == "onePointSwap"):
            self._crossing_one_point_swap(other_candidate)
        #Crossing parameters
    def mutation(self,mutation_rate):
        #Make mutations
        print("mutation")

    def evaluate(self):
        #Start Calculation
        print("evaluate")

    def get_diversity(self,otherCandidate):
        #from 0 to 1
        print("get_div")
        return random.random()
    def _crossing_one_point_swap(self,other_candidate):
        print("_crossing_one_point_swap")

    def get_fitness(self):
        print("get_fitness")
        return random.random()

def create_random_CandidateNN():
    print("create_random_CandidateNN")
    fd = open("network_spec.json")
    return CandidateNN(fd.read())