import json
import random
from utils import RangedInt

class CandidateNN:


    self._fitness = None

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
#can statik
def create_random_CandidateNN(runtime_spec):

    layer_cnt = RangedInt(1,6)

    for layer in range(layer_cnt.value):
        network_spec = {
          "id": runtime_spec["id"],
          "datadir": runtime_spec["dir"],
          "logdir": runtime_spec["logdir"],
          "validate_each_n_steps": runtime_spec[100],
          "max_number_of_iterations": runtime_spec[200],
          "max_runtime": runtime_spec[10],
          "hyperparameters":
          {
            "learningrate": 1e-4,
            "optimizer": "AdamOptimizer",
            "batchsize": RangedInt(20,80)
          },
        }


        cnt_layer_conv = RangedInt(0, layer_cnt.value)
        cnt_layer_max_pool = RangedInt(0, layer_cnt.value - cnt_layer_conv.value)
        cnt_layer_ff = layer_cnt.value - cnt_layer_conv.value - cnt_layer_max_pool.value

        layer_types = ["conv_layer" for x in range(cnt_layer_conv.value)]
        layer_types += ["maxpool_layer" for x in range(cnt_layer_max_pool.value)]
        random.shuffle(layer_types)
        layer_types += ["feedforward_layer" for x in range(cnt_layer_ff)]

        activation_fnc = ["relu", "relu6", "sigmoid", "tanh", "crelu"]

        for layer_type in range(layer_types):
            if layer_type == "conv_layer":
                layer = {
                    "type": "conv_layer",
                    "convolution": {
                        "filter": {
                            "height": RangedInt(1, 5),
                            "width": RangedInt(1, 5),
                            "outchannels": RangedInt(1, 64)
                        },
                        "strides": {
                            "x": RangedInt(1, 2),
                            "y": RangedInt(1, 2),
                            "batch": 1
                        }

                    },
                    "activation_function": activation_fnc[random.randint(0,len(activation_fnc))]
                }
            elif layer_type == "maxpool_layer":
                layer ={
                  "type": "maxpool_layer",
                  "kernel":
                  {
                    "height": RangedInt(1, 5),
                    "width": RangedInt(1, 5),
                    "outchannels": 1
                  },
                  "strides":
                  {
                    "y": RangedInt(1, 5),
                    "x": RangedInt(1, 5),
                    "batch": 1
                  }
                }
            elif layer_type == "feedforward_layer":
                layer ={
                  "type": "feedforward_layer",
                  "size": RangedInt(256, 2048),
                  "activation_function": activation_fnc[random.randint(0,len(activation_fnc))]
                }
    }


    return CandidateNN()
