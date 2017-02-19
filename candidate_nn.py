import json
import random
from utils import RangedNum, RangedInt, RangedJSONEncoder
from builder.network_builder import Network
import os
import math


class CandidateNN:

    # ---------------------- Static class attributes ----------------------

    RUNTIME_SPEC = {'id': 1,
                    'datadir': 'dir',
                    'logdir': '/tmp/gennn/',
                    'validate_each_n_steps': 100,
                    'max_number_of_iterations': 200,
                    'max_runtime': 10}

    OPTIMIZER_CHOICES = ('AdamOptimizer', 'AdadeltaOptimizer', 'AdagradOptimizer',
                         'FtrlOptimizer', 'ProximalGradientDescentOptimizer', 'ProximalAdagradOptimizer',
                         'RMSPropOptimizer', 'GradientDescentOptimizer')
    ACTIVATION_CHOICES = ('relu', 'relu6', 'sigmoid', 'tanh', 'crelu', 'elu', 'softplus', 'softsign')
    LAYER_TYPES = ("conv_layer", "maxpool_layer", "feedforward_layer")
    ACCURACY_WEIGHT = 20
    LAYER_CNT_WEIGHT = 2
    MAX_LAYERS = 3

    def __init__(self, network_spec=None, generation=0):

        self.generation = generation
        self._fitness = None
        if network_spec is None:
            network_spec = self._create_random_network()
        self.network_spec = network_spec

    def crossover(self, other_candidate, strategy='onePointSwap'):
        self._fitness = None
        if strategy == 'onePointSwap':
            self._crossing_one_point_swap(other_candidate)

    def mutation(self, mutation_rate):
        """
        Mutate properties(layer-structure and layer-values of a Candidate)
        """
        self.generation += 1
        self._fitness = None

        #Mutate layer
        for i, layer_spec in enumerate(self.network_spec['layers']):
            #Mutate complet layer
            if(random.uniform(0,1)<=mutation_rate):
                new_layer_type = random.choice(self.LAYER_TYPES)
                self.network_spec['layers'][i] = self._create_randomize_layer(layer_type=new_layer_type)
            else:#Only mutate Values if no new random layer
                self._mutate_layer_values(layer_dict=self.network_spec['layers'][i], mutation_rate=mutation_rate)

    def _mutate_layer_values(self, layer_dict, mutation_rate):
        """
        Mutate each value of a layer with a probability of mutation_rate
        """
        if(random.uniform(0,1) <= mutation_rate):
            layer_dict['activation_function'] = random.choice(self.ACTIVATION_CHOICES)

        if(layer_dict['type']=='conv_layer'):
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['convolution']['filter']['height'] = RangedInt(1, 5)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['convolution']['filter']['width'] = RangedInt(1, 5)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['convolution']['filter']['outchannels'] = RangedInt(1, 64)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['convolution']['strides']['x'] = RangedInt(1, 2)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['convolution']['strides']['y'] = RangedInt(1, 2)

        elif layer_dict['type'] == 'maxpool_layer':
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['kernel']['height'] = RangedInt(1, 5)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['kernel']['width'] = RangedInt(1, 5)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['strides']['x'] = RangedInt(1, 5)
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['strides']['y'] = RangedInt(1, 5)

        elif layer_dict['type'] == 'feedforward_layer':
            if (random.uniform(0, 1) <= mutation_rate):
                layer_dict['size'] = RangedInt(256, 2048)

    def get_diversity(self, otherCandidate):
        #print('get_div')
        return random.random()

    def get_fitness(self):
        """Get fitness of the candidate. If not yet tested, test the fitness based on the network specificaton."""
        if(self._fitness is None):
            network = Network(self._serialze_network_spec())
            extended_spec_json = network.evaluate()
            extended_spec = json.loads(extended_spec_json)
            result_spec = extended_spec['results']
            print(result_spec)
            self._fitness = self._fitness_function(result_spec)

        return self._fitness

    def _fitness_function(self, results):
        """Calculate the fitness based on the network evaluation."""
        # TODO: get the number of weights as penalty?
        return  1 / (- self.ACCURACY_WEIGHT * math.log(results['accuracy'])
                     + self.LAYER_CNT_WEIGHT * len(self.network_spec['layers']))

    def _crossing_one_point_swap(self, other_candidate):
        print('')

    def _create_random_network(self):
        """Construct a random network specification."""

        #TODO: should this be done in this class?
        # Finalize runtime specification.
        generation_dir = 'generation_{}/'.format(self.generation)
        self.RUNTIME_SPEC['logdir'] = os.path.join(self.RUNTIME_SPEC['logdir'], generation_dir, str(self.RUNTIME_SPEC['logdir']))

        layer_cnt = RangedInt(1, self.MAX_LAYERS)

        network_spec = {
            'hyperparameters': {
                'learningrate': RangedNum(1e-4, 1e-3),
                'optimizer': random.choice(self.OPTIMIZER_CHOICES),
                'batchsize': 50  # Fixed batch size for comparability.
            },
            'layers': []
        }

        cnt_layer_conv = RangedInt(0, layer_cnt.value)
        cnt_layer_max_pool = RangedInt(0, layer_cnt.value - cnt_layer_conv.value)
        cnt_layer_ff = layer_cnt.value - cnt_layer_conv.value - cnt_layer_max_pool.value

        layer_types = ['conv_layer' for _ in range(cnt_layer_conv.value)]
        layer_types += ['maxpool_layer' for _ in range(cnt_layer_max_pool.value)]
        random.shuffle(layer_types)
        layer_types += ['feedforward_layer' for _ in range(cnt_layer_ff)]

        for layer_type in layer_types:
            layer_spec = self._create_randomize_layer(layer_type=layer_type)
            # Add layer to the network spec.
            network_spec['layers'].append(layer_spec)

        network_spec.update(self.RUNTIME_SPEC)

        return network_spec

    def _create_randomize_layer(self, layer_type):
        """
        Create a layer based on layer_tape
        """
        if layer_type == 'conv_layer':
            layer_spec = self._create_conv_layer()
        elif layer_type == 'maxpool_layer':
            layer_spec = self._create_max_layer()
        elif layer_type == 'feedforward_layer':
            layer_spec = self._create_ff_layer()
        return layer_spec

    def _create_ff_layer(self):
        """
        Create dict for a random initialized Feedforward network
        :return:
        """

        layer = {
                  'type': 'feedforward_layer',
                  'size': RangedInt(256, 2048),
                  'activation_function': random.choice(self.ACTIVATION_CHOICES)
                }
        return layer

    def _create_conv_layer(self):
        """
        Create dict for a random initialized convolutional Layer
        """
        layer = {
            'type': 'conv_layer',
            'convolution': {
                'filter': {
                    'height': RangedInt(1, 5),
                    'width': RangedInt(1, 5),
                    'outchannels': RangedInt(1, 64)
                },
                'strides': {
                    'x': RangedInt(1, 2),
                    'y': RangedInt(1, 2),
                    'inchannels': 1,  # Must be 1. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                    'batch': 1
                }

            },
            'activation_function': random.choice(self.ACTIVATION_CHOICES)
        }
        return layer

    def _create_max_layer(self):
        """
        Create dict for a random initialized Maxpool-layer
        """
        layer = {
            'type': 'maxpool_layer',
            'kernel': {
                'height': RangedInt(1, 5),
                'width': RangedInt(1, 5),
                'inchannels': 1,
            # Must probably be 1 as well. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                'outchannels': 1,
            },
            'strides': {
                'y': RangedInt(1, 5),
                'x': RangedInt(1, 5),
                'inchannels': 1,
            # Must probably be 1 as well. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                'batch': 1
            }
        }
        return layer

    def _serialze_network_spec(self):

        return RangedJSONEncoder().encode(self.network_spec)
