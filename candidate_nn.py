import json
import random
from utils import RangedNum, RangedInt, RangedJSONEncoder
from builder.network_builder import Network
import os
import math
import copy


class CandidateNN:

    # ---------------------- Static class attributes ----------------------

    OPTIMIZER_CHOICES = ('AdamOptimizer', 'AdadeltaOptimizer', 'AdagradOptimizer',
                         'FtrlOptimizer', 'ProximalGradientDescentOptimizer', 'ProximalAdagradOptimizer',
                         'RMSPropOptimizer', 'GradientDescentOptimizer')
    ACTIVATION_CHOICES = ('relu', 'relu6', 'sigmoid', 'tanh', 'elu', 'softplus', 'softsign')
    LAYER_TYPES = ("conv_layer", "maxpool_layer", "feedforward_layer")
    OPTIMIZING_PARMS = {
        'conv_layer':
        [
            {'parms':
                {'hierarchi': ['convolution', 'filter', 'height'],
                 'min': 1,
                 'max': 5,
                 'type': 'int'}
            },
            {'parms':
                {'hierarchi': ['convolution', 'filter', 'width'],
                 'min': 1,
                 'max': 5,
                 'type': 'int'}
            },
            {'parms':
                {'hierarchi': ['convolution', 'filter', 'outchannels'],
                 'min': 1,
                 'max': 64,
                 'type': 'int'}
            },
            {'parms':
                 {'hierarchi': ['convolution', 'strides', 'x'],
                  'min': 1,
                  'max': 2,
                  'type': 'int'}
            },
            {'parms':
                 {'hierarchi': ['convolution', 'strides', 'y'],
                  'min': 1,
                  'max': 2,
                  'type': 'int'}
             },
            {'parms':
                 {'hierarchi': ['convolution', 'strides', 'inchannels'],
                  'min': 1,
                  'max': 1,
                  'type': 'int'}
             },
            {'parms':
                 {'hierarchi': ['convolution', 'strides', 'batch'],
                  'min': 1,
                  'max': 1,
                  'type': 'int'}
             },
        ],
        'maxpool_layer':
         [
             {'parms':
                  {'hierarchi': ['kernel', 'height'],
                   'min': 1,
                   'max': 5,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['kernel', 'width'],
                   'min': 1,
                   'max': 5,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['kernel', 'inchannels'],
                   'min': 1,
                   'max': 1,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['kernel', 'outchannels'],
                   'min': 1,
                   'max': 1,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['strides', 'x'],
                   'min': 1,
                   'max': 5,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['strides', 'y'],
                   'min': 1,
                   'max': 5,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['strides', 'inchannels'],
                   'min': 1,
                   'max': 1,
                   'type': 'int'}
              },
             {'parms':
                  {'hierarchi': ['strides', 'batch'],
                   'min': 1,
                   'max': 1,
                   'type': 'int'}
              }
         ],
        'feedforward_layer':
        [
            {'parms':
                 {'hierarchi': ['size'],
                  'min': 256,
                  'max': 2048,
                  'type': 'int'}
             }
        ]
    }

    ACCURACY_WEIGHT = 20
    LAYER_CNT_WEIGHT = 2
    MAX_LAYERS = 3

    def __init__(self, candidate_id, start_time_str, runtime_spec, network_spec=None ):
        self.runtime_spec = copy.deepcopy(runtime_spec)

        self._base_logdir = os.path.join(self.runtime_spec['logdir'], str(start_time_str))

        self._candidate_id = candidate_id
        self._fitness = None
        if network_spec is None:
            network_spec = self._create_random_network()
        self.network_spec = network_spec

    def to_next_generation(self, generation):

        generation_dir = 'generation_{}/'.format(generation)
        id_dir = '{}/'.format(self._candidate_id)
        self.runtime_spec['logdir'] = os.path.join(self._base_logdir, generation_dir, id_dir)
        self.network_spec.update(self.runtime_spec)


    def crossover(self, crossover_parms, other_candidate):
        self._fitness = None

        if crossover_parms['strategy'] == 'uniform_crossover':
            self._crossover_uniform(crossver_rate=crossover_parms['rate'],
                                    other_candidate= other_candidate,
                                    uniform_method=crossover_parms['uniform_method'])
        else:
            raise ValueError('not implemented crossover strategy')



    def _crossover_uniform(self, crossver_rate, other_candidate, uniform_method):
        """Performs a unifrom Crossover between two Candidates"""
        if(uniform_method == 'swap'):
            min_layers = min(len(self.network_spec['layers']),len(other_candidate.network_spec['layers']))
            for layer_idx, layer in enumerate(self.network_spec['layers'][:min_layers]):
                layer_dict = layer
                other_layer_dict = other_candidate.network_spec['layers'][layer_idx]

                if (random.uniform(0, 1) <= crossver_rate):
                    layer_dict['activation_function'] = random.choice(self.ACTIVATION_CHOICES)

                if(layer_dict['type'] == other_layer_dict['type']):
                    self._swap_values(layer_dict,other_layer_dict, crossver_rate)


        else:
            raise ValueError('not implemented uniform_crossover_method')
    def _swap_values(self, dict, other_dict,rate):
        """Swaps Properties between two Layers of the same type with Propapility rate"""
        for parm in self.OPTIMIZING_PARMS[dict['type']]:
            if random.uniform(0,1)<=rate:
                parm_h = parm['parms']['hierarchi']
                if len(parm_h) == 1:
                    # Save old own
                    tmp = dict[parm_h[0]]
                    # own in other
                    dict[parm_h[0]] = \
                        other_dict[parm_h[0]]
                    # saved in own
                    other_dict[parm_h[0]] = tmp
                elif len(parm_h) == 2:
                    # Save old own
                    tmp = dict[parm_h[0]][parm_h[1]]
                    # own in other
                    dict[parm_h[0]][parm_h[1]] = \
                        other_dict[parm_h[0]][parm_h[1]]
                    # saved in own
                    other_dict[parm_h[0]][parm_h[1]] = tmp
                elif len(parm_h) == 3:
                    # Save old own
                    tmp = dict[parm_h[0]][parm_h[1]][parm_h[2]]
                    # own in other
                    dict[parm_h[0]][parm_h[1]][parm_h[2]] = \
                        other_dict[parm_h[0]][parm_h[1]][parm_h[2]]
                    # saved in own
                    other_dict[parm_h[0]][parm_h[1]][parm_h[2]] = tmp
                else:
                    raise ValueError('length of hierarchi must 1,2 or 3')

    def mutation(self, mutation_rate):
        # TODO: Check the mutation of a layer and the mutation of properties, layer mutation can hide value mutation
        """
        Mutate properties(layer-structure and layer-values of a Candidate)

        """
        self._fitness = None

        #Mutate layer
        for i, layer_spec in enumerate(self.network_spec['layers']):
            #Mutate complet layer
            if(random.uniform(0,1)<=(mutation_rate/10)):
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
        for parms in self.OPTIMIZING_PARMS[layer_dict['type']]:
            if parms['parms']['max'] == parms['parms']['min']:
                break
            parm_h = parms['parms']['hierarchi']
            variance = (parms['parms']['max'] - parms['parms']['min']) / 2
            if variance == 0 :
                variance = 1
            if parms['parms']['type'] == 'int':
                variance = int(variance)

            if len(parm_h) == 1:
                layer_dict[parm_h[0]] = self._mutation_value_strategy(
                    old_value=layer_dict[parm_h[0]],
                    variance=variance)
            elif len(parm_h) == 2:
                layer_dict[parm_h[0]][parm_h[1]] = self._mutation_value_strategy(
                                                    old_value=layer_dict[parm_h[0]][parm_h[1]],
                                                    variance=variance)
            elif len(parm_h) == 3:
                layer_dict[parm_h[0]][parm_h[1]][parm_h[2]] = self._mutation_value_strategy(
                                                        old_value=layer_dict[parm_h[0]][parm_h[1]][parm_h[2]],
                                                        variance=variance)
            else:

                raise ValueError('length of hierarchi must 1,2 or 3')


    def _mutation_value_strategy(self, old_value, variance):
        """ sub/add a number between -variance and variance"""
        return old_value + old_value.__class__(-variance, variance).value

    def get_diversity(self, otherCandidate):

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
            del network
        return self._fitness

    def _fitness_function(self, results):
        """Calculate the fitness based on the network evaluation."""
        # TODO: get the number of weights as penalty?
        return  1 / (- self.ACCURACY_WEIGHT * math.log(results['accuracy'])
                     + self.LAYER_CNT_WEIGHT * len(self.network_spec['layers']))

    def _create_random_network(self):
        """Construct a random network specification."""

        #TODO: should this be done in this class?
        # Finalize runtime specification.

        layer_cnt = RangedInt(1, self.MAX_LAYERS)

        network_spec = {
            'hyperparameters': {
                'learningrate': RangedNum(1e-4, 1e-3),
                'optimizer': random.choice(self.OPTIMIZER_CHOICES),
                'batchsize': 100  # Fixed batch size for comparability.
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
            #layer_spec = self._generate_network_layer(type=layer_type)
            # Add layer to the network spec.
            network_spec['layers'].append(layer_spec)

        return network_spec

    def _create_randomize_layer(self, layer_type):
        """
        Create a layer based on layer_tape
        """
        if layer_type == 'conv_layer':
            layer_spec = self._create_conv_layer()
        elif layer_type == 'maxpool_layer':
            layer_spec = self._create_maxpool_layer()
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

    def _create_maxpool_layer(self):
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
    def _generate_network_layer(self, type):
        # TODO: Implement this!
        raise Exception("Not implemented")
        for parms in self.OPTIMIZING_PARMS[type]:
            parm = parms['parms']
            parm_h = parm['hierarchi']
            layer = dict()
            layer['type'] = type
            layer['activation_function'] = random.choice(self.ACTIVATION_CHOICES)
            if len(parm_h) == 1:
                layer[parm_h[0]] = RangedInt(min=parm['min'], max=parm['max'])
            elif len(parm_h) == 2:
                layer[parm_h[0]][parm_h[2]] = RangedInt(min=parm['min'], max=parm['max'])
            elif len(parm_h) == 3:
                #layer.update({parm_h[0]:{parm_h[1]:{parm_h[2]:RangedInt(min=parm['min'], max=parm['max'])}}})
                #layer[parm_h[0]][parm_h[1]][parm_h[2]] = RangedInt(min=parm['min'], max=parm['max'])
                raise Exception("Not implemented")
            else:
                raise ValueError('length of hierarchi must 1,2 or 3')
        return layer
    def _serialze_network_spec(self):

        return RangedJSONEncoder().encode(self.network_spec)
