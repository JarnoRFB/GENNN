import json
import random
from utils import RangedNum, RangedInt, RangedJSONEncoder
from builder.network_builder import Network
import os
import math
import copy
import logging


class CandidateNN:
    # ---------------------- Static class attributes ----------------------

    OPTIMIZER_CHOICES = ('AdamOptimizer', 'AdadeltaOptimizer', 'AdagradOptimizer',
                         'FtrlOptimizer', 'ProximalGradientDescentOptimizer', 'ProximalAdagradOptimizer',
                         'RMSPropOptimizer', 'GradientDescentOptimizer')
    ACTIVATION_CHOICES = ('relu', 'relu6', 'sigmoid', 'tanh', 'elu', 'softplus', 'softsign')
    LAYER_TYPES = ("conv_layer", "maxpool_layer", "feedforward_layer")
    ACCURACY_WEIGHT = 20
    LAYER_CNT_WEIGHT = 2
    WEIGHTS_CNT_WEIGHT = 0.1
    OPTIMIZING_PARMS = {
        'conv_layer':{
            ('filter','height'):{
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('filter','width'):{
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('filter','outchannel'):{
                'min': 1,
                'max': 64,
                'type': RangedInt
            },
            ('strides', 'x'): {
                'min': 1,
                'max': 2,
                'type': RangedInt
            },
            ('strides', 'y'): {
                'min': 1,
                'max': 2,
                'type': RangedInt
            },
            ('strides','inchannels'): {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
            ('strides', 'batch'): {
                'min': 1,
                'max': 1,
                'type': RangedInt
            }
        },
        'maxpool_layer':{
            ('kernel', 'height'): {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('kernel', 'width'): {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('kernel', 'outchannels'): {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
            ('strides', 'x'): {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('strides', 'y'): {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            ('strides', 'inchannels'): {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
            ('strides', 'batch'): {
                'min': 1,
                'max': 1,
                'type': RangedInt
            }
        },
        'feedforward_layer': {
            ('size'): {
                'min': 256,
                'max': 2048,
                'type': RangedInt
            }
        }
    }
    def __init__(self, candidate_id, start_time_str, runtime_spec, network_spec=None):
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
            self._crossover_uniform(crossover_rate=crossover_parms['rate'],
                                    other_candidate=other_candidate,
                                    uniform_method=crossover_parms['uniform_method'])
        else:
            raise ValueError('not implemented crossover strategy')

    def _crossover_uniform2(self, crossover_rate, other_candidate, uniform_method):
        """Performs a unifrom Crossover between two Candidates"""
        if uniform_method == 'swap':
            min_layers = min(len(self.network_spec['layers']), len(other_candidate.network_spec['layers']))
            for layer_idx, layer in enumerate(self.network_spec['layers'][:min_layers]):
                layer_dict = layer
                other_layer_dict = other_candidate.network_spec['layers'][layer_idx]

                # Cross whole layer
                if random.uniform(0, 1) <= crossover_rate / 5:
                    tmp = copy.deepcopy(other_layer_dict)
                    other_candidate.network_spec['layers'][layer_idx] = copy.deepcopy(layer)
                    self.network_spec['layers'][layer_idx] = tmp
                else:
                    if ('activation_function' in layer_dict and
                        'activation_function' in other_layer_dict and
                        random.uniform(0, 1) <= crossover_rate):

                        layer_dict['activation_function'] = other_layer_dict['activation_function']

                    if layer_dict['type'] == other_layer_dict['type']:
                        self._swap_values(layer_dict, other_layer_dict, crossover_rate)


        else:
            raise NotImplementedError('Not implemented uniform_crossover_method')

    def _crossover_uniform(self, crossover_rate, other_candidate, uniform_method):
        min_layers = min(len(self.network_spec['layers']), len(other_candidate.network_spec['layers']))
        num_layer_crossover = max(1, int(min_layers * crossover_rate))

        for swap_idx in range(num_layer_crossover):
            layer_idx1 = random.randint(0, len(self.network_spec['layers']) - 1)
            layer_idx2 = random.randint(0, len(other_candidate.network_spec['layers']) - 1)

            # If type is the same
            if self.network_spec['layers'][layer_idx1]['type'] == other_candidate.network_spec['layers'][layer_idx2][
                'type']:
                # Make complete or parm cross
                if (random.uniform(0, 1) <= 0.5):  # Cross complete layer with lower probability
                    logging.info("crossing:sameType:layer")
                    tmp = self.network_spec['layers'][layer_idx1]
                    self.network_spec['layers'][layer_idx1] = other_candidate.network_spec['layers'][layer_idx2]
                    other_candidate.network_spec['layers'][layer_idx2] = tmp

                else:  # Same Type and cross elementwise
                    logging.info("crossing:sameType:parms")
                    self._swap_values(self.network_spec['layers'][layer_idx1],
                                      other_candidate.network_spec['layers'][layer_idx2], crossover_rate)
                    # Cross activation functino
                    if ('activation_function' in self.network_spec['layers'][layer_idx1]
                        and 'activation_function' in other_candidate.network_spec['layers'][layer_idx2]
                        and random.uniform(0, 1) <= crossover_rate):
                        self.network_spec['layers'][layer_idx1]['activation_function'] \
                            = other_candidate.network_spec['layers'][layer_idx2]['activation_function']
            else:  # not the same, swap layer
                logging.info("crossing:layer")
                tmp = self.network_spec['layers'][layer_idx1]
                self.network_spec['layers'][layer_idx1] = other_candidate.network_spec['layers'][layer_idx2]
                other_candidate.network_spec['layers'][layer_idx2] = tmp

    def _swap_values(self, dict, other_dict, rate):
        """Swaps Properties between two Layers of the same type with Propapility rate"""
        for parm in self.OPTIMIZING_PARMS[dict['type']]:
            if random.uniform(0, 1) <= rate:
                parm_h = parm['parms']['hierarchy']
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
                    raise ValueError('length of hierarchy must 1,2 or 3')

    def mutation(self, mutation_rate):
        # TODO: Check the mutation of a layer and the mutation of properties, layer mutation can hide value mutation
        """
        Mutate properties(layer-structure and layer-values of a Candidate)

        """
        self._fitness = None

        # Determine whether to change number of layers.

        if random.uniform(0, 1) <= mutation_rate:
            if random.uniform(0, 1) <= 0.5:

                if len(self.network_spec['layers']) < self.runtime_spec['max_layer']:

                    # Get random index for insertion.
                    insertion_idx = random.randint(0, len(self.network_spec['layers']))
                    # Add random layer.
                    self.network_spec['layers'].insert(insertion_idx, self._create_randomize_layer())
            else:
                # Get random index for deletion.
                deletion_idx = random.randint(0, len(self.network_spec['layers']) - 1)
                # Delete one of the layers.
                del self.network_spec['layers'][deletion_idx]

        # Mutate layer
        for i, layer_spec in enumerate(self.network_spec['layers']):
            # Mutate complete layer
            if random.uniform(0, 1) <= (mutation_rate / 10):
                self.network_spec['layers'][i] = self._create_randomize_layer()
            else:
                # Only mutate Values if no new random layer
                self._mutate_layer_values(layer_spec=self.network_spec['layers'][i], mutation_rate=mutation_rate)

    def _mutate_layer_values(self, layer_spec, mutation_rate):
        """
        Mutate each value of a layer with a probability of `mutation_rate`.
        """
        if random.uniform(0, 1) <= mutation_rate:
            layer_spec['activation_function'] = random.choice(self.ACTIVATION_CHOICES)
        for parms in self.OPTIMIZING_PARMS[layer_spec['type']]:
            if parms['parms']['max'] != parms['parms']['min']:

                parm_h = parms['parms']['hierarchy']
                variance = (parms['parms']['max'] - parms['parms']['min']) / 2
                if variance == 0:
                    variance = 1
                if parms['parms']['type'] == 'int':
                    variance = int(variance)

                if len(parm_h) == 1:
                    layer_spec[parm_h[0]] = self._mutation_value_strategy(
                        old_value=layer_spec[parm_h[0]],
                        variance=variance)
                elif len(parm_h) == 2:
                    layer_spec[parm_h[0]][parm_h[1]] = self._mutation_value_strategy(
                        old_value=layer_spec[parm_h[0]][parm_h[1]],
                        variance=variance)
                elif len(parm_h) == 3:
                    layer_spec[parm_h[0]][parm_h[1]][parm_h[2]] = self._mutation_value_strategy(
                        old_value=layer_spec[parm_h[0]][parm_h[1]][parm_h[2]],
                        variance=variance)
                else:

                    raise ValueError('length of hierarchy must 1,2 or 3')

    def _mutation_value_strategy(self, old_value, variance):
        """ sub/add a number between -variance and variance"""
        return old_value + old_value.__class__(-variance, variance).value

    def get_diversity(self, other_candidate):

        div = 0
        div += abs(len(self.network_spec['layers']) - len(other_candidate.network_spec['layers']))

        min_layers = min(len(self.network_spec['layers']), len(other_candidate.network_spec['layers']))
        for layer_idx, layer in enumerate(self.network_spec['layers'][:min_layers]):
            layer_dict = layer
            other_layer_dict = other_candidate.network_spec['layers'][layer_idx]
            if layer_dict['type'] == other_layer_dict['type']:
                # make deeper compare
                mutable_parms = 0
                div_parms = 0
                for parms in self.OPTIMIZING_PARMS[layer_dict['type']]:
                    if parms['parms']['max'] == parms['parms']['min']:  # don't check on not mutable parms
                        break
                    mutable_parms += 1
                    parm_h = parms['parms']['hierarchy']
                    if len(parm_h) == 1:
                        if layer_dict[parm_h[0]] != other_layer_dict[parm_h[0]]:
                            div_parms += 1
                    elif len(parm_h) == 2:
                        if layer_dict[parm_h[0]][parm_h[1]] != other_layer_dict[parm_h[0]][parm_h[1]]:
                            div_parms += 1
                    elif len(parm_h) == 3:
                        if layer_dict[parm_h[0]][parm_h[1]][parm_h[2]] != other_layer_dict[parm_h[0]][parm_h[1]][parm_h[2]]:
                            div_parms += 1
                    else:

                        raise ValueError('length of hierarchy must 1,2 or 3')
                div += (div_parms / mutable_parms)
            else:
                div += 1
        max_layers = max(len(self.network_spec['layers']), len(other_candidate.network_spec['layers']))
        return div / max_layers

    def get_fitness(self, ):
        """Get fitness of the candidate. If not yet tested, test the fitness based on the network specificaton."""
        if self._fitness is None:
            network = Network(self._serialze_network_spec())
            extended_spec_json = network.evaluate()
            extended_spec = json.loads(extended_spec_json)
            result_spec = extended_spec['results']
            print(result_spec)
            del network

            if self.runtime_spec['fitness_strategy'] == 'accuracy':
                self._fitness = self._fitness_function_accuracy(result_spec, self.runtime_spec['fitness_power'])
            elif self.runtime_spec['fitness_strategy'] == 's1':
                self._fitness = self._fitness_function_s1(result_spec)
            else:
                raise ValueError('fitnesss strategy {} is not implemented.'.format(self.runtime_spec['fitness_strategy']))
        return self._fitness

    def _fitness_function_accuracy(self, results, power=1):
        return results['accuracy'] ** power

    def _fitness_function_s1(self, results):
        """Calculate the fitness based on the network evaluation."""
        # TODO: get the number of weights as penalty?
        return 1 / (- self.ACCURACY_WEIGHT * math.log(results['accuracy'])
                    + self.LAYER_CNT_WEIGHT * len(self.network_spec['layers'])
                    + self.WEIGHTS_CNT_WEIGHT * results['n_weights'])

    def _create_random_network(self):
        """Construct a random network specification."""

        # Finalize runtime specification.
        layer_cnt = RangedInt(1, self.runtime_spec['max_layer'])

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
            # layer_spec = self._generate_network_layer(type=layer_type)
            # Add layer to the network spec.
            network_spec['layers'].append(layer_spec)

        return network_spec

    def _create_randomize_layer(self, layer_type=None):
        """
        Create a layer based on layer_type
        """
        if layer_type is None:
            layer_type = random.choice(self.LAYER_TYPES)

        if layer_type == 'conv_layer':
            layer_spec = self._create_conv_layer()
        elif layer_type == 'maxpool_layer':
            layer_spec = self._create_maxpool_layer()
        elif layer_type == 'feedforward_layer':
            layer_spec = self._create_ff_layer()
        else:
            raise ValueError('Invalid layer type {}'.format(layer_type))
        return layer_spec

    def _create_ff_layer(self):
        """
        Create dict for a random initialized Feedforward network
        :return:
        """

        layer = {
            'type': 'feedforward_layer',
            'size': self.OPTIMIZING_PARMS['feedforward_layer']['size']['type'](
                        self.OPTIMIZING_PARMS['feedforward_layer']['size']['min'],
                        self.OPTIMIZING_PARMS['feedforward_layer']['size']['max']),
            'activation_function': random.choice(self.ACTIVATION_CHOICES)
        }
        return layer

    def _create_conv_layer(self):
        """
        Create dict for a random initialized convolutional Layer
        """
        layer = {
            'type': 'conv_layer',
            'filter': {
                'height': self.OPTIMIZING_PARMS['conv_layer']['strides','x']['type'](
                    self.OPTIMIZING_PARMS['conv_layer']['filter','height']['min'],
                    self.OPTIMIZING_PARMS['conv_layer']['filter','height']['max']),
                'width': self.OPTIMIZING_PARMS['conv_layer']['strides','x']['type'](
                    self.OPTIMIZING_PARMS['conv_layer']['filter','width']['min'],
                    self.OPTIMIZING_PARMS['conv_layer']['filter','width']['max']),
                'outchannels': self.OPTIMIZING_PARMS['conv_layer']['strides','x']['type'](
                    self.OPTIMIZING_PARMS['conv_layer']['filter','height']['min'],
                    self.OPTIMIZING_PARMS['conv_layer']['filter','outchannels']['max'])
            },
            'strides': {
                'x': self.OPTIMIZING_PARMS['conv_layer']['strides','x']['type'](
                        self.OPTIMIZING_PARMS['conv_layer']['strides','x']['min'],
                        self.OPTIMIZING_PARMS['conv_layer']['strides','x']['max']),
                'y': self.OPTIMIZING_PARMS['conv_layer']['strides','y']['type'](
                        self.OPTIMIZING_PARMS['conv_layer']['strides','y']['min'],
                        self.OPTIMIZING_PARMS['conv_layer']['strides','y']['max']),
                'inchannels': 1,  # Must be 1. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                'batch': 1
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
                'height': RangedInt(
                    self.OPTIMIZING_PARMS['maxpool_layer'][0]['parms']['min'],
                    self.OPTIMIZING_PARMS['maxpool_layer'][0]['parms']['max']),
                'width': RangedInt(
                    self.OPTIMIZING_PARMS['maxpool_layer'][1]['parms']['min'],
                    self.OPTIMIZING_PARMS['maxpool_layer'][1]['parms']['max']),
                'outchannels': 1,
            },
            'strides': {
                'y': RangedInt(
                    self.OPTIMIZING_PARMS['maxpool_layer'][4]['parms']['min'],
                    self.OPTIMIZING_PARMS['maxpool_layer'][4]['parms']['max']),
                'x': RangedInt(1, 5),
                'inchannels': 1,
            # Must probably be 1 as well. See https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                'batch': 1
            }
        }
        return layer

    def _serialze_network_spec(self):

        return RangedJSONEncoder().encode(self.network_spec)
