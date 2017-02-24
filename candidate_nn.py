import json
import random
from utils import RangedNum, RangedInt, RangedJSONEncoder, flip_coin
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
    'conv_layer': {
        'filter': {
            'height': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'width': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'outchannels': {
                'min': 1,
                'max': 64,
                'type': RangedInt
            },
        },
        'strides': {
            'x': {
                'min': 1,
                'max': 2,
                'type': RangedInt
            },
            'y': {
                'min': 1,
                'max': 2,
                'type': RangedInt
            },
            'inchannels': {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
        },
        'batch': {
            'min': 1,
            'max': 1,
            'type': RangedInt
        }
    },
    'maxpool_layer': {
        'kernel': {
            'height': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'width': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'outchannels': {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
        },
        'strides': {
            'x': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'y': {
                'min': 1,
                'max': 5,
                'type': RangedInt
            },
            'inchannels': {
                'min': 1,
                'max': 1,
                'type': RangedInt
            },
            'batch': {
                'min': 1,
                'max': 1,
                'type': RangedInt
            }
        }

    },
    'feedforward_layer': {
        'size': {
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
        """Transfer the candidate to the next generation."""
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
                if flip_coin():
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
                        and flip_coin(crossover_rate)):

                        self.network_spec['layers'][layer_idx1]['activation_function'] \
                            = other_candidate.network_spec['layers'][layer_idx2]['activation_function']
            else:  # not the same, swap layer
                logging.info("crossing:layer")
                tmp = self.network_spec['layers'][layer_idx1]
                self.network_spec['layers'][layer_idx1] = other_candidate.network_spec['layers'][layer_idx2]
                other_candidate.network_spec['layers'][layer_idx2] = tmp

    def _swap_values(self, dict, other_dict, rate):
        """Swaps Properties between two Layers of the same type with Propapility rate"""
        for idx, layer_parm in enumerate(self.OPTIMIZING_PARMS[dict['type']]):
            if flip_coin(rate):
                if len(layer_parm) == 1:
                    # Save old own
                    tmp = dict[layer_parm[0]]
                    # own in other
                    dict[layer_parm[0]] = other_dict[layer_parm[0]]
                    # saved in own
                    other_dict[layer_parm[0]] = tmp
                elif len(layer_parm) == 2:
                    # Save old own
                    tmp = dict[layer_parm[0]][layer_parm[1]]
                    # own in other
                    dict[layer_parm[0]][layer_parm[1]] = other_dict[layer_parm[0]][layer_parm[1]]
                    # saved in own
                    other_dict[layer_parm[0]][layer_parm[1]] = tmp
                else:
                    raise ValueError('length of hierarchy must 1 or 2')

    def mutation(self, mutation_rate):
        # TODO: Check the mutation of a layer and the mutation of properties, layer mutation can hide value mutation
        """
        Mutate properties(layer-structure and layer-values of a Candidate)

        """
        self._fitness = None

        # Determine whether to change number of layers.

        if flip_coin(mutation_rate):
            if flip_coin():

                if len(self.network_spec['layers']) < self.runtime_spec['max_layer']:

                    # Get random index for insertion.
                    insertion_idx = random.randint(0, len(self.network_spec['layers']))
                    # Add random layer.
                    self.network_spec['layers'].insert(insertion_idx, self._create_random_layer())
            else:
                # Get random index for deletion.
                deletion_idx = random.randint(0, len(self.network_spec['layers']) - 1)
                # Delete one of the layers.
                del self.network_spec['layers'][deletion_idx]

        # Mutate layer
        for i, layer_spec in enumerate(self.network_spec['layers']):
            # Mutate complete layer.
            if flip_coin(mutation_rate / 10):
                self.network_spec['layers'][i] = self._create_random_layer()
            else:
                # Only mutate Values if no new random layer
                self._mutate_layer_values(layer_spec=self.network_spec['layers'][i], mutation_rate=mutation_rate)

    def _mutate_layer_values(self, layer_spec, mutation_rate):
        """
        Mutate each value of a layer with a probability of `mutation_rate`.
        """
        if flip_coin(mutation_rate):
            layer_spec['activation_function'] = random.choice(self.ACTIVATION_CHOICES)
        for idx, layer_parm in enumerate(self.OPTIMIZING_PARMS[layer_spec['type']]):

            if layer_parm['max'] != layer_parm['min']:

                variance = (layer_parm['max'] - layer_parm['min']) / 2
                if variance == 0:
                    variance = 1
                if layer_parm['type'] is RangedInt:
                    variance = round(variance,0)

                if len(layer_parm) == 1:
                    layer_spec[layer_parm[0]] = self._mutation_value_strategy(
                        old_value=layer_spec[layer_parm[0]],
                        variance=variance)
                elif len(layer_parm) == 2:
                    layer_spec[layer_parm[0]][layer_parm[1]] = self._mutation_value_strategy(
                        old_value=layer_spec[layer_parm[0]][layer_parm[1]],
                        variance=variance)
                else:

                    raise ValueError('length of hierarchy must 1 or 2')

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
                for idx, layer_parm in enumerate(self.OPTIMIZING_PARMS[layer_dict['type']]):
                    if layer_parm['max'] == layer_parm['min']:  # don't check on not mutable parms
                        break
                    mutable_parms += 1
                    if len(layer_parm) == 1:
                        if layer_dict[layer_parm[0]] != other_layer_dict[layer_parm[0]]:
                            div_parms += 1
                    elif len(layer_parm) == 2:
                        if layer_dict[layer_parm[0]][layer_parm[1]] != other_layer_dict[layer_parm[0]][layer_parm[1]]:
                            div_parms += 1
                    else:

                        raise ValueError('length of hierarchy must 1 or 2')
                div += (div_parms / mutable_parms)
            else:
                div += 1
        max_layers = max(len(self.network_spec['layers']), len(other_candidate.network_spec['layers']))
        return div / max_layers

    def get_fitness(self, ):
        """Get fitness of the candidate. If not yet tested, test the fitness based on the network specificaton."""
        if self._fitness is None:
            network = Network(self._serialize_network_spec())
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
            layer_spec = self._create_random_layer(layer_type=layer_type)
            # layer_spec = self._generate_network_layer(type=layer_type)
            # Add layer to the network spec.
            network_spec['layers'].append(layer_spec)

        return network_spec

    def _create_random_layer(self, layer_type=None):
        """
        Create a layer based on layer_type
        """
        if layer_type is None:
            layer_type = random.choice(self.LAYER_TYPES)

        layer_skeleton = copy.deepcopy(self.OPTIMIZING_PARMS[layer_type])
        layer_spec = self._traverse_param_spec(layer_skeleton)
        layer_spec.update(type=layer_type)
        return layer_spec

    def _traverse_param_spec(self, param_spec):

        if self._is_base_num_param(param_spec):
            return self._resolve_param_spec(param_spec)
        elif self._is_base_choice_param(param_spec):
            return random.choice(param_spec)
        else:
            resolved_spec = {}
            for k, v in param_spec.items():
                resolved_spec[k] = self._traverse_param_spec(v)
        return resolved_spec

    def _resolve_param_spec(self, param_spec):
        return param_spec['type'](param_spec['min'], param_spec['max'])

    def _is_base_num_param(self, param_spec):
        """Check whether a base parameter specification was found."""
        return isinstance(param_spec, dict) and set(param_spec.keys()) == {'min', 'max', 'type'}

    def _is_base_choice_param(self, param_spec):

        return isinstance(param_spec, tuple) or isinstance(param_spec, list)

    def _serialize_network_spec(self):

        return RangedJSONEncoder().encode(self.network_spec)
