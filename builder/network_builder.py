import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from builder.helper import get_tensor_size
from tensorflow.examples.tutorials.mnist import input_data
import os
import datetime
import math

VALIDATION_SIZE = 5000
mnist = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=False, validation_size=VALIDATION_SIZE)


class Network:
    """A nerual network build from a JSON specification."""

    def __init__(self, json_network_spec):

        self.network_spec = json.loads(json_network_spec)
        if self.network_spec['max_number_of_iterations'] % self.network_spec['validate_each_n_steps'] != 0:
            raise(ValueError('max_number_of_iterations is no multiple of validate_each_n_steps.'))
        self.x = None
        self.y_ = None
        self.loss = None
        self.accuracy = None
        self.train_op = None
        self._build_network()

    def evaluate(self):
        """Evaluate performance of network.

        Returns:
            The accuracy on the validation data.
        """

        merged_summary = tf.summary.merge_all()
        # Time when starting the training.
        start_time = datetime.datetime.now()
        # Arrays for storing intermediate results.
        losses = np.zeros(self.network_spec['max_number_of_iterations'] // self.network_spec['validate_each_n_steps'])
        accuracies = np.zeros(self.network_spec['max_number_of_iterations'] // self.network_spec['validate_each_n_steps'])
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.network_spec['logdir'], graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(self.network_spec['max_number_of_iterations']):
                batch = mnist.train.next_batch(self.network_spec['hyperparameters']['batchsize'])
                self.train_op.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
                if i % self.network_spec['validate_each_n_steps'] == 0:
                    # Write summary and save data for plots.
                    summary, accuracy_val, loss_val = sess.run([merged_summary, self.accuracy, self.loss],
                                                               feed_dict={self.x: batch[0], self.y_: batch[1]})
                    writer.add_summary(summary, global_step=i)
                    losses[int(i / self.network_spec['validate_each_n_steps'])] = loss_val
                    accuracies[int(i / self.network_spec['validate_each_n_steps'])] = accuracy_val

                    # Check whether training has taken too long.
                    if (datetime.datetime.now() - start_time).seconds // 60 > self.network_spec['max_runtime']:
                        break

            # Since data is to big to fit in GPU, split data into chunks of 1000 and calculate the mean.
            chunk_size = 1000
            steps = int(VALIDATION_SIZE / chunk_size)
            validation_accuracies = np.zeros(steps)

            for i in range(steps):
                validation_accuracies[i] = sess.run(
                    self.accuracy,
                    feed_dict={self.x: mnist.validation.images[i * chunk_size:(i+1) * chunk_size],
                    self.y_: mnist.validation.labels[i * chunk_size:(i+1) * chunk_size]}
                )
            # Save plots for losses and accuracies.
            self._plot(loss=losses, accuracy=accuracies)

            # Get total number of weights.
            n_weights = 0

            for var in tf.trainable_variables():
                n_weights += get_tensor_size(var)

            extended_spec = self._extend_network_spec(accuracy=float(validation_accuracies.mean()),
                                                      n_weights=n_weights)

            # Write extended to logdir.
            self._write_to_logdir(extended_spec, 'network.json')

            return extended_spec

    def feedforward_layer(self, input_tensor, layer_number):
        """Build a feedforward layer ended with an activation function.

        Args:
            input_tensor: The output from the layer before.
            layer_number (int): The number of the layer in the network.

        Returns:
            tensor: The activated output.
        """
        layer_spec = self.network_spec['layers'][layer_number]
        with tf.name_scope('feedforward' + str(layer_number)):
            weighted = self._feedforward_step(input_tensor, layer_spec['size'])
            activation = getattr(tf.nn, layer_spec['activation_function'])(weighted)
        return activation

    def conv_layer(self, input_tensor, layer_number):
        """Build a convolution layer ended with an activation function.

        Args:
            input_tensor: The output from the layer before.
            layer_number (int): The number of the layer in the network.

        Returns:
            tensor: The activated output.
        """
        inchannels, input_tensor = self._ensure_2d(input_tensor)

        layer_spec = self.network_spec['layers'][layer_number]
        filter_shape = (layer_spec['filter']['height'],
                        layer_spec['filter']['width'],
                        inchannels,
                        layer_spec['filter']['outchannels'])
        filter_strides = (layer_spec['strides']['inchannels'],
                          layer_spec['strides']['x'],
                          layer_spec['strides']['y'],
                          layer_spec['strides']['batch'])
        with tf.name_scope('conv' + str(layer_number)):
            w = self._weight_variable(filter_shape, name='W')
            b = self._bias_variable([layer_spec['filter']['outchannels']], name='b')
            conv = tf.nn.conv2d(input_tensor, w, strides=filter_strides, padding='SAME')
            activation = getattr(tf.nn, layer_spec['activation_function'])(conv + b, name='activation')
        return activation

    def maxpool_layer(self, input_tensor, layer_number):
        """Build a maxpooling layer.

               Args:
                   input_tensor: The output from the layer before.
                   layer_number (int): The number of the layer in the network.

               Returns:
                   tensor: The max pooled output.
               """

        inchannels, input_tensor = self._ensure_2d(input_tensor)
        #TODO :fix inchannels
        layer_spec = self.network_spec['layers'][layer_number]
        kernel_shape = (1,
                        layer_spec['kernel']['height'],
                        layer_spec['kernel']['width'],
                        layer_spec['kernel']['outchannels'])
        kernel_strides = (layer_spec['strides']['inchannels'],
                          layer_spec['strides']['x'],
                          layer_spec['strides']['y'],
                          layer_spec['strides']['batch'])

        with tf.name_scope('maxpool' + str(layer_number)):
            pool = tf.nn.max_pool(input_tensor, ksize=kernel_shape,
                                  strides=kernel_strides, padding='SAME', name='maxpool')
        return pool

    def _build_network(self):
        """Build network based on JSON specification.

        Construct all layers according to the JSON specification. Then project
        everything on a readout layer. Then build loss and the training op.
        """
        # Write extended to logdir.
        os.makedirs(self.network_spec['logdir'], exist_ok=True)
        self._write_to_logdir(json.dumps(self.network_spec), 'network.json')
        # Reset the old graphs from previous candidates.
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input')
        self.y_ = tf.placeholder(tf.int32, shape=[None], name='labels')
        current_tensor = self._build_layers(self.x)
        readout = self._build_readout_layer(current_tensor, n_classes=10)
        loss = self._build_loss(readout, self.y_)
        self.train_op = self._build_train_op(loss)

    def _build_layers(self, current_tensor):
        """Build layers based on the JSON specification.

        Returns:
            tensor: The output from the last layer.
        """
        for i, layer_spec in enumerate(self.network_spec['layers']):
            current_tensor = getattr(self, layer_spec['type'])(current_tensor, layer_number=i)
        return current_tensor

    def _build_readout_layer(self, input_tensor, n_classes):
        """Project into tensor onto readout layer with n classes."""
        with tf.name_scope('readout'):
            readout = self._feedforward_step(input_tensor, n_classes)
        return readout

    def _build_loss(self, readout, labels):
        """Build the layer including the loss and the accuracy.

        Args:
            readout (tensor): The readout layer. A probability distribution over the classes.
            labels (tensor): Labels as integers.

        Returns:
            tensor: The loss tensor (cross entropy).
        """

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=readout, labels=labels))
            tf.summary.scalar('cross_entropy', self.loss)
            correct_prediction = tf.nn.in_top_k(readout, labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        return self.loss

    def _build_train_op(self, loss):
        """Build the training op.

        Args:
            loss (tensor): The loss function to be optimized.

        Returns:
            The training op.
        """
        with tf.name_scope('train'):
            learning_rate = self.network_spec['hyperparameters']['learningrate']
            optimizer = getattr(tf.train, self.network_spec['hyperparameters']['optimizer'])(learning_rate)
            train_op = optimizer.minimize(loss)
        return train_op

    def _feedforward_step(self, input_tensor, size):
        """Project tensor on column of `size` many neurons.

        Args:
            input_tensor: The tensor to be projected.
            size: The size of the feedforward layer.

        Returns:
            tensor: The forwarded tensor.
        """
        # Flatten the input tensor.
        flat_dim = get_tensor_size(input_tensor)
        input_tensor_flat = tf.reshape(input_tensor, [-1, flat_dim], name='reshape')
        w = self._weight_variable([flat_dim, size], name='W')
        b = self._bias_variable([size], name='b')
        weighted = tf.matmul(input_tensor_flat, w) + b
        return weighted

    def _ensure_2d(self, input_tensor):
        """Make sure that `input_tensor` can be used for convolution and maxpooling ops.

        Args:
            input_tensor (tensor): The tensor that potentially has to be converted to 2D.

        Returns:
            Number of inchannels for the next layer.
            The `input_tensor` for the next layer.

        """
        if len(input_tensor.get_shape()) > 2:
            inchannels = int(input_tensor.get_shape()[-1])  # inchannels
        else:
            inchannels = 1
            input_tensor = self._reshape_to_2d(input_tensor)

        return inchannels, input_tensor

    def _reshape_to_2d(self, input_tensor):

        # The length of the flat tensor.
        flat_size = int(input_tensor.get_shape()[1])
        side_length = math.ceil(math.sqrt(flat_size))
        padding_size = (side_length ** 2) - flat_size
        if padding_size != 0:
            padding = tf.zeros(shape=[tf.shape(input_tensor)[0], (side_length ** 2) - flat_size], name='padding')
            input_tensor = tf.concat([input_tensor, padding], axis=1)
        input_tensor_2d = tf.reshape(input_tensor, [-1, side_length, side_length, 1], name='reshape')
        return input_tensor_2d

    @staticmethod
    def _weight_variable(shape, name):
        """Initialize weights randomly with normal distribution."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _bias_variable(shape, name):
        """Set all biases to 0.1"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def _write_to_logdir(self, file_str, fname):

        file_loc = os.path.join(self.network_spec['logdir'], fname)
        with open(file_loc, 'w') as fp:
            fp.write(file_str)

    def _extend_network_spec(self, **kwargs):
        """Write results into JSON."""
        extended_spec = self.network_spec
        extended_spec['results'] = kwargs
        extended_json_spec = json.dumps(extended_spec)
        return extended_json_spec

    def _plot(self, **kwargs):
        """Save plots in logdir.

        Keyword Args:
            A mapping between a label and a numpy array to be plotted.
        """
        steps = np.arange(
            self.network_spec['max_number_of_iterations'] // self.network_spec['validate_each_n_steps']
        ) * self.network_spec['validate_each_n_steps']
        for y_label, y_vals in kwargs.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(steps, y_vals)
            ax.set_xlabel('batch')
            ax.set_ylabel(y_label)
            fig.savefig(self.network_spec['logdir'] + y_label + '.png', format='png')
            plt.close(fig)
