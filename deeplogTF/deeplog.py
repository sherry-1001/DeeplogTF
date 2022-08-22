import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTMCell, Softmax


class DeepLog(keras.Model):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """DeepLog model used for training and predicting logs.

            Parameters
            ----------
            input_size : int
                Dimension of input layer.

            hidden_size : int
                Dimension of hidden layer.

            output_size : int
                Dimension of output layer.

            num_layers : int, default=2
                Number of hidden layers, i.e. stacked LSTM modules.
            """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialise model layers
        lstm_cells = []
        for layer_idx in range(num_layers):
            lstm_cells.append(LSTMCell(hidden_size))

        self.rnn_layer = tf.keras.layers.RNN(lstm_cells,
                                             return_sequences=True,
                                             return_state=True)
        self.out = Dense(output_size, activation=None)
        self.softmax = Softmax(axis=-1)

    ########################################################################
    #                       Forward through network                        #
    ########################################################################

    def call(self, x):
        """Forward sample through DeepLog.

        Parameters
        ----------
        X : tensor
            Input to forward through DeepLog network.

        Returns
        -------
        result : tensor

        """
        # One hot encode X
        x = tf.one_hot(x, self.input_size, dtype=tf.float32)

        # Set initial hidden states
        hidden = self._get_initial_state(x)
        state = self._get_initial_state(x)

        out, hidden, state = self.rnn_layer(x, initial_state=[hidden, state])

        # Perform output layer
        out = self.out(out[:, -1, :])
        # Create probability
        out = self.softmax(out)
        # Return result
        return out

    ########################################################################
    #                            Predict method                            #
    ########################################################################

    def predict(self, X, batch_size=32, y=None, k=1, variable=False, verbose=1):
        """Predict the k most likely output values

            Parameters
            ----------
            X : tf.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)

            batch_size: default 32

            y : Ignored
                Ignored

            k : int, default=1
                Number of output items to generate

            verbose : boolean, default=True
                If True, print output

            Returns
            -------
            result : tf.Tensor of shape=(n_samples, k)
                k most likely outputs

            confidence : tf.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        # Get the predictions
        result = super().predict(X, batch_size, verbose=verbose)
        # Get the probabilities from the log probabilities
        # result = tf.constant(np.exp(result))
        result = np.exp(result)
        # Compute k most likely outputs
        confidence, result = self.find_topk(result, k)
        # confidence, result = tf.raw_ops.TopKV2(input=result, k=k, name="Topk")
        # Return result
        return result, confidence

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """

    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def _get_initial_state(self, X):
        """Return a given hidden state for X."""
        # Return tensor of correct shape as device
        return [tf.zeros([tf.shape(X)[0], self.hidden_size]),
                tf.zeros([tf.shape(X)[0], self.hidden_size])]

    def find_topk(self, a, k, axis=-1, largest=True, sorted=True):
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]
        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size-k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
        else:
            index_array = np.argpartition(a, k-1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)
        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(
                    sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis)
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis)
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices
