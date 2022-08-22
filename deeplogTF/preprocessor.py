import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class Preprocessor(object):
    """Preprocessor for loading data from standard data formats."""

    def __init__(self, length, timeout, NO_EVENT=-1337):
        """Preprocessor for loading data from standard data formats.

            Parameters
            ----------
            length : int
                Number of events in context.

            timeout : float
                Maximum time between context event and the actual event in
                seconds.

            NO_EVENT : int, default=-1337
                ID of NO_EVENT event, i.e., event returned for context when no
                event was present. This happens in case of timeout or if an
                event simply does not have enough preceding context events.
            """
        # Set context length
        self.context_length = length
        self.timeout = timeout

        # Set no-event event
        self.NO_EVENT = NO_EVENT

        # Set required columns
        self.REQUIRED_COLUMNS = {'timestamp', 'event', 'machine'}

    ########################################################################
    #                      General data preprocessing                      #
    ########################################################################

    def sequence(self, data, labels=None, verbose=False):
        """Transform pandas DataFrame into DeepCASE sequences.

            Parameters
            ----------
            data : pd.DataFrame
                Dataframe to preprocess.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            Returns
            -------
            context : tf.constant of shape=(n_samples, context_length)
                Context events for each event in events.

            events : tf.constant of shape=(n_samples,)
                Events in data.

            labels : tf.constant of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.

            mapping : dict()
                Mapping from new event_id to original event_id.
                Sequencing will map all events to a range from 0 to n_events.
                This is because event IDs may have large values, which is
                difficult for a one-hot encoding to deal with. Therefore, we map
                all Event ID values to a new value in that range and provide
                this mapping to translate back.
            """
        ################################################################
        #                  Transformations and checks                  #
        ################################################################

        # Case where a single label is given
        if isinstance(labels, int):
            # Set given label to all labels
            labels = np.full(data.shape[0], labels, dtype=int)

        # Transform labels to numpy array
        labels = np.asarray(labels)

        # Check if data contains required columns
        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                ".csv file must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        # Check if labels is same shape as data
        if labels.ndim and labels.shape[0] != data.shape[0]:
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )

        ################################################################
        #                          Map events                          #
        ################################################################

        # Create mapping of events
        mapping = {
            i: event for i, event in enumerate(np.unique(data['event'].values))
        }

        # Check that NO_EVENT is not in events
        if self.NO_EVENT in mapping.values():
            raise ValueError(
                "NO_EVENT ('{}') is also a valid Event ID".format(
                    self.NO_EVENT)
            )

        mapping[len(mapping)] = self.NO_EVENT
        mapping_inverse = {v: k for k, v in mapping.items()}

        # Apply mapping
        data['event'] = data['event'].map(mapping_inverse)

        ################################################################
        #                      Initialise results                      #
        ################################################################

        # Set events as events
        events = tf.constant(data['event'].values, dtype=tf.int64)

        # Set context full of NO_EVENTs
        context = np.full(
            shape=(data.shape[0], self.context_length),
            fill_value=mapping_inverse[self.NO_EVENT], dtype=np.int64)

        # Set labels if given
        if labels.ndim:
            labels = tf.constant(labels, dtype=tf.int64)
        # Set labels if contained in data
        elif 'label' in data.columns:
            labels = tf.constant(data['lable'].values, dtype=tf.int64)
        # Otherwise set labels to None
        else:
            labels = None

        ################################################################
        #                        Create context                        #
        ################################################################

        # Sort data by timestamp
        data = data.sort_values(by='timestamp')

        # Group by machines
        machine_grouped = data.groupby('machine')
        # Add verbosity
        if verbose:
            machine_grouped = tqdm(machine_grouped, desc='Loading')

        # Group by machine
        for machine, events_ in machine_grouped:
            # Get indices, timestamps and events
            indices = events_.index.values
            timestamps = events_['timestamp'].values
            events_ = events_['event'].values

            # Initialise context for single machine
            machine_context = np.full(
                (events_.shape[0], self.context_length),
                mapping_inverse[self.NO_EVENT],
                dtype=int,
            )

            # Loop over all parts of the context
            for i in range(self.context_length):

                # Compute time difference between context and event
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                # Check if time difference is larger than threshold
                timeout_mask = time_diff > self.timeout

                # Set mask to NO_EVENT
                machine_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_EVENT],
                    events_[:-i-1],
                )
            # Add machine_context to context
            context[indices] = machine_context

        # Convert to tensorflow tensor
        context = tf.constant(context, dtype=tf.int64)

        ################################################################
        #                        Return results                        #
        ################################################################

        # Return result
        return context, events, labels, mapping

    ########################################################################
    #                     Preprocess different formats                     #
    ########################################################################

    def text(self, path, nrows=None, labels=None, verbose=False):
        """Preprocess data from text file.

            Note
            ----
            **Format**: The assumed format of a text file is that each line in
            the text file contains a space-separated sequence of event IDs for a
            machine. I.e. for *n* machines, there will be *n* lines in the file.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            Returns
            -------
            events : tf.constant of shape=(n_samples,)
                Events in data.

            context : tf.constant of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : tf.constant of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Initialise data
        events = list()
        machines = list()

        # Open text file
        with open(path) as infile:

            # Loop over each line, i.e. machine
            for machine, line in enumerate(infile):

                # Break if machine >= nrows
                if nrows is not None and machine >= nrows:
                    break

                # Extract events for each machine
                for event in map(int, line.split()):

                    # Add data
                    events  .append(event)
                    machines.append(machine)

        # Transform to pandas DataFrame
        data = pd.DataFrame({
            'timestamp': np.arange(len(events)),  # Increasing order
            'event': events,
            'machine': machines,
        })

        # Transform to sequences and return
        return self.sequence(data, labels=labels, verbose=verbose)


if __name__ == "__main__":
    ########################################################################
    #                               Imports                                #
    ########################################################################

    import argformat
    import argparse
    import os

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Create Argument parser
    parser = argparse.ArgumentParser(
        description="Preprocessor: processes data from standard formats into DeepCASE sequences.",
        formatter_class=argformat.StructuredFormatter
    )

    # Add arguments
    parser.add_argument('file',
                        help='file      to preprocess')
    parser.add_argument('--write',
                        help='file      to write output')
    parser.add_argument('--type',              default='text',
                        help="file type to preprocess t(e)xt")
    parser.add_argument('--context', type=int, default=10,
                        help="size of context")
    parser.add_argument('--timeout', type=int, default=60 *
                        60*24, help="maximum time between context and event")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Parse type                              #
    ########################################################################

    # Allowed extensions
    ALLOWED_EXTENSIONS = {'txt', 'text'}

    ########################################################################
    #                              Preprocess                              #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length=args.context,
        timeout=args.timeout,
    )

    # Preprocess file
    if args.type == 'txt' or args.type == 'text':
        context, events, labels, mapping = preprocessor.text(args.file, verbose=True)
    else:
        raise ValueError("Unsupported file type: '{}'".format(args.type))

    ########################################################################
    #                             Show output                              #
    ########################################################################

    print("Events : {}".format(events))
    print("Context: {}".format(context))
    print("Labels : {}".format(labels))
