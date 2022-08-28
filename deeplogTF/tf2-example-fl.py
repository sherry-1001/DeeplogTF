import time

import keras
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
# import DeepLog and Preprocessor
from deeplog import DeepLog
from preprocessor import Preprocessor

tf.disable_eager_execution()

def example_local():

    # Configs
    config = tf.ConfigProto(intra_op_parallelism_threads=48, inter_op_parallelism_threads=48)
    BATCH_SIZE = 128
    EPO = 10

    with tf.Session(config=config) as sess:

        ##############################################################################
        #                                 Load data                                  #
        ##############################################################################

        # Create preprocessor for loading data
        preprocessor = Preprocessor(
            length=20,           # Extract sequences of 20 items
            # Do not include a maximum allowed time between events
            timeout=float('inf'),
        )

        # Load data from txt file
        X, y, label, mapping = preprocessor.text("data/hdfs_train", verbose=True)

        # X = tf.data.Dataset.from_tensor_slices(X)
        # y = tf.data.Dataset.from_tensor_slices(y)

        total_item = X.shape[0]
        BATCH_NUM = int(total_item/BATCH_SIZE)
        if not total_item % BATCH_SIZE == 0:
            BATCH_NUM += 1

        train = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(X),
            tf.data.Dataset.from_tensor_slices(y)
        ))

        ds = train.batch(BATCH_SIZE)
        ds = ds.repeat(EPO)

        t_itr = ds.make_initializable_iterator()
        next = t_itr.get_next()
        x = next[0]
        y = next[1]

        ##############################################################################
        #                                  DeepLog                                   #
        ##############################################################################

        # Create DeepLog object
        deeplog = DeepLog(
            input_size=30,  # Number of different events to expect
            hidden_size=64,  # Hidden dimension, we suggest 64
            output_size=30,  # Number of different events to expect
        )

        # Train
        out = deeplog.call(x)

        global_step = tf.train.get_or_create_global_step()

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=out)

        optimz = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)

        train_acc = tf.keras.metrics.sparse_categorical_accuracy(y, out)

        init = tf.global_variables_initializer()

        ##############################################################################
        #                                  Train                                     #
        ##############################################################################

        sess.run([t_itr.initializer])
        sess.run([init])

        for i in range(EPO):
            loss_list = []
            train_acc_list = []

            itr_time = 0
            itr_num = 0
            try:
                for j in range(BATCH_NUM):

                    ts_begin = time.time()

                    ret_x, ret_y, ret_fwd, ret_loss, ret_opt, ret_trainacc = sess.run(fetches=[x,y,out,loss,optimz,train_acc])

                    ts_end = time.time()

                    loss_list.append(np.mean(ret_loss))

                    train_acc_list.append(np.count_nonzero(ret_trainacc) / len(ret_trainacc))

                    itr_time += ts_end - ts_begin
                    itr_num += 1

                    print(time.strftime('%Y-%m-%d %H:%M:%S ',time.localtime(time.time())) + f"Loss: {np.mean(loss_list).astype(np.float).round(3)}, acc: {np.mean(train_acc_list).round(10)}")
            except tf.errors.OutOfRangeError:
                pass

            print(f"epoch num: {i}")
            print(f"batch time: {round(itr_time / itr_num, 7)} s, batch num: {itr_num}")
            # print("y:%f", ret_y)
            # print("out:%f", ret_fwd)

if __name__ == "__main__":
    example_local()
    print("Done")
