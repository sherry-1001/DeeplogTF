import time

import keras
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
# import DeepLog and Preprocessor
from deeplog import DeepLog
from preprocessor import Preprocessor

tf.disable_eager_execution()

tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("ps_hosts", "['localhost:72002']", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "['localhost:71002','localhost:71003']", "worker hosts")

FLAGS = tf.app.flags.FLAGS

ps_hosts = eval(FLAGS.ps_hosts)
worker_hosts = eval(FLAGS.worker_hosts)

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

def example_local():

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Configs
            config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)
            BATCH_SIZE = 128
            EPO = 10

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

            if FLAGS.task_index == 0:
                scf = tf.train.Scaffold(init_op=init, saver=None)
            else:
                scf = None

            ##############################################################################
            #                                  Train                                     #
            ##############################################################################

            print("Start session||||||")

            with tf.train.MonitoredTrainingSession(master="grpc://" + worker_hosts[FLAGS.task_index],
                                                   is_chief=(FLAGS.task_index == 0),
                                                   checkpoint_dir="model",
                                                   scaffold=scf,
                                                   save_checkpoint_steps=None,
                                                   config=config,
                                                   ) as sess:

                sess.run([t_itr.initializer])

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

if __name__ == "__main__":
    example_local()
    print("Done")
