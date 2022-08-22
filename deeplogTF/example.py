import tensorflow as tf
import numpy as np
from deeplog import DeepLog
from preprocessor import Preprocessor
from sklearn.metrics import classification_report

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
X_train, y_train, label, mapping = preprocessor.text(
    "data/hdfs_train", verbose=True)
X_test, y_test, label_test, mapping_test = preprocessor.text(
    "data/hdfs_test_normal", verbose=True)
X_test_anomaly, y_test_anomaly, label_test_anomaly, mapping_test_anomaly = preprocessor.text(
    "data/hdfs_test_abnormal", verbose=True)

##############################################################################
#                                  DeepLog                                   #
##############################################################################

# Create DeepLog object
deeplog = DeepLog(
    input_size=30,  # Number of different events to expect
    hidden_size=64,  # Hidden dimension, we suggest 64
    output_size=30,  # Number of different events to expect
)

deeplog.compile(tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

# Train deeplog
deeplog.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    batch_size=128,
    verbose=1
)

# show model structure
deeplog.summary()

# Predict nomal data using deeplog
y_pred_normal, confidence = deeplog.predict(
    X=X_test,
    k=9,
)

# Predict anomalous data using deeplog
y_pred_anomaly, confidence = deeplog.predict(
    X=X_test_anomaly,
    # Change this value to get the top k predictions (called 'g' in DeepLog paper, see Figure 6)
    k=9,
)

################################################################################
#                            Classification report                             #
################################################################################

print("Classification report - predictions")
print(classification_report(
    y_true=y_test,
    y_pred=y_pred_normal[:, 0],
    digits=4,
    zero_division=0,
))

################################################################################
#                             Check for anomalies                              #
################################################################################

# Check if the actual value matches any of the predictions
# If any prediction matches, it is not an anomaly, so to get the anomalies, we
# invert our answer using ~

# Check for anomalies in normal data (ideally, we should not find any)
anomalies_normal = ~np.any(y_pred_normal.T, axis=0)

# Check for anomalies in abnormal data (ideally, we should not find all)
anomalies_abnormal = ~np.any(y_pred_anomaly.T, axis=0)

# Compute classification report for anomalies
y_pred = np.concatenate((anomalies_normal, anomalies_abnormal), axis=0)
y_true = np.concatenate((np.zeros(anomalies_normal.shape[0], dtype=bool),
                        np.ones(anomalies_abnormal.shape[0], dtype=bool)),
                        axis=0)

print("Classification report - anomalies")
print(classification_report(
    y_pred=y_pred,
    y_true=y_true,
    labels=[False, True],
    target_names=["Normal", "Anomaly"],
    digits=4,
))
