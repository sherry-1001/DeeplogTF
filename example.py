import tensorflow as tf
# import DeepLog and Preprocessor
from deeplog import DeepLog
from preprocessor import Preprocessor

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

##############################################################################
#                                  DeepLog                                   #
##############################################################################

# Create DeepLog object
deeplog = DeepLog(
    input_size=300,  # Number of different events to expect
    hidden_size=64,  # Hidden dimension, we suggest 64
    output_size=300,  # Number of different events to expect
)

deeplog.compile(tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train deeplog
deeplog.fit(
    x=X,
    y=y,
    epochs=10,
    batch_size=128,
    verbose=1
)

# show model structure
deeplog.summary()

# Predict using deeplog
y_pred, confidence = deeplog.predict(
    X=X,
    k=3,
)
