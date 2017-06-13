from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from random import randint

from utils import *
from models import GCN, MLP, GCN_Hybrid
from scipy.sparse import csr_matrix

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_hybrid', 'Model string.')  # 'gcn', 'gcn_cheby', 'gcn_hybrid', 'dense'
flags.DEFINE_float('learning_rate', 0.010, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.') #originally 64
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('summaries_dir', "./logs", "Logs directory")
flags.DEFINE_string('run_name', "gcn_512_dropout_0.3", "Run name")

# import pdb; pdb.set_trace()

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_connectome_data()
#import pdb; pdb.set_trace()

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'gcn_hybrid':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN_Hybrid
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.acc_sum], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/" + FLAGS.run_name + "/train", graph=sess.graph)
valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/" + FLAGS.run_name + "/valid", graph=sess.graph)
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.acc_sum], feed_dict=feed_dict)
    train_writer.add_summary(outs[3], epoch)

    # Validation
    cost, acc, duration, val_sum = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    valid_writer.add_summary(val_sum, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration, test_sum = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
