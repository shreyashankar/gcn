import sys

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'dense', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.') #originally 64
flags.DEFINE_float('dropout', 0.01, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

def main():
	
if __name__ == '__main__':
    main()