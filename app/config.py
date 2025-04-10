""" module to store all config settings"""

import os

# hyper params
LEARNING_RATE = 0.45
EPOCHS = 1000
STEPS = 1000

# paths
CHECKPOINT_PATH = os.path.join(os.getcwd(), "model", "cp-{epoch:04d}.weights.h5")
TRAINED_WEIGHTS_PATH = os.path.join(os.getcwd(), "model")
PLOT_PATH = os.path.join(os.getcwd(), "plots")
