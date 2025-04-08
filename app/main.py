"""app entry point"""

import os
import numpy as np
import config as cfg
import tensorflow as tf

from tensorflow.keras.layers import Dense
from prettytable import PrettyTable


class DemoNetwork():
    """class to create a showcase network"""

    def __init__(self):
        self.network = None
        self.input_shape = 2

    def network_definition(self) -> None:
        """function the define the sample network"""

        self.network = tf.keras.Sequential()

        self.network.add(
            Dense(
                units=2, activation="sigmoid", input_shape=(self.input_shape,), name="Input_Layer"
            )
        )
        self.network.add(
            Dense(
                units=2, activation="sigmoid", name="Hidden_Layer_1"
            )
        )
        self.network.add(
            Dense(
                units=1, activation="sigmoid", name="Output_Layer"
            )
        )

        self.network.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
        )

    def check_saved_network_exists(self):
        """function to check if there is a trained model weights file"""

        return any(".weights.h5" in x for x in os.listdir())

    def train_network(self) -> None:
        """function to train the sample network"""

        if len(cfg.CHECKPOINT_PATH):
            self.network.load_weights(os.path.join("model", "cp-1000.weights.h5"))
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=cfg.CHECKPOINT_PATH, verbose=1, save_weights_only=True, save_freq=int(cfg.EPOCHS/20)
            )

            self.network.fit(
                tf.convert_to_tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                tf.convert_to_tensor([[0.], [1.], [1.], [0.]]),
                epochs=cfg.EPOCHS, callbacks=[cp_callback]
            )

    def check_trained_network(self) -> None:
        """function to check if the network has learned the provided logic gate data"""

        table = PrettyTable()
        table.field_names = ["INPUT 1", "INPUT 2", "ACTUAL OUTPUT", "PREDICTED OUPUT"]
        table.add_row(["0", "0", "0", str(tf.round(self.network.predict(tf.convert_to_tensor([[0., 0.]]))))])
        table.add_row(["0", "1", "1", str(tf.round(self.network.predict(tf.convert_to_tensor([[0., 1.]]))))])
        table.add_row(["1", "0", "1", str(tf.round(self.network.predict(tf.convert_to_tensor([[1., 0.]]))))])
        table.add_row(["1", "1", "0", str(tf.round(self.network.predict(tf.convert_to_tensor([[1., 1.]]))))])
        print(table)

class VizNetwork():
    """class to implement the weight vizualization"""

    def __init__(self):
        pass

    

if __name__ == "__main__":

    network_obj = DemoNetwork()
    network_obj.network_definition()
    network_obj.train_network()
    network_obj.check_trained_network()

    # visualization_obj = VizNetwork().get_trained_weights(0)
