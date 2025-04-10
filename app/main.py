"""app entry point"""

import os
import numpy as np
import config as cfg
import tensorflow as tf
import plotly.graph_objs as go

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

        # return to some other class if needed.
        return self.network

    def check_saved_network_exists(self):
        """function to check if there is a trained model weights file"""

        return any(".weights.h5" in x for x in os.listdir())

    def train_network(self) -> None:
        """function to train the sample network"""

        if len(os.listdir(cfg.TRAINED_WEIGHTS_PATH)):
            self.network.load_weights(os.path.join("model", "cp-1000.weights.h5"))
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=cfg.CHECKPOINT_PATH, verbose=1, save_weights_only=True, save_freq=int(cfg.EPOCHS/50)
            )

            self.network.fit(
                tf.convert_to_tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                tf.convert_to_tensor([[0.], [1.], [1.], [0.]]),
                epochs=cfg.EPOCHS, steps_per_epoch=cfg.STEPS, callbacks=[cp_callback]
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
        self.network = DemoNetwork().network_definition()
        self.weight_history = {}
        self.layer_trace_data = {
            "line_trace": {},
            "heat_trace": {}
        }

    def load_trained_weights(self) -> None:
        """function to load the saved weights during network training checkpoints"""

        for file_index, epoch_weights_file in enumerate(os.listdir(cfg.TRAINED_WEIGHTS_PATH)):
            self.network.load_weights(os.path.join(cfg.TRAINED_WEIGHTS_PATH, epoch_weights_file))
            self.weight_history[file_index] = {}
            for layer_index in range(len(self.network.layers)):
                self.weight_history[file_index][layer_index] = {
                    "weights": self.network.get_weights()[2*layer_index],
                    "bias": self.network.get_weights()[2*layer_index+1]
                }

    def get_line_trace(self, layer_data: dict, layer: int,  weights: np.ndarray, bias: np.ndarray) -> dict:
        """function to get the line trace of weight training"""

        flattened_weights = weights.reshape(-1)
        flattened_bias = bias.reshape(-1)

        if layer_data.get(layer, None) is None:
            layer_data[layer] = {}
            for weight_index, weight in enumerate(flattened_weights):
                layer_data[layer][f"weight ({int(weight_index/2 + 1)}, {int(weight_index%2 + 1)})"] = [weight]
            for bias_index, bias in enumerate(flattened_bias):
                layer_data[layer][f"bias ({int(bias_index/2 + 1)}, {int(bias_index%2 + 1)})"] = [bias]
        else:
            for weight_index, weight in enumerate(flattened_weights):
                layer_data[layer][f"weight ({int(weight_index/2 + 1)}, {int(weight_index%2 + 1)})"].append(weight)
            for bias_index, bias in enumerate(flattened_bias):
                layer_data[layer][f"bias ({int(bias_index/2 + 1)}, {int(bias_index%2 + 1)})"].append(bias)

        return layer_data


    def get_heat_trace(self) -> None:
        pass

    def get_trained_weights(self) -> None:
        """function to get historical weights data as needed"""

        for layer in range(len(self.network.layers)):
            for epoch in self.weight_history.keys():
                self.layer_trace_data["line_trace"] = self.get_line_trace(
                    self.layer_trace_data["line_trace"], layer,
                    self.weight_history[epoch][layer]["weights"],
                    self.weight_history[epoch][layer]["bias"]
                )
    
    def visualize_trace_data(self) -> None:
        """function to visualize the weight training"""

        for _, values in self.layer_trace_data.items():
            for layer, weights in values.items():
                layer_traces = []
                for weight_index, weight in weights.items():
                    layer_traces.append(
                        go.Scatter(
                            x=(np.arange(len(weight))+1)*20, y=np.asarray(weight),
                            name=weight_index, mode="lines+markers"
                        )
                    )
                fig = go.Figure(data=layer_traces)
                fig.update_layout(
                    template="plotly_dark", title=f"Layer-{layer} Weights",
                    xaxis_title="Epoch", yaxis_title="Value"
                )
                fig.write_html(os.path.join(cfg.PLOT_PATH, f"weight_convergence_layer_{layer}.html"))


if __name__ == "__main__":

    network_obj = DemoNetwork()
    network_obj.network_definition()
    network_obj.train_network()
    network_obj.check_trained_network()

    viznet_obj = VizNetwork()
    viznet_obj.load_trained_weights()
    viznet_obj.get_trained_weights()

    viznet_obj.visualize_trace_data()
