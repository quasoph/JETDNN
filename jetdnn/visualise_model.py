import tensorflow
from tensorflow import keras
import predict_ped
from keras.utils.vis_utils import plot_model

def graph_and_summary(model):
    """
    Function for displaying a visualisation of the DNN model used.
    """
    print(model.summary())
    plot_model(model, to_file='dnn_model_plot.png', show_shapes=True, show_layer_names=True)
    return
