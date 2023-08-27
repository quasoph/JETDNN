import tensorflow
from tensorflow import keras
import predict_ped
from keras.utils.vis_utils import plot_model

# VISUALISE MODELS

def display_model(model):
    """
    Function for displaying a visualisation of the DNN model used.
    """
    print(model.summary())
    plot_model(model, to_file='dnn_model_plot.png', show_shapes=True, show_layer_names=True)
    return

# VISUALISE PREDICTIONS

def density_npm_plot(nepeds,delta_nes):

    """
    Plots a Neutral Penetration Model pedestal prediction. Could also be used with real data rather than predictions.
    Only suitable for density pedestals.
    """

    C = 2 # set this equal to constant
    x = np.linspace(0,5,100) # set tp cross section width
    for n in range(0,10):
        npm_vals = []
        for i in range(x): # for each x value calculate a bunch of points
            npm = nepeds[n] * np.tanh(C - (x/delta_nes[n]))
            npm_vals.append(npm)
        plt.plot(x,npm_vals)
    return