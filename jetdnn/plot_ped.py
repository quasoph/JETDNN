import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras import layers

"""
Function for plotting pedestals. Intended for use with predict_ped.predict_single function, to plot predicted pedestals using NPM.
"""

def npm_plot(nepeds,delta_nes):

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