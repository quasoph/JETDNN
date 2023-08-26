import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pytest
from .predict_ped import single_ped

def test_single_ped():
    """Tests basic functionality of single_ped function to read a .csv or .dat file."""

    testdata = [[10,55],[67,3],[97,35],[34,68],[1,12]]
    testdf = pd.DataFrame(testdata,columns=["One","Two","Three","Four","Five"])

    return
