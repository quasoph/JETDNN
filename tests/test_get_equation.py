import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytest
import sys
from pathlib import Path

import pkg_resources

main_folder = Path(__file__).parent.parent
sys.path.insert(0, str(main_folder))
sys.path.insert(0, str(main_folder / 'jetdnn'))
sys.path.insert(0, str(main_folder / 'tests'))
os.chdir(main_folder / 'jetdnn')

import jetdnn

def test_get_equation():

    model = jetdnn.predict.build_and_test_single("filename.csv",["B-field","I_p","triangularity"],"ped_height")[0]
    input_cols = ["B-field","I_p","triangularity"]

    output_expected = model.predict(input_cols) # this should be filenamedf[input_cols], change get_equation to reflect this
    output = jetdnn.visualise.get_equation(model,input_cols)[1]

    assert output == pytest.approx(output_expected,abs(1e-3)) # checks that the output of the neural network checks out with the tensorflow predicted values