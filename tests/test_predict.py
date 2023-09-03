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

# DNN TESTING FUNCTIONS
"""
The error provided is the error found for temperature data in the dataset used for "On neutral ground: can neutrals influence plasma pedestal formation?" by S. Frankel, J. Simpson and E. Solano
It is suggested to use this dataset or temperature pedestal data for testing with the below functions.

Assumes that test_data_manip is successful.
"""

test_csv = "table_EUROfusion_db_JSimpson_24april2019_D_withpellets_normp_nokikcs_only_validated.dat"
df = pd.read_csv(os.path.abspath("../" + test_csv),sep="\s{3,}|\s{3,}|\t+|\s{3,}\t+|\t+\s{3,}",skipinitialspace=True)
print(df.columns)

def test_build_and_test_single():

    test_csv = "table_EUROfusion_db_JSimpson_24april2019_D_withpellets_normp_nokikcs_only_validated.dat"
    csv_path = os.path.abspath("../" + test_csv)
    test_data = pd.read_csv(csv_path,sep="\s{3,}|\s{3,}|\t+|\s{3,}\t+|\t+\s{3,}",skipinitialspace=True)

    input_cols = ["Ip (MA)","P_tot (MW)","B (T)"]
    real_output_col = ["Te ped height pre-ELM (keV)"]

    output_expected = test_data[real_output_col]
    output = jetdnn.predict.build_and_test_single(csv_path,input_cols,real_output_col)[1] # returns flat_ped, or the predictions

    assert output == pytest.approx(output_expected,abs(0.011)) # 0.011 keV based on temperature pedestal findings from summer placement

def test_predict_single():
    
    test_data = pd.read_csv("table_EUROfusion_db_JSimpson_24april2019_D_withpellets_normp_nokikcs_only_validated.dat")
    input_cols = ["B-field","I_p","triangularity"]
    real_output_col = ["ped_height"]

    model = jetdnn.predict.build_and_test_single(test_data,input_cols,real_output_col)[0]

    output_expected = test_data[real_output_col]
    output = jetdnn.predict.predict_single(model,test_data,input_cols)[1] # returns flat_ped, or the predictions

    assert output == pytest.approx(output_expected,abs(0.011)) # 0.011 keV based on temperature pedestal findings from summer placement