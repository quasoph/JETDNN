import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytest
import sys
import jetdnn

def test_read_data():
    """Tests basic functionality of read_data function to read a .csv or .dat file."""

    testdata = [[10,55],[67,3],[97,35],[34,68],[1,12]]
    testdf = pd.DataFrame(testdata,columns=["One","Two","Three","Four","Five"])
    testcsv = testdf.to_csv("filename.csv",index=False,encoding="utf-8")

    output_expected = testdf

    output = jetdnn.predict.read_data(testcsv)
    assert output == pytest.approx(output_expected,abs=1e-3)
