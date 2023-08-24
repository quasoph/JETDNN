"""
INSPECT DATA
"""

import seaborn as sns
from .predict_ped import multi_ped

"""
The line below means we have shot numbers corresponding with the test inputs, useful for plotting at the end. It's a separate DF as we don't want to
include the shots as a feature in the neural network input.
"""

def plot_params():

    sns.pairplot(train_data[["ne_ped_height_pre-ELM__10^19(m^-3)","Ip_(MA)","P_tot_(MW)","gas_flow_rate_of_main_species_10^22__(e/s)","B_(T)","triangularity"]],diag_kind = "kde")

    return
