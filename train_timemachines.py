from timemachines.skaters.sk.sfautoarima import sf_autoarima as f
import numpy as np
from tqdm import tqdm
from data.energy import get_energy_data

data = get_energy_data()

y = data["P"]
s = {}
y_pred = []
for yi in tqdm(y):
    ypred_i, y_pred_std, s = f(y=yi, s=s, k=1, e=1000)
    y_pred.append(ypred_i)
