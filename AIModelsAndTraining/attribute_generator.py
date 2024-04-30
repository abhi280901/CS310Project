#!/usr/bin/env python3
import numpy as np
import pandas as pd

DATAFILE = 'data/pokes_newest.xlsx'

def read(data):
    #print("Reading Excel file...")
    df = pd.read_excel(data)
    df = df.reset_index(drop=True)
    power_cost_hp = df[["skill_damage","skill_cost","hp"]]
    scale = df["power_scale"]
    #print("Done reading.")
    indices = np.where(scale == 0)[0]
    ps0 = power_cost_hp.iloc[indices]
    indices = np.where(scale == 1)[0]
    ps1 = power_cost_hp.iloc[indices]
    indices = np.where(scale == 2)[0]
    ps2 = power_cost_hp.iloc[indices]
    indices = np.where(scale == -1)[0]
    psneg1 = power_cost_hp.iloc[indices]
    indices = np.where(scale == -2)[0]
    psneg2 = power_cost_hp.iloc[indices]
    return ps0,ps1,ps2,psneg1,psneg2

def sample(ps):
    data = ps
    # calculating mean vector
    mean_vector = np.mean(data, axis=0)
    # centering data 
    centered_data = data - mean_vector
    # covariance matrix computation
    covariance_matrix = np.cov(centered_data, rowvar=False)
    sample = np.random.multivariate_normal(mean_vector.to_numpy(),covariance_matrix,1)
    # sample = [[Att, Cost, HP]]
    sample[0][0] = np.round(sample[0][0])
    sample[0][2] = np.round(sample[0][2])
    sample[0][1] = np.floor(sample[0][1])
    return sample
