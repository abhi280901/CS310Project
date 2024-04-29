import pandas as pd
import numpy as np

DATAFILE = 'data/pokes_newest.xlsx'

print("Reading Excel file...")
i = 0
df = pd.read_excel(DATAFILE)
df = df.reset_index(drop=True)
power_cost_hp = df[["skill_damage","skill_cost","hp"]]
scale = df["power_scale"]
print("Done reading.\n")


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


data_0 = ps0
#Calculate Mean Vector
mean_vector_0 = np.mean(data_0, axis=0)
print("Mean Vector 0:")
print(mean_vector_0.to_numpy())
#Center the Data
centered_data = data_0 - mean_vector_0

#Compute Covariance Matrix
covariance_matrix_0 = np.cov(centered_data, rowvar=False)

print("Covariance Matrix 0:")
print(covariance_matrix_0)
print("\n")


data_1 = ps1
# Calculate Mean Vector
mean_vector_1 = np.mean(data_1, axis=0)
print("Mean Vector 1:")
print(mean_vector_1.to_numpy())
# Center the Data
centered_data = data_1 - mean_vector_1

# Compute Covariance Matrix
covariance_matrix_1 = np.cov(centered_data, rowvar=False)

print("Covariance Matrix 1:")
print(covariance_matrix_1)
print("\n")


data_2 = ps2
#Calculate Mean Vector
mean_vector_2 = np.mean(data_2, axis=0)
print("Mean Vector 2:")
print(mean_vector_2.to_numpy())
#Center the Data
centered_data = data_2 - mean_vector_2

#Compute Covariance Matrix
covariance_matrix_2 = np.cov(centered_data, rowvar=False)

print("Covariance Matrix 2:")
print(covariance_matrix_2)
print("\n")



data_neg1 = psneg1
#Calculate Mean Vector
mean_vector_neg1 = np.mean(data_neg1, axis=0)
print("Mean Vector -1:")
print(mean_vector_neg1.to_numpy())
#Center the Data
centered_data = data_neg1 - mean_vector_neg1

#Compute Covariance Matrix
covariance_matrix_neg1 = np.cov(centered_data, rowvar=False)

print("Covariance Matrix -1:")
print(covariance_matrix_neg1)
print("\n")



data_neg2 = psneg2
# Calculate Mean Vector
mean_vector_neg2 = np.mean(data_neg2, axis=0)
print("Mean Vector -2:")
print(mean_vector_neg2.to_numpy())
# Center the Data
centered_data = data_neg2 - mean_vector_neg2

# Compute Covariance Matrix
covariance_matrix_neg2 = np.cov(centered_data, rowvar=False)

print("Covariance Matrix -2:")
print(covariance_matrix_neg2)
print("\n")

