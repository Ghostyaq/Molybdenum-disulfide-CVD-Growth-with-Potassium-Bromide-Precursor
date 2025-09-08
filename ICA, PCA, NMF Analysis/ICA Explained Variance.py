#Description: Explained Variance by ICA by number of components
#How to Use: Upload your file path and change n_components to preferred (5 default)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

file_path = "example.xlsx"
n_components = 5

df = pd.read_excel(file_path)
df_clean = df.drop(index=0).reset_index(drop=True)
x_axis = df_clean["REMOVE"].astype(float).values
spectra = df_clean.drop(columns=["REMOVE"]).astype(float)

#ICA
ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
ica_result = ica.fit_transform(spectra.T)
components = ica.components_

#Plot ICA components
plt.figure(figsize=(10, 6))

for i in range(n_components):
    component = components[i]

    # Flip IC1 and IC4 for visual alignment
    if i in [0, 3]:
        component = -component

    component_zeroed = (component - np.median(component)) * 10000

    plt.plot(x_axis, component_zeroed, label=f'IC{i+1}')

plt.xlim(300, 600)
plt.ylim(-10, 10)
plt.xlabel("Raman Shift (cm$^{-1}$)")
plt.ylabel("Normalized Component Intensity")
plt.title("Top 5 ICA Components (Median-Centered, Scaled)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#ICA mixing matrix (sample weights)
ica_result[:, 0] *= -1
ica_result[:, 3] *= -1
ica_df = pd.DataFrame(ica_result, columns=[f"IC{i+1}" for i in range(ica_result.shape[1])])

print("ICA Mixing Matrix (first x samples):")
print(ica_df.head())
