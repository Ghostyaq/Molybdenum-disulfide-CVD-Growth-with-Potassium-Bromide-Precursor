#Description: IC1 component vs IC2 component, plotted
#How to Use: Change the file_path

# %matplotlib qt  # Optional for interactive windows in Spyder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, NMF, PCA, TruncatedSVD
import mplcursors  # pip install mplcursors if needed
from matplotlib import cm

file_path = "sample.xlsx"
df = pd.read_excel(file_path)

x_axis = df.iloc[:, 0].values
spectra_df = df.iloc[:, 1:]
data = spectra_df.T
sample_labels = data.index.astype(str).tolist()

def extract_date(label):
    parts = str(label).split("|")
    return parts[0].strip() if len(parts) > 1 else "Unknown"

sample_dates = [extract_date(label) for label in sample_labels]
unique_dates = sorted(set(sample_dates))
date_to_color = {date: cm.Set2(i / len(unique_dates)) for i, date in enumerate(unique_dates)}
# date_to_color = {date: cm.viridis(i / len(unique_dates)) for i, date in enumerate(unique_dates)}

sample_colors = [date_to_color[date] for date in sample_dates]

data = spectra_df.T
sample_labels = data.index.astype(str).tolist()
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

#Denoised with TruncatedSVD
svd = TruncatedSVD(n_components=7, random_state=42)
data_svd = svd.fit_transform(data_std)

#ICA on denoised
n_components = 5
ica = FastICA(n_components=n_components, random_state=42)
S_ica = ica.fit_transform(data_svd)

#Make nonnegative by shifting
S_ica_nonneg = S_ica - np.min(S_ica, axis=0)

ica_df = pd.DataFrame(S_ica_nonneg, columns=[f"IC{i+1}" for i in range(n_components)], index=sample_labels)

variances = np.var(S_ica_nonneg, axis=0)
mean_magnitudes = np.mean(S_ica_nonneg, axis=0)

ica_importance_df = pd.DataFrame({
    'IC': [f'IC{i+1}' for i in range(n_components)],
    'Variance': variances,
    'Mean Magnitude': mean_magnitudes
})

print("Non-negative ICA component importance metrics:")
print(ica_importance_df)

# Reduce to 2D using ICA again (with non-neg shift)
ica_2d = FastICA(n_components=2, random_state=42)
ica_2d_result = ica_2d.fit_transform(data_std)
ica_2d_nonneg = ica_2d_result - np.min(ica_2d_result, axis=0)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(ica_2d_nonneg[:, 0], ica_2d_nonneg[:, 1],
                      c=sample_colors, edgecolors='k', alpha=0.85)
plt.colorbar(label='Color = Sample Date')
plt.title("2D ICA (Shifted to Non-negative) of Standardized Spectral Data")
plt.xlabel("IC1")
plt.ylabel("IC2")
plt.grid(True)

#Hover info
cursor1 = mplcursors.cursor(scatter, hover=True)
@cursor1.connect("add")
def on_add(sel):
    sel.annotation.set(text=f"Sample: {sample_labels[sel.index]}")

plt.show()
