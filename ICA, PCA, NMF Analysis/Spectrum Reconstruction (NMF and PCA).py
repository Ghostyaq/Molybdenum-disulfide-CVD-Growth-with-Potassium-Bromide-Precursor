#Description: Reconstructs the spectra using PCA and NMF data.
#How to Use: Change the file_path to what is needed, alter n_components_abc and n_samples_to_plot to alter the graph.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD

file_path = "example.xlsx"
n_components_pca = 5
n_components_nmf = 5
n_samples_to_plot = 3


df = pd.read_excel(file_path)
data = df.T  #each row = one spectrum, columns = Raman shifts
sample_labels = data.index.astype(str).tolist()  # sample names now are original column headers

#Standardize data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

#Denoise with TruncatedSVD
svd = TruncatedSVD(n_components=7, random_state=42)
data_svd = svd.fit_transform(data_std)

#PCA Reconstruction
pca_model = PCA(n_components=n_components_pca)
X_pca = pca_model.fit_transform(data_std)
X_pca_reconstructed_std = X_pca @ pca_model.components_ + pca_model.mean_
X_pca_reconstructed = scaler.inverse_transform(X_pca_reconstructed_std)

#NMF reconstruction
data_min = data.min().min()
data_shifted = data - data_min if data_min < 0 else data.copy()

nmf_model = NMF(n_components=n_components_nmf, init='random', random_state=42, max_iter=1000)
W = nmf_model.fit_transform(data_shifted)
H = nmf_model.components_
X_nmf_reconstructed = W @ H  # in shifted scale
if data_min < 0:
    X_nmf_reconstructed += data_min  # shift back

#Double-Checking x-axis
try:
    wavenumbers = data.columns.astype(float)
except:
    wavenumbers = np.arange(data.shape[1])

#Plot some random sample reconstructions
np.random.seed(42)
sample_indices = np.random.choice(data.shape[0], size=n_samples_to_plot, replace=False)

for idx in sample_indices:
    original = data.iloc[idx].values
    reconstructed_pca = X_pca_reconstructed[idx]
    reconstructed_nmf = X_nmf_reconstructed[idx]

    plt.figure(figsize=(10, 5))
    plt.plot(wavenumbers, original, label='Original', color='black', linewidth=2)
    plt.plot(wavenumbers, reconstructed_pca, label='PCA Reconstruction', color='blue', linestyle='--')
    plt.plot(wavenumbers, reconstructed_nmf, label='NMF Reconstruction', color='green', linestyle='-.')
    plt.title(f"Spectrum {sample_labels[idx]} - Reconstruction Comparison")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
