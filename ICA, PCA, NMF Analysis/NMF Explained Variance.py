#Description: Explained variance for NMF by number of components
#How to Use: Uploaded Excel file change n_components.

# %matplotlib qt  # interactive window
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, TruncatedSVD
import mplcursors  # pip install mplcursors

file_path = "example.xlsx"
n_components = 5

df = pd.read_excel(file_path)
x_axis = df.iloc[:, 0].values
spectra_df = df.iloc[:, 1:]
data = spectra_df.T
sample_labels = data.index.astype(str).tolist()

#Extract Sample Dates for Color
def extract_date(label):
    parts = str(label).split("|")
    return parts[0].strip() if len(parts) > 1 else "Unknown"

sample_dates = [extract_date(label) for label in sample_labels]
unique_dates = sorted(set(sample_dates))
date_to_color = {date: plt.cm.tab20(i / len(unique_dates)) for i, date in enumerate(unique_dates)}
sample_colors = [date_to_color[date] for date in sample_dates]

scaler = StandardScaler()
data_std = scaler.fit_transform(data)

#Denoising with TruncatedSVD
svd = TruncatedSVD(n_components=7, random_state=42)
data_svd = svd.fit_transform(data_std)

#Shift data to positive (required for NMF)
data_shifted = data_svd - np.min(data_svd)

#NMF Dimensionality Reduction
nmf = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=1000)
W = nmf.fit_transform(data_shifted)
H = nmf.components_

#NMF Component Importance
variances = np.var(W, axis=0)
mean_magnitudes = np.mean(W, axis=0)

nmf_importance_df = pd.DataFrame({
    'Component': [f'NMF{i+1}' for i in range(n_components)],
    'Variance': variances,
    'Mean Magnitude': mean_magnitudes
})

print("NMF component importance metrics:")
print(nmf_importance_df)

Reduce to 2D for Visualization
nmf_2d = NMF(n_components=2, init='nndsvd', random_state=42, max_iter=1000)
W_2d = nmf_2d.fit_transform(data_shifted)

#Color scaling by NMF 2D magnitude
color_values_nmf = np.linalg.norm(W_2d, axis=1)
norm_color_nmf = (color_values_nmf - color_values_nmf.min()) / (color_values_nmf.max() - color_values_nmf.min())

#Scatter Plot
scatter1 = plt.scatter(W_2d[:, 0], W_2d[:, 1],
                       color=sample_colors, edgecolors='k', alpha=0.85)

plt.title("2D NMF of Standardized Spectral Data")
plt.xlabel("NMF1")
plt.ylabel("NMF2")
plt.grid(True)
'''
#Optional legend
NUM_COLUMNS = 1
LEGEND_TITLE = "Sample Date"
LEGEND_FONT_SIZE = 9
LEGEND_LOCATION = "upper left"
LEGEND_BOX_ANCHOR = (1.05, 1)
legend_handles = [
    mpatches.Patch(color=date_to_color[date], label=date)
    for date in unique_dates
]

plt.legend(
    handles=legend_handles,
    title=LEGEND_TITLE,
    title_fontsize=LEGEND_FONT_SIZE,
    fontsize=LEGEND_FONT_SIZE,
    loc=LEGEND_LOCATION,
    bbox_to_anchor=LEGEND_BOX_ANCHOR,
    ncol=NUM_COLUMNS,
    borderaxespad=0.5
)
'''

# Hover Info
cursor1 = mplcursors.cursor(scatter1, hover=True)
@cursor1.connect("add")
def on_add(sel):
    sel.annotation.set(text=f"Sample: {sample_labels[sel.index]}")

plt.show()
