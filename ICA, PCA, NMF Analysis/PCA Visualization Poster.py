#Description: Comparison of PC1 vs PC2 plotting
#How to Use: Insert your file_path, change n_components based on preference


# %matplotlib qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
import mplcursors  # pip install mplcursors

file_path = "example.xlsx"
n_components = 5

df = pd.read_excel(file_path)
x_axis = df.iloc[:, 0].values
spectra_df = df.iloc[:, 1:]

data = spectra_df.T
sample_labels = data.index.astype(str).tolist()

#Extract Sample Dates for Coloring
def extract_date(label):
    parts = str(label).split("|")
    return parts[0].strip() if len(parts) > 1 else "Unknown"

sample_dates = [extract_date(label) for label in sample_labels]
unique_dates = sorted(set(sample_dates))
date_to_color = {date: plt.cm.tab20(i / len(unique_dates)) for i, date in enumerate(unique_dates)}
sample_colors = [date_to_color[date] for date in sample_dates]

scaler = StandardScaler()
data_std = scaler.fit_transform(data)

#Denoising with TruncatedSV
svd = TruncatedSVD(n_components=7, random_state=42)
data_svd = svd.fit_transform(data_std)

#PCA Dimensionality Reduction
pca = PCA(n_components=n_components, random_state=42)
S_pca = pca.fit_transform(data_svd)

#PCA Component Metrics
variances = np.var(S_pca, axis=0)
mean_magnitudes = np.mean(np.abs(S_pca), axis=0)

pca_importance_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(n_components)],
    'Variance': variances,
    'Mean Magnitude': mean_magnitudes
})

print("PCA component importance metrics:")
print(pca_importance_df)

#PCA to 2D for Visualization
pca_2d = PCA(n_components=2, random_state=42)
pca_2d_result = pca_2d.fit_transform(data_std)

#Shift to Non-negative for Visualization
pca_2d_nonneg = pca_2d_result - np.min(pca_2d_result, axis=0)

#Color Scaling by PCA magnitude
color_values_pca = np.linalg.norm(pca_2d_nonneg, axis=1)
norm_color_pca = (color_values_pca - color_values_pca.min()) / (color_values_pca.max() - color_values_pca.min())

scatter1 = plt.scatter(pca_2d_nonneg[:, 0], pca_2d_nonneg[:, 1],
                       color=sample_colors, edgecolors='k', alpha=0.85)

plt.title("2D PCA (Shifted to Non-negative) of Standardized Spectral Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

#Legend
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

# Hover Info
cursor1 = mplcursors.cursor(scatter1, hover=True)
@cursor1.connect("add")
def on_add(sel):
    sel.annotation.set(text=f"Sample: {sample_labels[sel.index]}")

plt.show()
