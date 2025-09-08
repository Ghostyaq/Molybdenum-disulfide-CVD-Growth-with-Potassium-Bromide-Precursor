#Description: Show the explained variance based on number of components
#How to Use: Upload your file path, then change n_components to visualize the explained variance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path = "example.xlsx"
n_components = 5

data = pd.read_excel(file_path)
#Compute global baseline
individual_means = data.mean(axis=1)
global_baseline = individual_means.mean()
print(f"Global average baseline: {global_baseline:.4f}")

pca = PCA()
pca_result = pca.fit_transform(data)

#Plot top 5 PCA components
plt.figure(figsize=(10, 6))

for i in range(n_components):  # PC1 to PC5
    component = pca.components_[i]

    # Flip sign if negative-dominant
    if np.mean(component) < 0:
        component = -component

    # Median-center and shift to start at zero
    component_zeroed = component - np.median(component)
  
    # Add to plot
    plt.plot(x_axis, component_zeroed, label=f'PC{i+1}')

plt.xlim(350, 550)
plt.ylim(-0.35, 0.3)
plt.xlabel("Raman Shift")
plt.ylabel("Normalized Component Intensity")
plt.title("Top 5 PCA Components (Overlaid, Median-Centered, Peak Normalized)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
