#prg 2....

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'Age': [20, 22, 24, 26, 28],
    'Height': [150, 155, 160, 165, 170],
    'Weight': [45, 50, 55, 60, 65],
    'Marks': [70, 75, 80, 85, 90]
}
df = pd.DataFrame(data)
corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
