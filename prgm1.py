#prg 1...
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Age': [20, 22, 21, 23, 24, 25, 26, 27, 28, 29],
    'Height': [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    'Weight': [45, 48, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)


df.hist(figsize=(10, 6), bins=5)

plt.suptitle("Histogram for Numerical Features")
plt.show()


df.plot(kind='box', figsize=(8, 5))

plt.title("Box Plot for Numerical Features")
plt.show()
