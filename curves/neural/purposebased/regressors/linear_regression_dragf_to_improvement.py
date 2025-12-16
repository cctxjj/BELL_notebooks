
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd

"""
Lineare Regression für cwv nach g, plottet Resultat 
"""

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("g for prediction: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()

ratio = df["ratio"].tolist()

x, y = np.reshape(n, (-1, 1)), drag_improvement

regressor = LinearRegression()
regressor.fit(x, y)
concrete_prediction = regressor.predict(np.reshape([to_predict], (-1, 1)))[0]
print(f"Predicted cwv for g={to_predict}: {concrete_prediction}")

general_prediction = regressor.predict(np.reshape([0, 15], (-1, 1)))

# plot prediction for drag factor
plt.figure(figsize=(6, 4))
plt.plot([0, 15], general_prediction, color="purple",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, drag_improvement, color="red", label="reale Modelperformance")
plt.ylabel("cwv")
plt.xlabel("g")
plt.title("cwv nach Gleichgewichtsfaktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
