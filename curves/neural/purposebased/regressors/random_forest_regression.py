import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

"""
Erstellt RFR zur Vorhersage von MSEz und cwv gem g
"""

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("g for prediction: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()


x, y = np.reshape(n, (-1, 1)), np.column_stack((drag_improvement, bez_shift))


regressor = RandomForestRegressor(
    n_estimators=1000
)
regressor.fit(x, y)
pred = regressor.predict(np.reshape([x/100 for x in range(0, 1500)], (-1, 1)))

# predictions
if to_predict >= 0:
    prediction = regressor.predict(np.reshape([to_predict], (-1, 1)))[0]
    print(f"Predicted values for g={to_predict}: cwv={prediction[0]}, MSEz={prediction[1]}")

# plot prediction
plt.figure(figsize=(6, 4))
plt.plot([x/100 for x in range(0, 1500)], pred[:, 0], color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, drag_improvement, color="red", label="reale Modelperformance")
plt.xlabel("g")
plt.ylabel("cwv")
plt.title("Verhalten cwv nach Gleichgewichtsfaktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()