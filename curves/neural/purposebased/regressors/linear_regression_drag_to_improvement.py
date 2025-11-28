
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd

# creation of two random forest to predict bez shift and drag improvement according to drag factor --> using ratio

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("drag factor: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()

ratio = df["ratio"].tolist()

x_train, x_test, y_train, y_test = train_test_split(np.reshape(n, (-1, 1)), drag_improvement, test_size=0.2)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Score: " + str(regressor.score(x_test, y_test)))
concrete_prediction = regressor.predict(np.reshape([to_predict], (-1, 1)))[0]
print(f"Predicted drag improvement for drag factor={to_predict}: {concrete_prediction}")

general_prediction = regressor.predict(np.reshape([0, 30], (-1, 1)))

# plot prediction for drag factor
plt.figure(figsize=(8, 4))
plt.plot([0, 30], general_prediction, color="purple",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, drag_improvement, color="red", label="reale Modelperformance")
plt.ylabel("Drag Improvement")
plt.xlabel("Drag Faktor")
plt.title("Drag Improvement nach Drag Faktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
