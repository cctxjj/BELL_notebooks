
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd

# creation of two random forest to predict bez shift and drag improvement according to drag factor --> using ratio

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("target drag improvement: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()

ratio = df["ratio"].tolist()

x_train, x_test, y_train, y_test = train_test_split(np.reshape(drag_improvement, (-1, 1)), n, test_size=0.2)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Score: " + str(regressor.score(x_test, y_test)))
concrete_prediction = regressor.predict(np.reshape([to_predict], (-1, 1)))[0]
print(f"Predicted drag factor for drag improvement={to_predict}: {concrete_prediction}")

general_prediction = regressor.predict(np.reshape([x for x in range(0, 2)], (-1, 1)))

# plot prediction for drag factor
plt.figure(figsize=(8, 4))
plt.plot([x for x in range(0, 2)], general_prediction, color="purple",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(drag_improvement, n, color="red", label="reale Modelperformance")
plt.xlabel("Drag Improvement")
plt.ylabel("Drag Faktor")
plt.title("Drag Faktor nach Drag Improvement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
