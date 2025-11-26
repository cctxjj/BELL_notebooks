import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

# creation of random forest to predict drag factor according to bez shift and drag improvement

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

n = df["n"].tolist()[:-7]
drag_improvement = df["drag_improvement"].tolist()[:-7]
bez_shift = df["bezier_shift"].tolist()[:-7]

# np.column_stack((drag_improvement, bez_shift))
x, y = np.reshape(n, (-1, 1)), drag_improvement
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(n)
print(drag_improvement)
print(bez_shift)

regressor = RandomForestRegressor(n_estimators=500)
regressor.fit(x_train, y_train)
#pred = regressor.predict(np.reshape([x/1000 for x in range(200, 500)], (-1, 1))) # Kriterium definieren --> bez shift muss immer etwa n mal so groß wie z sein --> sieht man dann in Fläche
pred = regressor.predict(np.reshape([x/100 for x in range(100, 1100)], (-1, 1)))
print(regressor.score(x_test, y_test))
# TODO: hier irgend ne Fläche, exponentielle Regression für drag-Gleichgewicht und maybe Gleichgewicht noch irgendwie mappen --> Bedeutung ratio

plt.figure(figsize=(8, 4))
plt.plot([x/100 for x in range(100, 1100)], pred, color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, drag_improvement, color="red", label="reale Modelperformance")
plt.xlabel("Drag Faktor")
plt.ylabel("Drag Improvement ")
plt.title("Verhalten Drag improvement nach Drag Faktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()