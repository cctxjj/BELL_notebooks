import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

# creation of random forest to predict drag factor according to bez shift and drag improvement

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("drag factor: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()

ratio = df["ratio"].tolist()

# np.column_stack((drag_improvement, bez_shift))
# ratio regression
# TODO: Maybe ein Diagramm ratio - drag improvement machen?
# --> zu große Abweichung bei drag_factor direkt auf drag improvement
x_ratio, y_ratio = np.reshape(n, (-1, 1)), ratio
x_train_ratio, x_test_ratio, y_train_ratio, y_test_ratio = train_test_split(x_ratio, y_ratio, test_size=0.2)


regressor_ratio = RandomForestRegressor(n_estimators=500)
regressor_ratio.fit(x_train_ratio, y_train_ratio)
#pred = regressor.predict(np.reshape([x/1000 for x in range(200, 500)], (-1, 1))) # Kriterium definieren --> bez shift muss immer etwa n mal so groß wie z sein --> sieht man dann in Fläche
pred_1 = regressor_ratio.predict(np.reshape([x/100 for x in range(100, 3000)], (-1, 1)))
print("DragF-Ratio RFR Score:" + str(regressor_ratio.score(x_test_ratio, y_test_ratio)))
# TODO: hier irgend ne Fläche, exponentielle Regression für drag-Gleichgewicht und maybe Gleichgewicht noch irgendwie mappen --> Bedeutung ratio

x_bez_drag, y_bez_drag = np.reshape(ratio, (-1, 1)), np.column_stack((drag_improvement, bez_shift))
x_train_bez_drag, x_test_bez_drag, y_train_bez_drag, y_test_bez_drag = train_test_split(x_bez_drag, y_bez_drag, test_size=0.2)

regressor_bez_drag = RandomForestRegressor(n_estimators=500)
regressor_bez_drag.fit(x_train_bez_drag, y_train_bez_drag)
#pred = regressor.predict(np.reshape([x/1000 for x in range(200, 500)], (-1, 1))) # Kriterium definieren --> bez shift muss immer etwa n mal so groß wie z sein --> sieht man dann in Fläche
pred_2 = regressor_bez_drag.predict(np.reshape([x for x in range(0, 12000, 100)], (-1, 1)))
print("Ratio - DragI/BezSh RFR Score:" + str(regressor_bez_drag.score(x_test_bez_drag, y_test_bez_drag)))

# predictions
if to_predict >= 0:
    predicted_ratio = regressor_ratio.predict(np.reshape([to_predict], (-1, 1)))[0]
    predicted_drag_improvement, predicted_bez_shift = regressor_bez_drag.predict(np.reshape([predicted_ratio], (-1, 1)))[0]
    print(f"Predicted values for drag_factor={to_predict}: ratio={predicted_ratio}, drag improvement={predicted_drag_improvement}, bez shift={predicted_bez_shift}")

# plot prediction for ratio
plt.figure(figsize=(8, 4))
plt.plot([x/100 for x in range(100, 3000)], pred_1, color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, ratio, color="red", label="reale Modelperformance")
plt.xlabel("Drag Faktor")
plt.ylabel("Ratio")
plt.title("Verhalten Ratio gem Drag Faktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# plot ratio - drag_improvement
plt.figure(figsize=(8, 4))
plt.plot([x for x in range(0, 12000, 100)], pred_2[:, 0], color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagtes Drag Improvement")
plt.scatter(ratio, drag_improvement, color="red", label="reale Modelperformance")
plt.xlabel("ratio")
plt.ylabel("Drag Improvement")
plt.title("Verhalten Drag Improvement gem Ratio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# plot ratio - bez_shift
plt.figure(figsize=(8, 4))
plt.plot([x for x in range(0, 12000, 100)], pred_2[:, 1], color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagter Bez Shift")
plt.scatter(ratio, bez_shift, color="red", label="reale Modelperformance")
plt.xlabel("ratio")
plt.ylabel("Bez Shift")
plt.title("Verhalten Bez Shift gem Ratio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()