import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

# creation of random forest to predict drag factor according to bez shift and drag improvement

spec = input("csv specification: ")
df = pd.read_csv(f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\crossmodel_analysis_{spec}.csv")

to_predict = float(input("drag factor: "))

n = df["n"].tolist()
drag_improvement = df["drag_improvement"].tolist()
bez_shift = df["bezier_shift"].tolist()

# np.column_stack((drag_improvement, bez_shift))
# ratio regression
# TODO: Maybe ein Diagramm ratio - drag improvement machen?
# --> zu große Abweichung bei drag_factor direkt auf drag improvement
x_ratio, y_ratio = np.reshape(n, (-1, 1)), np.column_stack((drag_improvement, bez_shift))
x_train_ratio, x_test_ratio, y_train_ratio, y_test_ratio = train_test_split(x_ratio, y_ratio, test_size=0.2)


regressor = RandomForestRegressor(
    n_estimators=1000
)
regressor.fit(x_train_ratio, y_train_ratio)
#pred = regressor.predict(np.reshape([x/1000 for x in range(200, 500)], (-1, 1))) # Kriterium definieren --> bez shift muss immer etwa n mal so groß wie z sein --> sieht man dann in Fläche
pred = regressor.predict(np.reshape([x/100 for x in range(0, 3000)], (-1, 1)))
print("Score:" + str(regressor.score(x_test_ratio, y_test_ratio)))
# TODO: hier irgend ne Fläche, exponentielle Regression für drag-Gleichgewicht und maybe Gleichgewicht noch irgendwie mappen --> Bedeutung ratio



# predictions
if to_predict >= 0:
    prediction = regressor.predict(np.reshape([to_predict], (-1, 1)))[0]
    print(f"Predicted values for drag_factor={to_predict}: drag improvement={prediction[0]}, bez shift={prediction[1]}")

# plot prediction for ratio
plt.figure(figsize=(8, 4))
plt.plot([x/100 for x in range(0, 3000)], pred[:, 0], color="purple", linestyle="dashed",
    linewidth=2.0,
    alpha=0.8,
    solid_joinstyle="round",
    solid_capstyle="round",
    antialiased=True,
    zorder=2,
  label="vorhergesagte Modelperformance")
plt.scatter(n, drag_improvement, color="red", label="reale Modelperformance")
plt.xlabel("Drag Faktor")
plt.ylabel("Drag improvement")
plt.title("Verhalten Drag Improvement gem Drag Faktor")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()