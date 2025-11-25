import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Eingabe: nur der Mittelteil, Rest wird automatisch ergänzt
base_name = input("Model name: ")
filename = f"C:\\Users\\Sebastian\\PycharmProjects\\BELL_notebooks\\data\\model_analysis_1\\equilibrium_data_model_{base_name}.csv"

# CSV laden
df = pd.read_csv(filename)

def norm(arr):
    res = []
    d = arr.max() - arr.min() + 1e-12
    for el in arr:
        res.append((el - arr.min()) / d)
    return res

# Arrays extrahieren (so schlicht wie möglich)
epoch = df["epoch"].to_numpy()
loss = df["loss"].to_numpy()
loss_bez = df["loss_bez"].to_numpy()
loss_drag = df["loss_drag"].to_numpy()
loss_range = df["loss_range"].to_numpy()
drag_improvement = df["drag_improvement"].to_numpy()
bezier_shift = df["bezier_shift"].to_numpy()
ratio = np.divide(loss_drag, loss_bez, out=np.full_like(loss_drag, np.nan, dtype=float), where=loss_bez!=0)

print("Model achieved max drag improvement of " + str(np.max(drag_improvement)) + " at epoch " + str(np.argmax(drag_improvement)))
print("Model achieved max bez shift of " + str(np.max(bezier_shift)) + " at epoch " + str(np.argmax(bezier_shift)))
print("Model achieved end drag improvement of " + str(drag_improvement.tolist()[-1]))
print("Model achieved end bez shift of " + str(bezier_shift.tolist()[-1]))
print("drag/mse at " + str(loss_drag.tolist()[-1]/loss_bez.tolist()[-1]))

drag_improvement = norm(drag_improvement)
bezier_shift = norm(bezier_shift)

loss = norm(loss)
loss_bez = norm(loss_bez)
loss_drag = norm(loss_drag)
loss_range = norm(loss_range)

# plot normed losses
plt.figure(figsize=(8, 4))
plt.plot(epoch, loss, "o-", label="loss")
plt.plot(epoch, loss_drag, "o-", label="drag loss")
plt.plot(epoch, loss_range, "o-", label="range loss")
plt.plot(epoch, loss_bez, "o-", label="bezier mse loss")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Loss-Entwicklung (normalisiert)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 2) drag_improvement and bezier_shift
plt.figure(figsize=(8, 4))
plt.plot(epoch, drag_improvement, "o-", label="drag improvement")
plt.plot(epoch, bezier_shift, "o-", label="bezier shift")
plt.xlabel("Epoche")
plt.ylabel("Wert")
plt.title("Drag Improvement und Bezier Shift")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 3) ratio loss_drag / loss_bez
plt.figure(figsize=(8, 4))
plt.plot(epoch, ratio, "o-", label="loss_drag / loss_bez")
plt.xlabel("Epoche")
plt.ylabel("Verhältnis")
plt.title("Verhältnis von drag zu bezier mse loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()