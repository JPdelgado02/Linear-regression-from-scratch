import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


# variable X is the amount of fertilizer (kg/ha) and variable Y is the production (Ton/ha)
x = np.array([50, 60, 70, 80, 60, 100, 110, 120, 130, 140]). reshape(-1, 1)
y = np.array([2.1, 2.4, 2.8, 3.1, 3.5, 3.7, 4.0, 4.3, 4.6, 4.8])

modelo = LinearRegression()
modelo.fit(x, y) 

y_pred = modelo.predict(x)

plt.scatter(x, y, color="green", label="Datos reales (maiz)")
plt.plot(x, y_pred, color="red", linewidth=2, label="Regresion lineal")
plt.xlabel("Fertilizante aplicado (Kh/ha)")
plt.ylabel("Rendimiento (ton/ha)")
plt.title("Regresion Lineal: Fertilizante Vs Rendimiento de Maiz")
plt.legend()
plt.show()

print("pendiente (coeficiente):", modelo.coef_)
print("intercepto:", modelo.intercept_)
