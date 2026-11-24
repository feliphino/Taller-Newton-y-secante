import matplotlib.pyplot as plt
import numpy as np

P = lambda x: 230 * x**4 + 18 * x**3 + 9 * x**2 - 221 * x - 9

# Generar las primeras iteraciones de la secante (p0=0, p1=1)
pts = [0.0, 1.0]
for _ in range(3):  # calcula p2, p3, p4
  p_prev2, p_prev1 = pts[-2], pts[-1]
  p_next = p_prev1 - P(p_prev1) * (p_prev1 - p_prev2) / (
      P(p_prev1) - P(p_prev2)
  )
  pts.append(p_next)

x = np.linspace(-1.4, 1.1, 500)
plt.figure(figsize=(8, 5))
plt.plot(x, P(x), label=r"$P(x) = 230x^4 + 18x^3 + 9x^2 - 221x - 9$")
plt.axhline(0, color="black", linewidth=0.8)

# Marcar los dos ceros reales
ceros = [-0.040659, 0.962398]
plt.scatter(ceros, [0, 0], color="green", zorder=5, label="Ceros reales")

# Marcar y numerar p0 a p4
plt.scatter(
    pts, [P(p) for p in pts], color="red", zorder=6, label="Iteraciones secante"
)
for i, p in enumerate(pts):
  plt.annotate(
      f"$p_{i}$",
      (p, P(p)),
      textcoords="offset points",
      xytext=(0, 8),
      ha="center",
  )

plt.ylim(-130, 160)
plt.xlabel("x")
plt.ylabel("P(x)")
plt.title("Comportamiento de la secante en P(x) desde [0, 1]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("grafica_polinomio.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 12, 500)
s = 1.2 * np.exp(-t / 2) + np.exp(-t / 2) * np.cos(2 * t)
s_sin_amort = s * np.exp(t / 2)  # Equivale a 1.2 + cos(2t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Gráfica izquierda: s(t)
ax1.plot(t, s, color="blue", label=r"$s(t) = e^{-t/2}(1.2 + \cos(2t))$")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("Señal amortiguada $s(t)$")
ax1.set_xlabel("t")
ax1.set_ylabel("s(t)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica derecha: s(t) * e^(t/2)
ax2.plot(
    t, s_sin_amort, color="purple", label=r"$s(t)e^{t/2} = 1.2 + \cos(2t)$"
)
ax2.axhline(
    0.2, color="red", linestyle="--", label="Cota inferior $y = 0.2$"
)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Señal sin factor de amortiguamiento")
ax2.set_xlabel("t")
ax2.set_ylabel(r"$s(t)e^{t/2}$")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafica_senal.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

f = lambda x: 3 * x - np.exp(x)
df = lambda x: 3 - np.exp(x)


# Funciones reutilizables que devuelven (raiz, historial_de_aproximaciones)
def biseccion_hist(func, a, b, n_iter=30):
  hist = []
  for _ in range(n_iter):
    p = (a + b) / 2.0
    hist.append(p)
    if func(a) * func(p) < 0:
      b = p
    else:
      a = p
  return p, hist


def newton_raphson_hist(func, dfunc, p0, tol=1e-14, max_iter=50):
  hist = [p0]
  p_prev = p0
  for _ in range(max_iter):
    p = p_prev - func(p_prev) / dfunc(p_prev)
    hist.append(p)
    if abs(p - p_prev) < tol:
      break
    p_prev = p
  return p, hist


def secante_hist(func, p0, p1, tol=1e-14, max_iter=50):
  hist = [p0, p1]
  for _ in range(max_iter):
    p_prev2, p_prev1 = hist[-2], hist[-1]
    denom = func(p_prev1) - func(p_prev2)
    if denom == 0:
      break
    p = p_prev1 - func(p_prev1) * (p_prev1 - p_prev2) / denom
    hist.append(p)
    if abs(p - p_prev1) < tol:
      break
  return p, hist


# 1. Raíz de referencia de alta precisión
p_exacta, _ = secante_hist(f, 1.0, 2.0, tol=1e-14)

# 2. Historiales de cada método
_, hist_bis = biseccion_hist(f, 1.0, 2.0, n_iter=30)
_, hist_new = newton_raphson_hist(f, df, 1.5, tol=1e-14)
_, hist_sec = secante_hist(f, 1.0, 2.0, tol=1e-14)

# 3. Calcular errores omitiendo el error exactamente igual a 0
err_bis = [
    (n, abs(p - p_exacta))
    for n, p in enumerate(hist_bis, start=1)
    if abs(p - p_exacta) > 0
]
err_new = [
    (n, abs(p - p_exacta))
    for n, p in enumerate(hist_new)
    if abs(p - p_exacta) > 0
]
err_sec = [
    (n, abs(p - p_exacta))
    for n, p in enumerate(hist_sec)
    if abs(p - p_exacta) > 0
]

# 4. Graficar en escala semilogarítmica
plt.figure(figsize=(9, 5.5))
plt.semilogy(*zip(*err_bis), "o-", label="Bisección en [1, 2]")
plt.semilogy(*zip(*err_new), "s-", label="Newton ($p_0 = 1.5$)")
plt.semilogy(*zip(*err_sec), "^-", label="Secante ($p_0 = 1, p_1 = 2$)")

# Cota de la bisección: (b - a) / 2^n
n_vals = np.arange(1, 31)
cota_bis = (2.0 - 1.0) / (2.0**n_vals)
plt.semilogy(
    n_vals,
    cota_bis,
    "--",
    color="gray",
    label=r"Cota bisección $(b-a)/2^n$",
)

# Línea horizontal en 10^-8
plt.axhline(1e-8, color="red", linestyle=":", label=r"Tolerancia $10^{-8}$")

plt.xlabel("Iteración $n$")
plt.ylabel(r"Error absoluto $|p_n - p|$")
plt.title(r"Comparación de convergencia en $3x - e^x = 0$")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("grafica_convergencia.png", dpi=300)
plt.show()