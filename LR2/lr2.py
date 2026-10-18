import math
import time
import numpy as np
import matplotlib.pyplot as plt


def piyavsky(f, a, b, L, eps):
    points = [(a, f(a)), (b, f(b))]
    start = time.perf_counter()
    iterations = 0

    while iterations < 10000:
        points.sort()

        lower = []

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            x = (x1 + x2) / 2 + (y1 - y2) / (2 * L)
            y = (y1 + y2) / 2 - L * (x2 - x1) / 2

            lower.append((y, x))

        p_min, x_new = min(lower)
        y_new = f(x_new)

        points.append((x_new, y_new))
        iterations += 1

        best_x, best_y = min(points, key=lambda p: p[1])

        if best_y - p_min < eps:
            break

    elapsed = time.perf_counter() - start

    return best_x, best_y, iterations, elapsed, points


formula = input("Введите функцию f(x): ")
a = float(input("Левая граница a: "))
b = float(input("Правая граница b: "))
L = float(input("Константа Липшица L: "))
eps = float(input("Точность eps: "))

if a >= b or L <= 0 or eps <= 0:
    raise ValueError("Неверные входные данные")

allowed = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
    "abs": abs,
    "pi": math.pi,
    "e": math.e
}

def f(x):
    return eval(
        formula,
        {"__builtins__": {}},
        {**allowed, "x": x}
    )


x_min, y_min, iterations, elapsed, points = piyavsky(
    f, a, b, L, eps
)

print("\nРезультаты:")
print(f"x* = {x_min:.6f}")
print(f"f(x*) = {y_min:.6f}")
print(f"Количество итераций: {iterations}")
print(f"Время работы: {elapsed:.6f} секунд")

points.sort()

x_graph = np.linspace(a, b, 2000)
y_graph = [f(x) for x in x_graph]

minorant = []

for x in x_graph:
    values = []

    for xi, yi in points:
        values.append(yi - L * abs(x - xi))

    minorant.append(max(values))

plt.figure(figsize=(10, 6))

plt.plot(x_graph, y_graph, label="Исходная функция")
plt.plot(x_graph, minorant, label="Ломаная")

plt.scatter(
    [p[0] for p in points],
    [p[1] for p in points],
    s=15,
    label="Верхние вершины"
)

plt.scatter(
    x_min,
    y_min,
    color="red",
    s=100,
    marker="*",
    label="Глобальный минимум"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Метод Пиявского")
plt.grid()
plt.legend()
plt.show()