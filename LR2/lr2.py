import math
import time
import numpy as np
import matplotlib.pyplot as plt

def checked(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}: получено NaN или Inf")
    return value


def estimate_l(f, a, b, r):
    xs = np.linspace(a, b, 1001).tolist()
    ys = [checked(f(x), f"f({x})") for x in xs]
    maximum = 0.0

    for i in range(len(xs) - 1):
        if xs[i + 1] == xs[i]:
            continue

        slope = abs((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        maximum = max(maximum, checked(slope, "Секущий наклон"))

    L = r * maximum if maximum > 0 else 1.0
    return checked(L, "Оценка L")


def lower_vertices(points, L):
    lower = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        x = x1 / 2 + x2 / 2 + (y1 / 2 - y2 / 2) / L
        y = y1 / 2 + y2 / 2 - L * ((x2 - x1) / 2)

        lower.append((
            checked(y, "Нижняя оценка"),
            checked(x, "Точка пересечения"),
            i
        ))

    return lower


def piyavsky(f, a, b, eps, max_iter=10000):
    start = time.perf_counter()

    a = checked(a, "a")
    b = checked(b, "b")
    eps = checked(eps, "eps")

    if a >= b or eps <= 0:
        raise ValueError("Нужно, чтобы a < b и eps > 0")

    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("Лимит итераций должен быть целым неотрицательным числом")

    checked(b - a, "Длина отрезка")

    r = 1.2
    L = estimate_l(f, a, b, r)

    points = [
        (a, checked(f(a), "f(a)")),
        (b, checked(f(b), "f(b)"))
    ]

    iterations = 0

    while True:
        lower = lower_vertices(points, L)
        p_min, x_new, index = min(lower)

        best_x, best_y = min(points, key=lambda point: point[1])
        gap = checked(best_y - p_min, "Итоговый разрыв")

        if gap < 0:
            status = "Численная ошибка: отрицательный разрыв"
            break

        if gap <= eps:
            status = "Точность достигнута по оценке L"
            break

        if iterations >= max_iter:
            status = f"Достигнут лимит {max_iter} итераций; точность не достигнута"
            break

        x1, y1 = points[index]
        x2, y2 = points[index + 1]

        if not x1 < x_new < x2:
            status = "Предел машинной точности; требуемая точность не достигнута"
            break

        y_new = checked(f(x_new), f"f({x_new})")

        left_slope = checked(
            abs((y_new - y1) / (x_new - x1)),
            "Левый секущий наклон"
        )
        right_slope = checked(
            abs((y2 - y_new) / (x2 - x_new)),
            "Правый секущий наклон"
        )

        slope = max(left_slope, right_slope)

        if slope > L:
            L = checked(r * slope, "Обновлённая оценка L")

        points.insert(index + 1, (x_new, y_new))
        iterations += 1

    elapsed = time.perf_counter() - start

    return best_x, best_y, iterations, elapsed, points, L, gap, status


def main():
    try:
        formula = input("Введите функцию f(x): ")
        a = float(input("Левая граница a: "))
        b = float(input("Правая граница b: "))
        eps = float(input("Точность eps: "))

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

        expression = compile(formula, "<функция>", "eval")

        def f(x):
            try:
                value = eval(
                    expression,
                    {"__builtins__": {}},
                    {**allowed, "x": x}
                )
                return checked(value, f"f({x})")
            except (ArithmeticError, ValueError, TypeError, NameError) as error:
                raise ValueError(
                    f"Ошибка вычисления функции при x = {x}: {error}"
                ) from error

        x_min, y_min, iterations, elapsed, points, L, gap, status = piyavsky(
            f, a, b, eps
        )

    except (ArithmeticError, ValueError, TypeError, NameError, SyntaxError) as error:
        print("\nСтатус завершения: ошибка входных данных или вычислений")
        print(f"Причина: {error}")
        print("Итоговый разрыв: не определён")
        return

    print("\nРезультаты:")
    print(f"Статус завершения: {status}")
    print(f"Оценка L = {L:.6f}")
    print(f"x* = {x_min:.8f}")
    print(f"f(x*) = {y_min:.8f}")
    print(f"Итоговый разрыв = {gap:.10g}")
    print(f"Количество итераций: {iterations}")
    print(f"Время расчёта: {elapsed:.6f} секунд")

    try:
        x_graph = np.linspace(a, b, 2000).tolist()
        y_graph = [f(x) for x in x_graph]

        line = []

        for y, x, index in lower_vertices(points, L):
            line.append(points[index])
            line.append((x, y))

        line.append(points[-1])

    except (ArithmeticError, ValueError, TypeError, NameError) as error:
        print(f"График не построен: {error}")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(x_graph, y_graph, label="Исходная функция")
    plt.plot(
        [x for x, y in line],
        [y for x, y in line],
        label="Ломаная"
    )

    plt.scatter(
        [x for x, y in points],
        [y for x, y in points],
        s=15,
        label="Верхние вершины"
    )

    plt.scatter(
        x_min,
        y_min,
        s=100,
        marker="*",
        label="Найденное приближение"
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Метод Пиявского")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()