import sympy as sp

EPS = 1e-8

def is_real_solution(solution, unknowns):
    if any(v not in solution for v in unknowns):
        raise ValueError("Получено семейство решений: нужен отдельный анализ")
    values = [sp.simplify(solution[v]) for v in unknowns]
    if any(value.free_symbols for value in values):
        raise ValueError("В решении остались параметры: нужен отдельный анализ")
    if any(v.is_real is False or v.is_finite is False for v in values):
        return False
    if any(v.is_real is None or v.is_finite is None for v in values):
        raise ValueError("Не удалось подтвердить действительность решения")
    return True


def residual_is_zero(residual):
    residual = sp.simplify(residual)
    if residual.is_real is not True or residual.is_finite is not True:
        return False
    if residual.has(sp.Float):
        return abs(float(residual)) <= EPS
    return residual == 0


def main():
    x, y = sp.symbols("x y", real=True)
    lam = sp.symbols("lambda", real=True)
    f = x + y
    constraints = [x**2 + y**2 - 2]
    variables = [x, y]
    multipliers = [lam]

    L = f + sum(a * h for a, h in zip(multipliers, constraints))
    derivatives = [sp.diff(L, v) for v in variables]
    equations = derivatives + constraints
    unknowns = variables + multipliers
    solutions = sp.solve(equations, unknowns, dict=True)

    print("Вариант 1")
    print("Целевая функция:", f)
    print("Ограничения:")
    for number, h in enumerate(constraints, start=1):
        print(f"  h{number}(x, y) = {h} = 0")
    print("Функция Лагранжа:", L)
    print("Частные производные:")
    for v, derivative in zip(variables, derivatives):
        print(f"  dL/d{v} = {derivative}")
    print("Уравнения стационарности с ограничениями:")
    for equation in equations:
        print(f"  {equation} = 0")
    print("Область определения: R^2; f и h1 — многочлены")
    print("Допуск для численных невязок:", EPS)

    jacobian = sp.Matrix(constraints).jacobian(variables)
    zero_gradient_points = sp.solve(list(jacobian), variables, dict=True)
    singular_points = []
    print("\nМатрица Якоби:", jacobian)
    for point in zero_gradient_points:
        if not is_real_solution(point, variables):
            continue
        residuals = [sp.simplify(h.subs(point)) for h in constraints]
        print("Точка нулевого градиента:", tuple(point[v] for v in variables))
        print("  невязки ограничений:", residuals)
        if all(residual_is_zero(r) for r in residuals):
            singular_points.append(point)
    if singular_points:
        raise ValueError("Найдены особые допустимые точки: нужен отдельный анализ")
    print("Особых допустимых точек нет; ранг Якоби на окружности равен 1 = m")

    candidates = []
    for solution in solutions:
        if not is_real_solution(solution, unknowns):
            print("Отброшено недействительное или бесконечное решение:", solution)
            continue
        point_subs = {v: solution[v] for v in variables}
        residuals = [sp.simplify(h.subs(point_subs)) for h in constraints]
        value = sp.simplify(f.subs(point_subs))
        domain_ok = value.is_real is True and value.is_finite is True
        admissible = domain_ok and all(residual_is_zero(r) for r in residuals)
        system_ok = all(residual_is_zero(e.subs(solution)) for e in equations)
        candidate = {
            "point": tuple(solution[v] for v in variables),
            "multipliers": tuple(solution[a] for a in multipliers),
            "value": value,
            "residuals": residuals,
            "admissible": admissible,
            "system_ok": system_ok,
            "rank": jacobian.subs(point_subs).rank(),
        }
        candidates.append(candidate)

    for number, candidate in enumerate(candidates, start=1):
        print(f"\nКандидат {number}:")
        print("  точка:", candidate["point"])
        print("  множители:", dict(zip(multipliers, candidate["multipliers"])))
        print("  значение функции:", candidate["value"])
        print("  невязки ограничений:", candidate["residuals"])
        print("  допустимость:", "да" if candidate["admissible"] else "нет")
        print("  система выполнена:", "да" if candidate["system_ok"] else "нет")
        print("  ранг Якоби:", candidate["rank"])

    valid = [c for c in candidates if c["admissible"] and c["system_ok"]]
    if not valid or len(valid) != len(candidates):
        raise ValueError("Проверка кандидатов не завершена: вывод о глобальности невозможен.")

    minimum = maximum = valid[0]
    for candidate in valid[1:]:
        if candidate["value"] < minimum["value"]:
            minimum = candidate
        if candidate["value"] > maximum["value"]:
            maximum = candidate

    print("\nИтог:")
    print(f"  глобальный минимум: {minimum['value']} в точке {minimum['point']}")
    print("  множитель при минимуме:", minimum["multipliers"][0])
    print(f"  глобальный максимум: {maximum['value']} в точке {maximum['point']}")
    print("  множитель при максимуме:", maximum["multipliers"][0])
    print("Основание классификации для варианта 1:")
    print("  Допустимые точки найдены, поэтому множество непусто.")
    print("  Окружность замкнута и ограничена; целевая функция непрерывна.")
    print("  По теореме Вейерштрасса оба глобальных экстремума достигаются.")
    print("  Все допустимые точки регулярны; экстремумы входят в систему Лагранжа.")
    print("  Полнота: lambda = 0 даёт 1 = 0, поэтому lambda != 0.")
    print("  Из разности первых двух уравнений x = y, затем x**2 = 1.")
    print("  Оба случая найдены; сравнение их значений даёт глобальные экстремумы.")
    return {"candidates": candidates, "minimum": minimum, "maximum": maximum}


if __name__ == "__main__":
    main()
