EPS = 1e-9


def pivot(table, row, col):
    p = table[row][col]

    table[row] = [x / p for x in table[row]]

    for i in range(len(table)):
        if i != row:
            k = table[i][col]
            table[i] = [
                table[i][j] - k * table[row][j]
                for j in range(len(table[i]))
            ]


def simplex(table, basis):
    while True:
        last = table[-1]

        col = min(
            range(len(last) - 1),
            key=lambda j: last[j]
        )

        if last[col] >= -EPS:
            return True

        variants = []

        for i in range(len(table) - 1):
            if table[i][col] > EPS:
                variants.append(
                    (table[i][-1] / table[i][col], i)
                )

        if not variants:
            return False

        row = min(variants)[1]

        pivot(table, row, col)
        basis[row] = col


def canonical(c, A, signs, b, free_vars):
    A = [row[:] for row in A]
    c = c[:]
    signs = signs[:]
    b = b[:]
    negative_columns = {}

    for j in sorted(free_vars):
        negative_columns[j] = len(c)
        c.append(-c[j])

        for row in A:
            row.append(-row[j])

    for i in range(len(A)):
        if b[i] < 0:
            A[i] = [-x for x in A[i]]
            b[i] = -b[i]

            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    for i in range(len(A)):
        if signs[i] != "=":
            for row in A:
                row.append(0)

            c.append(0)
            A[i][-1] = 1 if signs[i] == "<=" else -1

    return c, A, b, negative_columns


def solve(c, A, signs, b, direction, free_vars=()):
    original_c = c[:]
    n = len(c)

    if direction == "max":
        c = [-x for x in c]

    c, A, b, negative_columns = canonical(c, A, signs, b, free_vars)

    canonical_n = len(c)

    print("\nКанонический вид:")

    for i in range(len(A)):
        print(A[i], "=", b[i])

    artificial = []
    basis = []

    for i in range(len(A)):
        for row in A:
            row.append(0)

        A[i][-1] = 1

        artificial.append(len(A[i]) - 1)
        basis.append(len(A[i]) - 1)

    table = []

    for i in range(len(A)):
        table.append(A[i][:] + [b[i]])

    objective = [0] * (len(A[0]) + 1)

    for j in artificial:
        objective[j] = 1

    for i in range(len(A)):
        objective = [
            objective[j] - table[i][j]
            for j in range(len(objective))
        ]

    table.append(objective)

    print("\nРешение вспомогательной задачи:")

    if not simplex(table, basis):
        print("Решений нет")
        return

    W = -table[-1][-1]

    if abs(W) < EPS:
        W = 0

    print("W' =", W)

    if W > EPS:
        print("Исходная задача не имеет решений")
        return

    i = 0

    while i < len(basis):
        if basis[i] not in artificial:
            i += 1
            continue

        if abs(table[i][-1]) > EPS:
            raise ArithmeticError("Искусственная переменная не равна нулю")

        table[i][-1] = 0.0
        col = None

        for j in range(canonical_n):
            if j not in basis and abs(table[i][j]) > EPS:
                col = j
                break

        if col is None:
            if any(abs(v) > EPS for v in table[i][:canonical_n]):
                raise ArithmeticError("Не удалось исключить искусственную переменную")

            table.pop(i)
            basis.pop(i)
            print("Удалена зависимая строка с нулевой искусственной переменной.")
        else:
            pivot(table, i, col)
            basis[i] = col
            i += 1

    new_table = []

    for i in range(len(table) - 1):
        new_table.append(
            table[i][:canonical_n] + [table[i][-1]]
        )

    table = new_table

    objective = c[:] + [0]

    for i in range(len(basis)):
        if basis[i] < canonical_n:
            k = objective[basis[i]]

            if abs(k) > EPS:
                objective = [
                    objective[j] - k * table[i][j]
                    for j in range(len(objective))
                ]

    table.append(objective)


    if not simplex(table, basis):
        print("Целевая функция не ограничена")
        return

    x = [0] * canonical_n

    for i in range(len(basis)):
        if basis[i] < canonical_n:
            x[basis[i]] = table[i][-1]

    answer = x[:n]

    for j, k in negative_columns.items():
        answer[j] -= x[k]

    Z = sum(
        original_c[i] * answer[i]
        for i in range(n)
    )

    print("\nОтвет:")

    for i in range(n):
        if abs(answer[i]) < EPS:
            answer[i] = 0

        print(f"x{i + 1} = {answer[i]:g}")

    print(f"Z = {Z:g}")

if __name__ == "__main__":
    n = int(input("Количество переменных: "))
    m = int(input("Количество ограничений: "))

    direction = input("min или max: ").strip().lower()

    if direction not in ("min", "max"):
        raise ValueError("Направление должно быть min или max")

    print("Коэффициенты целевой функции:")
    c = list(map(float, input().split()))

    free_vars = {
        int(value) - 1
        for value in input(
            "Номера свободных по знаку переменных: "
        ).split()
    }

    if any(j < 0 or j >= n for j in free_vars):
        raise ValueError("Номер свободной переменной должен быть от 1 до n")

    A = []
    signs = []
    b = []

    for i in range(m):
        print(f"\nОграничение {i + 1}")

        row = list(
            map(float, input("Коэффициенты: ").split())
        )

        sign = input("Знак (<=, =, >=): ").strip()
        if sign not in ("<=", "=", ">="):
            raise ValueError("Неверный знак ограничения")

        right = float(input("Правая часть: "))

        A.append(row)
        signs.append(sign)
        b.append(right)

    solve(c, A, signs, b, direction, free_vars)