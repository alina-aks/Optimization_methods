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


def canonical(c, A, signs, b):
    A = [row[:] for row in A]
    c = c[:]
    signs = signs[:]
    b = b[:]

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

            if signs[i] == "<=":
                A[i][-1] = 1
            else:
                A[i][-1] = -1

    return c, A, b


def solve(c, A, signs, b, direction):
    original_c = c[:]
    n = len(c)

    if direction == "max":
        c = [-x for x in c]

    c, A, b = canonical(c, A, signs, b)

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

    print("\nРешение вспомогательной задачи...")

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

    for i in range(len(basis)):
        if basis[i] in artificial:
            for j in range(canonical_n):
                if abs(table[i][j]) > EPS:
                    pivot(table, i, j)
                    basis[i] = j
                    break

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

    print("\nПереход к основной задаче...")
    print("Решение основной задачи...")

    if not simplex(table, basis):
        print("Целевая функция не ограничена")
        return

    x = [0] * canonical_n

    for i in range(len(basis)):
        if basis[i] < canonical_n:
            x[basis[i]] = table[i][-1]

    answer = x[:n]

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


n = int(input("Количество переменных: "))
m = int(input("Количество ограничений: "))

direction = input("min или max: ").strip().lower()

print("Коэффициенты целевой функции:")
c = list(map(float, input().split()))

A = []
signs = []
b = []

for i in range(m):
    print(f"\nОграничение {i + 1}")

    row = list(
        map(float, input("Коэффициенты: ").split())
    )

    sign = input("Знак (<=, =, >=): ").strip()
    right = float(input("Правая часть: "))

    A.append(row)
    signs.append(sign)
    b.append(right)

solve(c, A, signs, b, direction)