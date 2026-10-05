from scipy.optimize import linprog

c = [1, 2, 3, 1]

A = [
    [1, 2, 1, 0],
    [-1, 0, 0, -1]
]
b = [7, -2]

A_eq = [[0, 1, 1, 1]]
b_eq = [6]

res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=(0, None)
)

print("x1 =", res.x[0])
print("x2 =", res.x[1])
print("x3 =", res.x[2])
print("x4 =", res.x[3])
print("Z =", res.fun)