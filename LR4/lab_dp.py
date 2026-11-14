from itertools import product
from pathlib import Path
from time import perf_counter
import json
import math
import sys

import numpy as np

EPS = 1e-9

DATA = {
    "asset_names": ["ЦБ1", "ЦБ2", "Депозит"],
    "initial": [100, 800, 400, 600],
    "packages": [25, 200, 100],
    "minimum": [30, 150, 100],
    "fees": [0.04, 0.07, 0.05],
    "probabilities": [[0.60, 0.30, 0.10],
                      [0.30, 0.20, 0.50],
                      [0.40, 0.40, 0.20]],
    "returns": [[[1.20, 1.10, 1.07], [1.05, 1.02, 1.03], [0.80, 0.95, 1.00]],
                [[1.40, 1.15, 1.01], [1.05, 1.00, 1.00], [0.60, 0.90, 1.00]],
                [[1.15, 1.12, 1.05], [1.05, 1.01, 1.01], [0.70, 0.94, 1.00]]],
    "scenario_names": ["Б", "Н", "О"],
}


class PortfolioDP:

    def __init__(self, data):
        self.data = data
        self.q = np.asarray(data["packages"], dtype=float)
        self.minimum = np.asarray(data["minimum"], dtype=float)
        self.fees = np.asarray(data["fees"], dtype=float)
        self.initial = np.asarray(data["initial"], dtype=float)
        self.p = [np.asarray(row, dtype=float) for row in data["probabilities"]]
        self.r = [np.asarray(row, dtype=float) for row in data["returns"]]
        self.n = len(self.q)
        self.steps = len(self.p)
        self.validate()
        self.cache = {}
        self.states_count = 0
        self.actions_count = 0
        self.last_states_count = 0
        self.sale_variants_count = 0
        self.mean_last = self.p[-1] @ self.r[-1]
        self.fast_last = bool(np.all(self.mean_last > 1 - self.fees + EPS)
                              and np.all(self.mean_last < 1 + self.fees - EPS))

    def validate(self):
        if self.n == 0 or self.steps == 0:
            raise ValueError("Нужны хотя бы один актив и один этап")
        if self.initial.shape != (self.n + 1,):
            raise ValueError("Неверный размер начального состояния")
        if self.minimum.shape != (self.n,) or self.fees.shape != (self.n,):
            raise ValueError("Неверные размеры минимумов или комиссий")
        if len(self.r) != self.steps:
            raise ValueError("Число этапов в таблицах не совпадает")
        arrays = [self.initial, self.q, self.minimum, self.fees] + self.p + self.r
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("Исходные данные должны быть конечными числами")
        if np.any(self.initial < 0) or np.any(self.minimum < 0):
            raise ValueError("Отрицательные средства или минимумы недопустимы")
        if np.any(self.q <= 0) or np.any(self.fees < 0) or np.any(self.fees >= 1):
            raise ValueError("Неверные пакеты или комиссии")
        for probabilities, coefficients in zip(self.p, self.r):
            if probabilities.ndim != 1 or len(probabilities) == 0:
                raise ValueError("Не заданы вероятности сценариев")
            if np.any(probabilities < 0) or abs(probabilities.sum() - 1) > EPS:
                raise ValueError("Вероятности должны быть неотрицательными с суммой 1")
            if coefficients.shape != (len(probabilities), self.n):
                raise ValueError("Неполная таблица коэффициентов")
            if np.any(coefficients < 0):
                raise ValueError("Коэффициенты стоимости не могут быть отрицательными")

    def actions(self, state):
        state = np.asarray(state, dtype=float)
        lower = np.ceil((self.minimum - state[:-1] - EPS) / self.q).astype(int)
        max_sale_cash = np.maximum(0, -lower) * self.q * (1 - self.fees)
        upper = np.floor((state[-1] + max_sale_cash.sum() - max_sale_cash + EPS)
                         / (self.q * (1 + self.fees))).astype(int)
        if np.any(lower > upper):
            return np.empty((0, self.n), dtype=int), np.empty((0, self.n + 1))
        grids = np.meshgrid(*(np.arange(lower[i], upper[i] + 1)
                              for i in range(self.n)), indexing="ij")
        controls = np.stack(grids, axis=-1).reshape(-1, self.n)
        money = state[-1] - np.sum(self.q * (controls + self.fees * abs(controls)), axis=1)
        allowed = money >= -EPS
        controls = controls[allowed]
        money = np.maximum(0, money[allowed])
        after = np.column_stack((state[:-1] + controls * self.q, money))
        self.actions_count += len(controls)
        return controls, after

    def trade(self, state, action):
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=int)
        after = np.r_[state[:-1] + self.q * action,
                      state[-1] - np.sum(self.q * (action + self.fees * abs(action)))]
        if np.any(after[:-1] < self.minimum - EPS) or after[-1] < -EPS:
            raise ValueError("Управление нарушает ограничения")
        after[-1] = max(0, after[-1])
        return after

    def transition(self, stage, after, scenario):
        result = np.array(after, dtype=float, copy=True)
        result[..., :-1] *= self.r[stage][scenario]
        return result

    def last_values(self, states):
        self.last_states_count += len(states)
        if not self.fast_last:
            return np.array([self.solve(self.steps - 1, s)[0] for s in states])
        buy = np.maximum(0, np.ceil((self.minimum - states[:, :-1] - EPS)
                                   / self.q)).astype(int)
        money = states[:, -1] - np.sum(buy * self.q * (1 + self.fees), axis=1)
        values = (states[:, :-1] + buy * self.q) @ self.mean_last + money
        shortage_rows = np.flatnonzero(money < -EPS)
        if len(shortage_rows) == 0:
            return values
        shortage = -money[shortage_rows]
        available = np.maximum(0, np.floor((states[shortage_rows, :-1]
                                           - self.minimum + EPS) / self.q)).astype(int)
        bounds = np.ceil(shortage[:, None] / (self.q * (1 - self.fees))).astype(int)
        available = np.minimum(available, bounds)
        loss_per_package = self.q * (self.mean_last - 1 + self.fees)
        best_loss = np.full(len(shortage_rows), np.inf)
        for sale in product(*(range(v + 1) for v in available.max(axis=0))):
            sale = np.array(sale)
            self.sale_variants_count += 1
            proceeds = np.sum(sale * self.q * (1 - self.fees))
            loss = float(sale @ loss_per_package)
            allowed = (proceeds >= shortage - EPS) & np.all(sale <= available, axis=1)
            best_loss[allowed] = np.minimum(best_loss[allowed], loss)
        values[shortage_rows] -= best_loss
        return values

    def action_values(self, stage, after):
        if stage == self.steps - 1:
            return after[:, :-1] @ self.mean_last + after[:, -1]

        values = np.zeros(len(after))

        for scenario, probability in enumerate(self.p[stage]):
            if probability == 0:
                continue

            next_states = self.transition(stage, after, scenario)

            if stage + 1 == self.steps - 1:
                continuation = self.last_values(next_states)
            else:
                continuation = np.array([
                    self.solve(stage + 1, s)[0]
                    for s in next_states
                ])

            values += probability * continuation

        return values

    def solve(self, stage=0, state=None):
        if state is None:
            state = self.initial
        if stage == self.steps:
            return float(np.sum(state)), ()
        key = (stage, tuple(float(x) for x in state))
        if key in self.cache:
            return self.cache[key]
        self.states_count += 1
        controls, after = self.actions(state)
        if len(controls) == 0:
            answer = (-math.inf, ())
        else:
            values = self.action_values(stage, after)
            best = float(np.max(values))
            optimal = () if not math.isfinite(best) else tuple(
                tuple(int(x) for x in u) for u in controls[np.abs(values - best) <= EPS])
            answer = (best, optimal)
        self.cache[key] = answer
        return answer

    def table(self, stage, state):
        controls, after = self.actions(state)
        values = self.action_values(stage, after)
        best = float(np.max(values)) if len(values) else -math.inf
        return controls, after, values, best


def number(value):
    return f"{value:.8f}".rstrip("0").rstrip(".").replace(".", ",")


def operation_text(dp, stage, action):
    parts = []
    for i, count in enumerate(action):
        if count:
            word = "покупка" if count > 0 else "продажа"
            name = dp.data.get("asset_names", [f"Актив {i + 1}" for i in range(dp.n)])[i]
            parts.append(f"{word} {name} на {number(abs(count) * dp.q[i])} д.е.")
    commission = float(np.sum(abs(np.array(action)) * dp.q * dp.fees))
    return f"Этап {stage + 1}: " + ("; ".join(parts) if parts else "без сделок") + f"; комиссия {number(commission)} д.е."


def strategy_tree(dp, first_action=None):
    nodes, outcomes = [], []
    initial_capital = float(dp.initial.sum())
    labels = dp.data.get("scenario_names", [])

    def visit(stage, state, probability, history, operations, commissions):
        if stage == dp.steps:
            capital = float(np.sum(state))
            profit = capital - initial_capital
            outcomes.append({"history": history, "probability": probability,
                             "final_state": list(map(float, state)), "operations": operations,
                             "commissions": commissions, "total_commission": sum(commissions),
                             "capital": capital, "profit": profit,
                             "return_percent": 100 * profit / initial_capital if initial_capital else None})
            return
        bellman, optimal = dp.solve(stage, state)
        if not optimal:
            raise ValueError("Для достигнутого состояния нет допустимого продолжения")
        action = tuple(first_action) if stage == 0 and first_action is not None else optimal[0]
        after = dp.trade(state, action)
        commission = float(np.sum(abs(np.array(action)) * dp.q * dp.fees))
        nodes.append({"id": "S_" + (history or "0"), "stage": stage + 1,
                      "history": history or "-", "probability": probability,
                      "state": list(map(float, state)), "chosen_action": list(action),
                      "optimal_actions": [list(u) for u in optimal],
                      "after_trade": list(map(float, after)), "commission": commission,
                      "bellman": bellman})
        for scenario, p in enumerate(dp.p[stage]):
            if p == 0:
                continue
            label = labels[scenario] if scenario < len(labels) else str(scenario + 1)
            visit(stage + 1, dp.transition(stage, after, scenario), probability * float(p),
                  history + label, operations + [operation_text(dp, stage, action)],
                  commissions + [commission])

    visit(0, dp.initial, 1.0, "", [], [])
    expected = sum(row["probability"] * row["capital"] for row in outcomes)
    summary = {"capital0": initial_capital, "expected_capital": expected,
               "expected_profit": expected - initial_capital,
               "expected_return_percent": 100 * (expected - initial_capital) / initial_capital if initial_capital else None,
               "loss_probability": sum(row["probability"] for row in outcomes if row["profit"] < -EPS),
               "minimum_profit": min(row["profit"] for row in outcomes),
               "maximum_profit": max(row["profit"] for row in outcomes),
               "total_probability": sum(row["probability"] for row in outcomes)}
    assert abs(summary["total_probability"] - 1) <= EPS
    return nodes, outcomes, summary


def full_policy_values(dp, stage=0, state=None):
    if state is None:
        state = dp.initial
    if stage == dp.steps:
        return [(float(np.sum(state)), ())]
    result = []
    controls, after_states = dp.actions(state)
    for action, after in zip(controls, after_states):
        branches = [(float(p), full_policy_values(dp, stage + 1, dp.transition(stage, after, j)))
                    for j, p in enumerate(dp.p[stage]) if p > 0]
        for choices in product(*(branch[1] for branch in branches)):
            value = sum(branches[j][0] * choices[j][0] for j in range(len(branches)))
            result.append((value, tuple(int(x) for x in action)))
    return result


def small_data(initial, packages, minimum, fees, stages):
    return {"initial": initial, "packages": packages, "minimum": minimum, "fees": fees,
            "probabilities": [[p for p, r in stage] for stage in stages],
            "returns": [[r for p, r in stage] for stage in stages]}


def run_tests():
    cases = [
        ("Нулевые свободные средства", small_data([4, 0], [1], [1], [.1], [[(1, [1.1])]]), 4.4, [(0,)]),
        ("Один этап", small_data([4, 3], [1], [1], [.1], [[(1, [1.5])]]), 9.8, [(2,)]),
        ("Нулевая доходность", small_data([4, 3], [1], [1], [.1], [[(1, [1])], [(1, [1])]]), 7, [(0,)]),
        ("Несколько оптимумов", small_data([4, 2], [1], [1], [0], [[(1, [1])]]), 6,
         [(-3,), (-2,), (-1,), (0,), (1,), (2,)]),
        ("Все свободные средства в один актив", small_data([4, 4, 2], [1, 1], [4, 4], [0, 0],
                                                         [[(1, [1.5, 1])]]), 13, [(2, 0)]),
        ("Случайный контрольный пример", small_data([4, 2], [1], [1], [.1],
                                                   [[(.5, [1.5]), (.5, [.5])],
                                                    [(.5, [1.6]), (.5, [.8])]]), 6.9, [(0,), (1,)]),
    ]
    tests = []
    for name, data, expected, expected_actions in cases:
        dp = PortfolioDP(data)
        value, optimal = dp.solve()
        brute = full_policy_values(dp)
        brute_value = max(row[0] for row in brute)
        brute_actions = set(row[1] for row in brute if abs(row[0] - brute_value) <= EPS)
        assert abs(value - expected) <= EPS and abs(value - brute_value) <= EPS
        assert set(optimal) == set(expected_actions) == brute_actions
        tests.append({"name": name, "input": data, "expected": expected, "actual": value,
                      "expected_actions": [list(u) for u in expected_actions],
                      "actual_actions": [list(u) for u in optimal],
                      "brute_force_value": brute_value, "policies_count": len(brute), "passed": True})
    bad_cases = [
        ("Отрицательные свободные средства", small_data([4, -1], [1], [1], [.1], [[(1, [1])]])),
        ("Неполная таблица", small_data([4, 4, 2], [1, 1], [4, 4], [0, 0], [[(1, [1.5])]])),
    ]
    for name, data in bad_cases:
        try:
            PortfolioDP(data)
        except ValueError as error:
            tests.append({"name": name, "input": data, "expected": "ValueError",
                          "actual": "ValueError", "message": str(error), "passed": True})
        else:
            raise AssertionError("Некорректные данные не отклонены")
    data = small_data([4, 0], [1], [4], [.1],
                      [[(1, [1]), (0, [0])], [(1, [1])]])
    value, _ = PortfolioDP(data).solve()
    assert abs(value - 4) <= EPS
    tests.append({"name": "Нулевая вероятность недопустимой ветви", "input": data,
                  "expected": 4, "actual": value, "passed": True})
    return tests


def build_report_data():
    dp = PortfolioDP(DATA)
    start = perf_counter()
    optimum, optimal = dp.solve()
    runtime = perf_counter() - start
    counters = {"states_count": dp.states_count, "actions_count": dp.actions_count,
                "last_stage_states_evaluated": dp.last_states_count,
                "last_stage_sale_vectors": dp.sale_variants_count}
    cache_bytes = sys.getsizeof(dp.cache)
    for key, value in dp.cache.items():
        cache_bytes += sys.getsizeof(key) + sys.getsizeof(key[1]) + sys.getsizeof(value)
        cache_bytes += sys.getsizeof(value[1]) + sum(sys.getsizeof(u) for u in value[1])
    nodes, outcomes, summary = strategy_tree(dp)
    assert abs(optimum - summary["expected_capital"]) <= EPS
    actions = []
    for node in nodes:
        controls, after, values, best = dp.table(node["stage"] - 1, node["state"])
        for action, state_after, value in zip(controls, after, values):
            actions.append({"node_id": node["id"], "stage": node["stage"], "history": node["history"],
                            "action": [int(x) for x in action], "after_trade": list(map(float, state_after)),
                            "commission": float(np.sum(abs(action) * dp.q * dp.fees)),
                            "q_value": float(value) if math.isfinite(value) else None,
                            "feasible_continuation": bool(math.isfinite(value)),
                            "is_optimal": bool(math.isfinite(value) and abs(value - best) <= EPS)})
    competitor_nodes, competitor_outcomes, competitor_summary = strategy_tree(dp, (19, 0, 1))
    competitor_nodes[0]["bellman"] = competitor_summary["expected_capital"]
    competitor_nodes[0]["optimal_actions"] = [[19, 0, 1]]
    return {"input": DATA, "optimum": optimum, "runtime": runtime, **counters,
            "cache_bytes_estimate": cache_bytes,
            "optimal_first_actions": [list(u) for u in optimal],
            "summary": summary, "nodes": nodes, "outcomes": outcomes, "actions": actions,
            "root_actions": [a for a in actions if a["stage"] == 1], "tests": run_tests(),
            "competitor": {"first_action": [19, 0, 1], "nodes": competitor_nodes,
                           "summary": competitor_summary}}


def main():
    result = build_report_data()
    path = Path(__file__).resolve().with_name("lab_results.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = result["summary"]
    print("Максимальный ожидаемый конечный капитал:", number(result["optimum"]))
    print("Ожидаемый чистый доход:", number(summary["expected_profit"]))
    print("Ожидаемая доходность, %:", number(summary["expected_return_percent"]))
    print("Вероятность убытка:", number(summary["loss_probability"]))
    print("Оптимальные действия по узлам:")
    for node in result["nodes"]:
        print(node["stage"], node["history"], node["chosen_action"], "V =", number(node["bellman"]))
    print("Время основного решения, с:", round(result["runtime"], 4))
    print("Состояний с сохраненными V и управлениями:", result["states_count"])
    print("Перебрано допустимых управлений:", result["actions_count"])
    print("Расчетов состояний последнего этапа:", result["last_stage_states_evaluated"])
    print("Проверок пройдено:", len(result["tests"]))


if __name__ == "__main__":
    main()
