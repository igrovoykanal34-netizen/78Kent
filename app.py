import numpy as np
import pandas as pd
import streamlit as st

EPS = 1e-9


# ───────────────────────────── Вспомогательное ─────────────────────────────

def fmt(x):
    """Красивый вывод числа: целые без .0, дробные до 4 знаков."""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def call_wide(fn, *args, **kwargs):
    """Растянуть виджет на всю ширину и в новых, и в старых версиях Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ───────────────────────────── Ядро расчёта ─────────────────────────────

def balance_problem(supply, demand, costs):
    s = [float(x) for x in supply]
    d = [float(x) for x in demand]
    c = [[float(x) for x in row] for row in costs]
    ts, td = sum(s), sum(d)
    fict_row = fict_col = None

    if abs(ts - td) < EPS:
        msg = "Задача закрытая: сумма запасов равна сумме потребностей."
    elif ts > td:
        diff = ts - td
        d.append(diff)
        for row in c:
            row.append(0.0)
        fict_col = len(d) - 1
        msg = (f"Задача открытая: запасы ({fmt(ts)}) больше потребностей ({fmt(td)}) на {fmt(diff)}. "
               f"Добавлен фиктивный потребитель B{len(d)} с нулевыми тарифами.")
    else:
        diff = td - ts
        s.append(diff)
        c.append([0.0] * len(d))
        fict_row = len(s) - 1
        msg = (f"Задача открытая: потребности ({fmt(td)}) больше запасов ({fmt(ts)}) на {fmt(diff)}. "
               f"Добавлен фиктивный поставщик A{len(s)} с нулевыми тарифами.")

    return np.array(s), np.array(d), np.array(c), fict_row, fict_col, msg


def find_cycle(basis, start_cell):
    nodes = list(basis) + [start_cell]

    def dfs(curr, path, look_row):
        if len(path) >= 4 and curr == start_cell:
            return path[:-1]
        for node in nodes:
            if node == curr:
                continue
            if look_row and node[0] == curr[0]:
                if node == start_cell and len(path) >= 4:
                    return path
                if node not in path:
                    res = dfs(node, path + [node], not look_row)
                    if res:
                        return res
            elif not look_row and node[1] == curr[1]:
                if node == start_cell and len(path) >= 4:
                    return path
                if node not in path:
                    res = dfs(node, path + [node], not look_row)
                    if res:
                        return res
        return None

    return dfs(start_cell, [start_cell], True)


def ensure_non_degenerate(basis, m, n, costs, prefer_last=False):
    """Дополняет базис нулевыми клетками до m+n-1: берём свободную клетку с наименьшим тарифом,
    не образующую цикл. При равных тарифах prefer_last=True берёт нижнюю (затем правую) клетку."""
    basis = set(basis)
    all_cells = [(r, c) for r in range(m) for c in range(n)]
    if prefer_last:
        all_cells.sort(key=lambda x: (costs[x[0], x[1]], -x[0], -x[1]))
    else:
        all_cells.sort(key=lambda x: costs[x[0], x[1]])
    while len(basis) < m + n - 1:
        added = False
        for cell in all_cells:
            if cell not in basis and find_cycle(basis, cell) is None:
                basis.add(cell)
                added = True
                break
        if not added:
            break
    return basis


def vogel_plan(supply, demand, costs, tie="first"):
    """Метод Фогеля (аппроксимации). Штраф = разность двух наименьших тарифов в строке/столбце.
    При одновременном обнулении строки и столбца вычёркивается только строка: столбец остаётся
    с нулевой потребностью, и в него позже ставится нулевая (базисная) поставка."""
    s = supply.copy()
    d = demand.copy()
    m, n = costs.shape
    alloc = np.zeros((m, n))
    basis = set()
    log = []
    rows, cols = set(range(m)), set(range(n))

    def put(i, j):
        val = min(s[i], d[j])
        alloc[i, j] = val
        s[i] -= val
        d[j] -= val
        basis.add((i, j))
        return val

    step = 0
    while rows and cols:
        step += 1
        if len(rows) == 1 or len(cols) == 1:
            # остался один пункт: распределяем всё, что осталось
            if len(rows) == 1:
                i = next(iter(rows))
                for j in sorted(cols, key=lambda j: (costs[i, j], j)):
                    val = put(i, j)
                    log.append({"step": step, "rp": {}, "cp": {}, "pick": "остаток", "cell": (i, j), "val": val})
            else:
                j = next(iter(cols))
                for i in sorted(rows, key=lambda i: (costs[i, j], i)):
                    val = put(i, j)
                    log.append({"step": step, "rp": {}, "cp": {}, "pick": "остаток", "cell": (i, j), "val": val})
            break

        rp, cp = {}, {}
        for i in rows:
            v = sorted(costs[i, j] for j in cols)
            rp[i] = float(v[1] - v[0])
        for j in cols:
            v = sorted(costs[i, j] for i in rows)
            cp[j] = float(v[1] - v[0])
        mx = max(list(rp.values()) + list(cp.values()))
        cands = [("A", i) for i in sorted(rows) if abs(rp[i] - mx) < EPS] + \
                [("B", j) for j in sorted(cols) if abs(cp[j] - mx) < EPS]

        def line_min(k):
            kind, idx = k
            if kind == "A":
                return min(costs[idx, j] for j in cols)
            return min(costs[i, idx] for i in rows)

        if tie == "first":
            kind, idx = cands[0]
        elif tie == "mincost":
            kind, idx = min(enumerate(cands), key=lambda t: (line_min(t[1]), t[0]))[1]
        else:
            kind, idx = cands[-1]

        if kind == "A":
            i = idx
            j = min(cols, key=lambda j: (costs[i, j], -min(s[i], d[j]), j))
        else:
            j = idx
            i = min(rows, key=lambda i: (costs[i, j], -min(s[i], d[j]), i))
        val = put(i, j)
        log.append({"step": step, "rp": dict(rp), "cp": dict(cp),
                    "pick": f"{'A' if kind == 'A' else 'B'}{idx + 1} (штраф {fmt(mx)})",
                    "cell": (i, j), "val": val})
        if s[i] <= EPS:
            rows.discard(i)
        else:
            cols.discard(j)

    basis = ensure_non_degenerate(basis, m, n, costs, prefer_last=True)
    return alloc, basis, log


def nw_plan(supply, demand, costs):
    """Метод северо-западного угла. Если строка и столбец закрылись одновременно,
    следующая клетка (ниже) получает нулевую поставку и входит в базис."""
    s = supply.copy()
    d = demand.copy()
    m, n = costs.shape
    alloc = np.zeros((m, n))
    basis = set()
    log = []
    i = j = 0
    while i < m and j < n:
        val = min(s[i], d[j])
        alloc[i, j] = val
        basis.add((i, j))
        s[i] -= val
        d[j] -= val
        log.append({"cell": (i, j), "val": val})
        if s[i] <= EPS and d[j] <= EPS and not (i == m - 1 and j == n - 1):
            if i + 1 < m:
                i += 1
            else:
                j += 1
        elif s[i] <= EPS:
            i += 1
        else:
            j += 1
    basis = ensure_non_degenerate(basis, m, n, costs)
    return alloc, basis, []


def min_line_plan(supply, demand, costs, by, fict_row, fict_col):
    """Метод наименьшего элемента по строкам (by='row') или по столбцам (by='col').
    Идём строка за строкой (столбец за столбцом) и в каждой заполняем самые дешёвые клетки,
    пока не исчерпаем запас строки (потребность столбца). Фиктивные пункты заполняются в последнюю очередь."""
    s = supply.copy()
    d = demand.copy()
    m, n = costs.shape
    alloc = np.zeros((m, n))
    basis = set()

    def fict(i, j):
        return i == fict_row or j == fict_col

    def put(i, j):
        val = min(s[i], d[j])
        alloc[i, j] = val
        s[i] -= val
        d[j] -= val
        basis.add((i, j))

    if by == "row":
        for i in range(m):
            while s[i] > EPS:
                open_cols = [j for j in range(n) if d[j] > EPS]
                if not open_cols:
                    break
                j = min(open_cols, key=lambda j: (fict(i, j), costs[i, j], j))
                put(i, j)
    else:
        for j in range(n):
            while d[j] > EPS:
                open_rows = [i for i in range(m) if s[i] > EPS]
                if not open_rows:
                    break
                i = min(open_rows, key=lambda i: (fict(i, j), costs[i, j], i))
                put(i, j)

    # вырождение: недостающие нулевые клетки — с наименьшим тарифом, без циклов
    basis = ensure_non_degenerate(basis, m, n, costs, prefer_last=True)
    return alloc, basis, []


def double_pref_plan(supply, demand, costs, fict_row, fict_col):
    """Метод двойного предпочтения: в каждой строке и каждом столбце отмечается клетка с наименьшим
    тарифом (при равных — все такие клетки). Сначала заполняются клетки с двумя отметками, затем с одной,
    затем остальные — в каждой группе по возрастанию тарифа. Фиктивные пункты не отмечаются и идут последними."""
    s = supply.copy()
    d = demand.copy()
    m, n = costs.shape
    alloc = np.zeros((m, n))
    basis = set()

    def fict(i, j):
        return i == fict_row or j == fict_col

    marks = np.zeros((m, n), dtype=int)
    for i in range(m):
        if i == fict_row:
            continue
        real = [j for j in range(n) if not fict(i, j)]
        lo = min(costs[i, j] for j in real)
        for j in real:
            if abs(costs[i, j] - lo) < EPS:
                marks[i, j] += 1
    for j in range(n):
        if j == fict_col:
            continue
        real = [i for i in range(m) if not fict(i, j)]
        lo = min(costs[i, j] for i in real)
        for i in real:
            if abs(costs[i, j] - lo) < EPS:
                marks[i, j] += 1

    cells = [(i, j) for i in range(m) for j in range(n)]
    order = sorted(cells, key=lambda x: (-marks[x] if not fict(*x) else 0, fict(*x), costs[x], x))
    for i, j in order:
        if s[i] > EPS and d[j] > EPS:
            val = min(s[i], d[j])
            alloc[i, j] = val
            s[i] -= val
            d[j] -= val
            basis.add((i, j))

    basis = ensure_non_degenerate(basis, m, n, costs, prefer_last=True)
    return alloc, basis, []


def initial_plan(supply, demand, costs, method, fict_row, fict_col, tie="first"):
    if method == "vogel":
        return vogel_plan(supply, demand, costs, tie)
    if method == "nw":
        return nw_plan(supply, demand, costs)
    if method == "dp":
        return double_pref_plan(supply, demand, costs, fict_row, fict_col)
    return min_line_plan(supply, demand, costs, "col" if method == "min_col" else "row", fict_row, fict_col)


def calc_potentials(basis, costs):
    m, n = costs.shape
    u = [None] * m
    v = [None] * n
    u[0] = 0.0
    for _ in range(m + n):
        changed = False
        for r, c in basis:
            if u[r] is not None and v[c] is None:
                v[c] = float(costs[r, c] - u[r])
                changed = True
            elif v[c] is not None and u[r] is None:
                u[r] = float(costs[r, c] - v[c])
                changed = True
        if not changed:
            for i in range(m):
                if u[i] is None:
                    u[i] = 0.0
                    changed = True
                    break
            if not changed:
                break
    return u, v


def solve(raw_s, raw_d, raw_c, method="nw", tie="first", max_iter=200):
    s, d, c, fict_row, fict_col, msg = balance_problem(raw_s, raw_d, raw_c)
    alloc, basis, log = initial_plan(s, d, c, method, fict_row, fict_col, tie)
    init = {"alloc": alloc.copy(), "basis": set(basis), "cost": float(np.sum(alloc * c)), "log": log}
    m, n = c.shape
    steps = []
    optimal = False

    for it in range(1, max_iter + 1):
        basis = ensure_non_degenerate(basis, m, n, c)
        u, v = calc_potentials(basis, c)

        deltas = {}
        for r in range(m):
            for col in range(n):
                if (r, col) not in basis:
                    deltas[(r, col)] = round(float(c[r, col] - (u[r] + v[col])), 6)

        # В базис вводится клетка с отрицательной оценкой, имеющая НАИМЕНЬШИЙ ТАРИФ
        # (при равных тарифах — самая верхняя, затем самая левая).
        negatives = [x for x, dv in deltas.items() if dv < -1e-7]
        entering = min(negatives, key=lambda x: (c[x], x[0], x[1])) if negatives else None
        min_delta = deltas[entering] if entering is not None else 0.0

        step = {"iter": it, "alloc": alloc.copy(), "basis": set(basis), "u": u, "v": v,
                "deltas": deltas, "cost": float(np.sum(alloc * c)),
                "entering": None, "cycle": None}

        if entering is None:
            optimal = True
            steps.append(step)
            break

        cycle = find_cycle(basis, entering)
        if cycle is None:
            raise RuntimeError("Не удалось построить контур пересчёта.")
        minus = [cycle[k] for k in range(1, len(cycle), 2)]
        theta = min(alloc[x] for x in minus)
        # при равных минимумах выходит самая левая клетка (затем самая верхняя)
        ties = [x for x in minus if abs(alloc[x] - theta) < 1e-9]
        leaving = min(ties, key=lambda x: (x[1], x[0]))
        step.update(entering=entering, delta=min_delta, cycle=cycle, theta=float(theta), leaving=leaving)
        steps.append(step)

        for idx, cell in enumerate(cycle):
            alloc[cell] += theta if idx % 2 == 0 else -theta
        alloc[np.abs(alloc) < EPS] = 0.0
        basis.remove(leaving)
        basis.add(entering)

    return {"s": s, "d": d, "c": c, "fict_row": fict_row, "fict_col": fict_col, "msg": msg,
            "init": init, "steps": steps, "optimal": optimal}


# ───────────────────────────── Оформление ─────────────────────────────

CSS = """
<style>
.hero{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#0ea5e9 100%);color:#fff;
      padding:24px 30px;border-radius:18px;margin:0 0 18px;box-shadow:0 10px 30px rgba(79,70,229,.28);}
.hero-title{font-size:30px;font-weight:800;line-height:1.2;}
.hero-sub{font-size:15px;opacity:.92;margin-top:6px;}

.tp-wrap{overflow-x:auto;background:#fff;border-radius:14px;padding:14px;margin:6px 0 10px;
         border:1px solid #e2e8f0;box-shadow:0 4px 18px rgba(15,23,42,.10);}
.tp-table{border-collapse:collapse;width:100%;min-width:660px;color:#0f172a;
          font-family:"Segoe UI",Roboto,Arial,sans-serif;}
.tp-table th,.tp-table td{border:1.5px solid #475569;text-align:center;padding:0;}
.tp-table th{background:#eef2ff;font-weight:700;font-size:15px;padding:10px 8px;color:#1e1b4b;}
.tp-table th.tp-top{background:#e0e7ff;}
.tp-table th.tp-rowh{font-size:20px;padding:6px 12px;}
.tp-f{display:block;font-size:10px;font-weight:600;color:#7c3aed;}

.tp-cell{position:relative;height:80px;min-width:100px;}
.tp-c{position:absolute;top:6px;right:9px;font-size:15px;font-weight:600;color:#475569;}
.tp-x{position:absolute;left:11px;bottom:8px;font-size:21px;font-weight:800;color:#1e3a8a;}
.tp-x.zero{color:#94a3b8;font-weight:700;}
.tp-d{position:absolute;top:6px;left:9px;font-size:16px;font-weight:700;}
.tp-d.pos{color:#16a34a;}
.tp-d.neg{color:#dc2626;}
.tp-d.zer{color:#d97706;border:1.5px solid #d97706;border-radius:50%;min-width:20px;height:20px;
          line-height:17px;font-size:12px;text-align:center;padding:0 1px;}
.tp-sign{position:absolute;right:9px;bottom:8px;width:24px;height:24px;border-radius:50%;
         font-weight:800;font-size:16px;line-height:24px;text-align:center;color:#fff;}
.sg-plus{background:#16a34a;}
.sg-minus{background:#dc2626;}
td.basis{background:#ecfdf5;}
td.cyc-plus{background:#dcfce7;box-shadow:inset 0 0 0 2.5px #16a34a;}
td.cyc-minus{background:#fee2e2;box-shadow:inset 0 0 0 2.5px #dc2626;}

.tp-total{background:#f8fafc;font-weight:700;font-size:17px;padding:10px 8px;}
.tp-pot{background:#fffbeb;font-weight:800;font-size:17px;color:#92400e;padding:10px 8px;}
.tp-empty{background:#f8fafc;}
.p-line{font-size:24px;font-weight:700;margin:4px 2px 16px;}
.legend{font-size:13px;opacity:.85;margin:0 2px 12px;}
</style>
"""


def render_table(costs, alloc, basis, supply, demand, u=None, v=None, deltas=None, cycle=None,
                 fict_row=None, fict_col=None, total=None, unit="т·км", show_values=False):
    m, n = costs.shape
    cyc = {}
    if cycle:
        for idx, cell in enumerate(cycle):
            cyc[cell] = "+" if idx % 2 == 0 else "-"

    h = ['<div class="tp-wrap"><table class="tp-table"><thead><tr>',
         '<th rowspan="2">Грузообразующие<br>пункты</th>',
         f'<th colspan="{n}" class="tp-top">Грузопоглощающие пункты</th>',
         '<th rowspan="2">Итого</th><th rowspan="2">U<sub>i</sub></th></tr><tr>']
    for j in range(n):
        fl = '<span class="tp-f">фикт.</span>' if j == fict_col else ""
        h.append(f"<th>B<sub>{j + 1}</sub>{fl}</th>")
    h.append("</tr></thead><tbody>")

    for i in range(m):
        fl = '<span class="tp-f">фикт.</span>' if i == fict_row else ""
        h.append(f'<tr><th class="tp-rowh">A<sub>{i + 1}</sub>{fl}</th>')
        for j in range(n):
            cell = (i, j)
            is_b = cell in basis
            sign = cyc.get(cell)
            cls = "basis" if is_b else ""
            if sign == "+":
                cls = "cyc-plus"
            elif sign == "-":
                cls = "cyc-minus"

            inner = f'<span class="tp-c">{fmt(costs[i, j])}</span>'
            if is_b:
                val = alloc[i, j]
                zero = " zero" if val < EPS else ""
                inner += f'<span class="tp-x{zero}">{fmt(val)}</span>'
            elif deltas is not None and cell in deltas:
                dv = deltas[cell]
                if dv < -EPS:
                    inner += f'<span class="tp-d neg" title="Δ = {fmt(dv)}">{fmt(dv)}</span>'
                elif dv <= EPS:
                    inner += '<span class="tp-d zer" title="Δ = 0">0</span>'
                else:
                    txt = f"+{fmt(dv)}" if show_values else "+"
                    inner += f'<span class="tp-d pos" title="Δ = {fmt(dv)}">{txt}</span>'
            if sign == "+":
                inner += '<span class="tp-sign sg-plus">+</span>'
            elif sign == "-":
                inner += '<span class="tp-sign sg-minus">−</span>'

            h.append(f'<td class="{cls}"><div class="tp-cell">{inner}</div></td>')

        u_txt = fmt(u[i]) if u is not None and u[i] is not None else "—"
        h.append(f'<td class="tp-total">{fmt(supply[i])}</td><td class="tp-pot">{u_txt}</td></tr>')

    h.append('<tr><th>Итого</th>')
    for j in range(n):
        h.append(f'<td class="tp-total">{fmt(demand[j])}</td>')
    h.append(f'<td class="tp-total">{fmt(np.sum(supply))}</td><td class="tp-empty"></td></tr>')

    h.append('<tr><th>V<sub>j</sub></th>')
    for j in range(n):
        v_txt = fmt(v[j]) if v is not None and v[j] is not None else "—"
        h.append(f'<td class="tp-pot">{v_txt}</td>')
    h.append('<td class="tp-empty"></td><td class="tp-empty"></td></tr></tbody></table></div>')

    if total is not None:
        h.append(f'<div class="p-line">P = {fmt(total)} {unit}</div>')
    return "".join(h)


LEGEND = ('<div class="legend">В клетке: <b>тариф</b> — справа сверху, <b>объём перевозки</b> — слева снизу '
          '(зелёный фон — базисная клетка). Слева сверху у свободных клеток: <b style="color:#16a34a">+</b> — Δ &gt; 0, '
          '<b style="color:#d97706">0</b> — Δ = 0, красное число — Δ &lt; 0 (план можно улучшить).</div>')


# ───────────────────────────── Интерфейс ─────────────────────────────

EXAMPLE_C = [[8, 11, 15, 4, 7], [11, 10, 14, 6, 9], [6, 5, 9, 11, 8], [3, 6, 10, 12, 15]]
EXAMPLE_S = [200, 175, 75, 300]
EXAMPLE_D = [100, 125, 150, 175, 200]


def build_grid(m, n):
    arr = np.full((m + 1, n + 1), np.nan)
    for i in range(m):
        for j in range(n):
            arr[i, j] = EXAMPLE_C[i][j] if i < 4 and j < 5 else 10.0
        arr[i, n] = EXAMPLE_S[i] if i < 4 else 100.0
    for j in range(n):
        arr[m, j] = EXAMPLE_D[j] if j < 5 else 100.0
    return pd.DataFrame(
        arr,
        index=[f"A{i + 1}" for i in range(m)] + ["Потребности Bj"],
        columns=[f"B{j + 1}" for j in range(n)] + ["Запасы Ai"],
    )


METHOD_NAMES = {"vogel": "Фогеля (аппроксимации)", "nw": "северо-западного угла",
                "min_row": "наименьшего элемента по строкам", "min_col": "наименьшего элемента по столбцам",
                "dp": "двойного предпочтения"}


def vogel_log_table(log):
    rows = []
    for e in log:
        if e["rp"] or e["cp"]:
            rp = ", ".join(f"A{i + 1}: {fmt(v)}" for i, v in sorted(e["rp"].items()))
            cp = ", ".join(f"B{j + 1}: {fmt(v)}" for j, v in sorted(e["cp"].items()))
        else:
            rp = cp = "—"
        r, col = e["cell"]
        rows.append({"Шаг": e["step"], "Штрафы строк": rp, "Штрафы столбцов": cp,
                     "Выбрана линия": e["pick"], "Клетка": f"A{r + 1}B{col + 1}", "Поставка": fmt(e["val"])})
    df = pd.DataFrame(rows)
    return df.set_index("Шаг")


def show_results(raw_s, raw_d, raw_c, method, unit, show_vals, tie="first"):
    res = solve(raw_s, raw_d, raw_c, method, tie)
    c, s, d = res["c"], res["s"], res["d"]
    fr, fc = res["fict_row"], res["fict_col"]
    kw = dict(fict_row=fr, fict_col=fc, unit=unit, show_values=show_vals)
    steps = res["steps"]
    method_name = METHOD_NAMES[method]

    st.divider()
    st.markdown("### 2. Баланс и опорный план")
    st.info(res["msg"])
    init = res["init"]
    st.markdown(f"Опорный план построен **методом {method_name}**. Начальная стоимость: "
                f"**P₀ = {fmt(init['cost'])} {unit}**")
    if init["log"] and method == "vogel":
        with st.expander("Шаги метода Фогеля (штрафы и выбор клеток)", expanded=False):
            st.dataframe(vogel_log_table(init["log"]))
    st.markdown(render_table(c, init["alloc"], init["basis"], s, d, **kw), unsafe_allow_html=True)

    st.markdown("### 3. Оптимизация методом потенциалов")
    st.markdown(LEGEND, unsafe_allow_html=True)
    for step in steps:
        last = step["entering"] is None
        title = f"Итерация {step['iter']} — оптимальный план" if last else \
            f"Итерация {step['iter']} · P = {fmt(step['cost'])} {unit}"
        with st.expander(title, expanded=True):
            if last:
                st.markdown(render_table(c, step["alloc"], step["basis"], s, d, step["u"], step["v"],
                                         step["deltas"], None, total=step["cost"], **kw),
                            unsafe_allow_html=True)
                st.success("Все оценки свободных клеток Δ ≥ 0 — план оптимален.")
            else:
                er, ec = step["entering"]
                st.markdown(f"Минимальная оценка: **Δ(A{er + 1}, B{ec + 1}) = {fmt(step['delta'])} < 0**. "
                            f"Вводим клетку A{er + 1}B{ec + 1} в базис и строим контур пересчёта:")
                st.markdown(render_table(c, step["alloc"], step["basis"], s, d, step["u"], step["v"],
                                         step["deltas"], step["cycle"], **kw), unsafe_allow_html=True)
                path = " → ".join(f"{'+' if k % 2 == 0 else '−'}A{r + 1}B{col + 1}"
                                  for k, (r, col) in enumerate(step["cycle"]))
                lr, lc = step["leaving"]
                st.warning(f"**Контур:** {path}  \n"
                           f"**θ = {fmt(step['theta'])}** (минимум среди клеток со знаком «−»), "
                           f"клетка A{lr + 1}B{lc + 1} выходит из базиса.")

    st.divider()
    st.markdown("### 4. Итог")
    if not res["optimal"]:
        st.warning("Достигнут лимит итераций, план может быть неоптимальным.")
    last = steps[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Начальный план P₀", f"{fmt(init['cost'])} {unit}")
    c2.metric("Оптимальный план P", f"{fmt(last['cost'])} {unit}")
    c3.metric("Итераций улучшения", len(steps) - 1)

    st.markdown(render_table(c, last["alloc"], last["basis"], s, d, last["u"], last["v"],
                             last["deltas"], None, total=last["cost"], **kw), unsafe_allow_html=True)
    if any(abs(x) <= EPS for x in last["deltas"].values()):
        st.info("Есть свободные клетки с Δ = 0 — существует несколько оптимальных планов с одинаковым P.")

    routes = []
    for r in range(c.shape[0]):
        for col in range(c.shape[1]):
            if last["alloc"][r, col] > EPS:
                routes.append({
                    "Маршрут": f"A{r + 1}{' (фикт.)' if r == fr else ''} → "
                               f"B{col + 1}{' (фикт.)' if col == fc else ''}",
                    "Объём": fmt(last["alloc"][r, col]),
                    "Тариф": fmt(c[r, col]),
                    "Стоимость": fmt(last["alloc"][r, col] * c[r, col]),
                })
    if routes:
        df = pd.DataFrame(routes)
        df.index = range(1, len(df) + 1)
        st.markdown("**План перевозок:**")
        st.table(df)


def main():
    st.set_page_config(page_title="Транспортная задача — метод потенциалов", page_icon="🚛", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-title">🚛 Транспортная задача</div>'
                '<div class="hero-sub">Метод потенциалов: баланс → опорный план → потенциалы Uᵢ, Vⱼ → '
                'контуры пересчёта → оптимальный план</div></div>', unsafe_allow_html=True)

    st.markdown("### 1. Исходные данные")
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1])
    m = int(c1.number_input("Грузообразующие пункты (A)", 2, 10, 4, 1))
    n = int(c2.number_input("Грузопоглощающие пункты (B)", 2, 10, 5, 1))
    method_label = c3.selectbox("Начальный план (опорный)",
                                ["Метод северо-западного угла", "Метод Фогеля",
                                 "Метод наименьшего элемента по строкам",
                                 "Метод наименьшего элемента по столбцам",
                                 "Метод двойного предпочтения"])
    unit = c4.text_input("Единицы для P", "т·км")
    method = {"Метод Фогеля": "vogel",
              "Метод наименьшего элемента по строкам": "min_row",
              "Метод наименьшего элемента по столбцам": "min_col",
              "Метод двойного предпочтения": "dp"}.get(method_label, "nw")
    tie = "first"
    if method == "vogel":
        tie_label = st.selectbox(
            "Если у нескольких строк/столбцов одинаковый наибольший штраф, брать",
            ["первую (верхнюю строку / левый столбец)", "последнюю (нижнюю строку / правый столбец)",
             "ту, где тариф минимальный"])
        tie = "first" if tie_label.startswith("первую") else "mincost" if tie_label.startswith("ту") else "last"
    show_vals = st.checkbox("Показывать числа оценок Δ в свободных клетках (вместо знака «+»)", value=False)

    st.caption("В клетках — тарифы cᵢⱼ. Последний столбец — запасы Aᵢ, последняя строка — потребности Bⱼ "
               "(правая нижняя клетка не используется). Если суммы не равны, фиктивный пункт добавится сам.")
    grid = build_grid(m, n)
    cfg = {col: st.column_config.NumberColumn(col, min_value=0.0) for col in grid.columns}
    edited = call_wide(st.data_editor, grid, key=f"grid_{m}_{n}", column_config=cfg)

    if call_wide(st.button, "🚀 Рассчитать", type="primary"):
        st.session_state["calc"] = True

    if not st.session_state.get("calc"):
        return

    arr = edited.to_numpy(dtype=float)
    costs, supply, demand = arr[:m, :n], arr[:m, n], arr[m, :n]
    if np.isnan(costs).any() or np.isnan(supply).any() or np.isnan(demand).any():
        st.error("Заполните все ячейки: тарифы, запасы и потребности (пустые клетки не допускаются).")
        return
    if (costs < 0).any() or (supply < 0).any() or (demand < 0).any():
        st.error("Отрицательные значения не допускаются.")
        return
    if supply.sum() <= 0 or demand.sum() <= 0:
        st.error("Суммарные запасы и потребности должны быть больше нуля.")
        return

    show_results(supply, demand, costs, method, unit, show_vals, tie)


if __name__ == "__main__":
    main()