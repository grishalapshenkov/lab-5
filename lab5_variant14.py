"""
Лабораторная работа № 5. Вариант 14.
Хеш-таблица: метод умножения; разрешение коллизий цепочками и квадратичным
пробированием. Целые случайные ключи, n = 25 000, m = 32 768.
Дополнительное задание В7: подсчёт частот в логе из 10^6 записей.
"""
import gc
import heapq
import json
import math
import random
import statistics
import time
from collections import Counter

M = 32_768                    # размер таблицы (по варианту)
N = 25_000                    # число хранимых ключей
KEY_RANGE = 10_000_000        # ключи — случайные целые из [0; 10^7)
A = (math.sqrt(5) - 1) / 2    # постоянная Кнута, A ≈ 0,6180339887
QUERIES = 10_000              # число запросов каждого вида
REPEATS = 5                   # повторов замера (берётся медиана)


# ------------------------------------------------------------ хеш-функции
def hash_mul(key, m):
    """Метод умножения: h(k) = floor(m * {k * A})."""
    return int(m * ((key * A) % 1.0))


def hash_bad(key, m):
    """Заведомо неудачная функция: всего 10 различных значений."""
    return key % 10


# ------------------------------------------------- метод цепочек
class ChainHashTable:
    """Хеш-таблица с разрешением коллизий методом цепочек.
    Проба — просмотр одного элемента цепочки (сравнение ключей)."""

    def __init__(self, m=M, hash_func=hash_mul):
        self.m, self.hash_func = m, hash_func
        self.slots = [[] for _ in range(m)]
        self.n = 0
        self.probes = 0

    def insert(self, key, value):
        chain = self.slots[self.hash_func(key, self.m)]
        for item in chain:
            self.probes += 1
            if item[0] == key:
                item[1] = value
                return
        chain.append([key, value])
        self.n += 1

    def add(self, key, delta=1):
        """Увеличить значение по ключу (счётчик); при отсутствии ключа — создать."""
        chain = self.slots[self.hash_func(key, self.m)]
        for item in chain:
            self.probes += 1
            if item[0] == key:
                item[1] += delta
                return
        chain.append([key, delta])
        self.n += 1

    def search(self, key):
        for item in self.slots[self.hash_func(key, self.m)]:
            self.probes += 1
            if item[0] == key:
                return item[1]
        return None

    def delete(self, key):
        chain = self.slots[self.hash_func(key, self.m)]
        for i, item in enumerate(chain):
            self.probes += 1
            if item[0] == key:
                del chain[i]
                self.n -= 1
                return True
        return False

    def items(self):
        return [(k, v) for chain in self.slots for k, v in chain]

    def chain_lengths(self):
        return [len(c) for c in self.slots]


# ---------------------------------------- квадратичное пробирование
class QuadHashTable:
    """Хеш-таблица с открытой адресацией и квадратичным пробированием
    h_i(k) = (h(k) + i/2 + i^2/2) mod m = (h(k) + i(i+1)/2) mod m.
    При m = 2^s такая последовательность проходит все ячейки таблицы.
    Удаление — с пометкой DELETED (tombstone)."""

    EMPTY = object()
    DELETED = object()

    def __init__(self, m=M, hash_func=hash_mul):
        self.m, self.hash_func = m, hash_func
        self.keys = [self.EMPTY] * m
        self.values = [None] * m
        self.n = 0
        self.probes = 0

    def _locate(self, key):
        """Возвращает (позиция ключа или -1, позиция для вставки или -1)."""
        m, keys = self.m, self.keys
        pos = self.hash_func(key, m)
        first_free = -1
        for i in range(1, m + 1):
            self.probes += 1
            k = keys[pos]
            if k is self.EMPTY:
                return -1, (first_free if first_free != -1 else pos)
            if k is self.DELETED:
                if first_free == -1:
                    first_free = pos
            elif k == key:
                return pos, -1
            pos = (pos + i) % m            # следующий член: h + i(i+1)/2
        return -1, first_free              # просмотрена вся таблица

    def _store(self, key, value, free):
        if free == -1:
            raise RuntimeError("Таблица переполнена")
        self.keys[free], self.values[free] = key, value
        self.n += 1

    def insert(self, key, value):
        pos, free = self._locate(key)
        if pos != -1:
            self.values[pos] = value
        else:
            self._store(key, value, free)

    def add(self, key, delta=1):
        pos, free = self._locate(key)
        if pos != -1:
            self.values[pos] += delta
        else:
            self._store(key, delta, free)

    def search(self, key):
        pos, _ = self._locate(key)
        return self.values[pos] if pos != -1 else None

    def delete(self, key):
        pos, _ = self._locate(key)
        if pos == -1:
            return False
        self.keys[pos], self.values[pos] = self.DELETED, None
        self.n -= 1
        return True

    def items(self):
        return [(k, v) for k, v in zip(self.keys, self.values)
                if k is not self.EMPTY and k is not self.DELETED]


# ------------------------------------------------------------- проверка
def self_test():
    """Контроль корректности: случайные операции сверяются со словарём dict."""
    probe_cells = {(i * (i + 1) // 2) % M for i in range(M)}
    assert len(probe_cells) == M, "последовательность проб не покрывает таблицу"
    rng = random.Random(1)
    for cls in (ChainHashTable, QuadHashTable):
        t, ref = cls(64), {}
        for _ in range(20_000):
            key, op = rng.randrange(40), rng.random()
            if op < 0.4:
                t.insert(key, key * 2)
                ref[key] = key * 2
            elif op < 0.7:
                assert t.delete(key) == (key in ref)
                ref.pop(key, None)
            else:
                assert t.search(key) == ref.get(key)
        assert t.n == len(ref) and dict(t.items()) == ref
    print("Самопроверка пройдена: таблицы согласованы с dict, пробирование покрывает все ячейки.")


# --------------------------------------------------------------- замеры
def make_keys(n, rng):
    keys = rng.sample(range(KEY_RANGE), n)
    present = set(keys)
    found = [rng.choice(keys) for _ in range(QUERIES)]
    absent = []
    while len(absent) < QUERIES:
        q = rng.randrange(KEY_RANGE)
        if q not in present:
            absent.append(q)
    return keys, found, absent


def timed_pass(func, queries):
    gc.collect()
    t0 = time.perf_counter()
    for q in queries:
        func(q)
    return time.perf_counter() - t0


def bench_table(factory, keys, found, absent, repeats=REPEATS):
    """Время вставки/поиска и среднее число проб для одной реализации."""
    t_ins = []
    for r in range(repeats):
        table = factory()
        gc.collect()
        t0 = time.perf_counter()
        for k in keys:
            table.insert(k, k)
        t_ins.append(time.perf_counter() - t0)
        if r == 0:
            ins_probes = table.probes / len(keys)
    out = {"t_insert": statistics.median(t_ins), "p_insert": ins_probes}
    for label, qs in (("found", found), ("absent", absent)):
        table.probes = 0
        times = [timed_pass(table.search, qs) for _ in range(repeats)]
        out["t_" + label] = statistics.median(times)
        out["p_" + label] = table.probes / (len(qs) * repeats)
    return out, table


def bench_dict(keys, found, absent, repeats=REPEATS):
    t_ins = []
    for _ in range(repeats):
        d = {}
        gc.collect()
        t0 = time.perf_counter()
        for k in keys:
            d[k] = k
        t_ins.append(time.perf_counter() - t0)
    return {"t_insert": statistics.median(t_ins),
            "t_found": statistics.median(timed_pass(d.get, found) for _ in range(repeats)),
            "t_absent": statistics.median(timed_pass(d.get, absent) for _ in range(repeats))}


def theory_probes(alpha):
    """Теоретические оценки числа проб (равномерное хеширование)."""
    return {"chain_found": 1 + alpha / 2, "chain_absent": alpha,
            "quad_found": math.log(1 / (1 - alpha)) / alpha,
            "quad_absent": 1 / (1 - alpha)}


# ------------------------------------------------- задание В7 (лог)
def make_log(total, clients, rng):
    """Лог операций: запись = идентификатор клиента. Активность клиентов
    неравномерна (распределение Ципфа: вес клиента с рангом r ~ 1/r)."""
    ids = rng.sample(range(1_000_000, 10_000_000), clients)
    weights = [1 / r for r in range(1, clients + 1)]
    return rng.choices(ids, weights=weights, k=total)


def count_list(log):
    """Решение «через список и in»: два параллельных списка."""
    seen, counts = [], []
    for c in log:
        if c in seen:                       # Θ(u): линейный просмотр списка
            counts[seen.index(c)] += 1      # ещё один просмотр
        else:
            seen.append(c)
            counts.append(1)
    return dict(zip(seen, counts))


def count_hash(log, table_cls, m):
    table = table_cls(m)
    for c in log:
        table.add(c)
    return dict(table.items())


def count_dict(log):
    d = {}
    for c in log:
        d[c] = d.get(c, 0) + 1
    return d


def count_counter(log):
    return Counter(log)


def top10(freq):
    """10 клиентов с наибольшим числом операций (при равенстве — по возрастанию id)."""
    return heapq.nlargest(10, freq.items(), key=lambda kv: (kv[1], -kv[0]))


# ------------------------------------------------------------- графики
def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Рис. 1. Число проб от коэффициента заполнения
    sw = res["sweep"]
    al = [r["alpha"] for r in sw]
    fine = [i / 200 for i in range(10, 195)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, kind, title in ((axes[0], "found", "Успешный поиск"), (axes[1], "absent", "Неуспешный поиск")):
        ax.plot(al, [r["chain"]["p_" + kind] for r in sw], "o-", label="цепочки (опыт)")
        ax.plot(al, [r["quad"]["p_" + kind] for r in sw], "s-", label="квадр. пробирование (опыт)")
        ax.plot(fine, [theory_probes(a)["chain_" + kind] for a in fine], "--", color="C0", lw=1,
                label="цепочки (теория)")
        ax.plot(fine, [theory_probes(a)["quad_" + kind] for a in fine], "--", color="C1", lw=1,
                label="открытая адресация (теория)")
        ax.set_yscale("log"); ax.set_xlabel("коэффициент заполнения α")
        ax.set_ylabel("среднее число проб"); ax.set_title(title)
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("img/lr5_alpha.png", dpi=170); plt.close()

    # Рис. 2. Длины цепочек и распределение Пуассона
    pt = res["poisson"]
    plt.figure(figsize=(6.4, 4.2))
    xs = [r["len"] for r in pt]
    w = 0.4
    plt.bar([x - w / 2 for x in xs], [r["obs"] for r in pt], w, label="опыт")
    plt.bar([x + w / 2 for x in xs], [r["exp"] for r in pt], w, label="Пуассон, λ = α")
    plt.xlabel("длина цепочки"); plt.ylabel("число ячеек")
    plt.title("Распределение длин цепочек (n = 25 000, m = 32 768)")
    plt.legend(); plt.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig("img/lr5_chains.png", dpi=170); plt.close()

    # Рис. 3. В7: время подсчёта частот
    v7 = res["v7"]
    plt.figure(figsize=(6.8, 4.4))
    sizes = [r["N"] for r in v7["runs"]]
    for name, style in (("список и in", "o-"), ("своя таблица (цепочки)", "s-"),
                        ("своя таблица (квадр.)", "^-"), ("dict", "d-"), ("Counter", "v-")):
        pts = [(r["N"], r["t"][name]) for r in v7["runs"] if name in r["t"]]
        plt.plot(*zip(*pts), style, label=name)
    ext = v7.get("list_extrapolated")
    if ext:
        plt.plot([e["N"] for e in ext], [e["t"] for e in ext], "x--", color="C0",
                 label="список и in (экстраполяция)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("число записей лога N"); plt.ylabel("время, с")
    plt.title("Подсчёт частот клиентов в логе"); plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("img/lr5_v7.png", dpi=170); plt.close()


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    self_test()
    rng = random.Random(14)
    res = {}

    # ---- 1. Основной эксперимент: n = 25 000, m = 32 768
    keys, found, absent = make_keys(N, rng)
    chain, t_chain = bench_table(lambda: ChainHashTable(M), keys, found, absent)
    quad, t_quad = bench_table(lambda: QuadHashTable(M), keys, found, absent)
    res["main"] = {"alpha": N / M, "chain": chain, "quad": quad,
                   "dict": bench_dict(keys, found, absent)}
    print(f"n = {N}, m = {M}, α = {N / M:.4f}")
    for name, r in (("Цепочки", chain), ("Квадр. пробирование", quad)):
        print(f"{name:<22} вставка {r['t_insert']:.4f} с, поиск {r['t_found']:.4f} с, "
              f"проб: вставка {r['p_insert']:.2f}, найден {r['p_found']:.2f}, не найден {r['p_absent']:.2f}")
    print("dict", res["main"]["dict"])

    # ---- 2. Распределение длин цепочек и пуассоновская модель
    lengths = t_chain.chain_lengths()
    alpha = N / M
    cnt = Counter(lengths)
    res["chains"] = {"max": max(lengths), "empty": cnt[0], "m": M}
    res["poisson"] = [{"len": j, "obs": cnt.get(j, 0),
                       "exp": M * math.exp(-alpha) * alpha ** j / math.factorial(j)}
                      for j in range(max(lengths) + 1)]
    for row in res["poisson"]:
        print(f"  длина {row['len']}: ячеек {row['obs']:>6}, по Пуассону {row['exp']:>9.1f}")

    # ---- 3. Зависимость числа проб от α
    res["sweep"] = []
    for a in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
        n = round(a * M)
        k2, f2, a2 = make_keys(n, rng)
        ch, _ = bench_table(lambda: ChainHashTable(M), k2, f2, a2, repeats=1)
        qd, _ = bench_table(lambda: QuadHashTable(M), k2, f2, a2, repeats=1)
        res["sweep"].append({"alpha": a, "n": n, "chain": ch, "quad": qd, "theory": theory_probes(a)})
        print(f"α = {a}: цепочки {ch['p_found']:.2f}/{ch['p_absent']:.2f}, "
              f"квадр. {qd['p_found']:.2f}/{qd['p_absent']:.2f}")

    # ---- 4. Качественная и неудачная хеш-функции (n = 5000)
    nb = 5_000
    kb, fb, ab = make_keys(nb, rng)
    res["bad"] = {"n": nb, "rows": []}
    for hname, hf in (("h = ⌊m·{kA}⌋", hash_mul), ("h = k mod 10", hash_bad)):
        for cname, cls in (("Цепочки", ChainHashTable), ("Квадр. пробирование", QuadHashTable)):
            r, tab = bench_table(lambda: cls(M, hf), kb, fb, ab, repeats=1)
            distinct = len({hf(k, M) for k in kb})
            longest = max(tab.chain_lengths()) if cls is ChainHashTable else None
            res["bad"]["rows"].append({"hash": hname, "method": cname, "distinct": distinct,
                                       "longest": longest, **r})
            print(hname, cname, distinct, longest, {k: round(v, 3) for k, v in r.items()})

    # ---- 5. В7: частоты клиентов в логе из 10^6 записей
    TOTAL, CLIENTS = 1_000_000, 50_000
    log = make_log(TOTAL, CLIENTS, random.Random(7))
    sizes = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
    LIMIT = 240.0                                     # предельное время одного замера, с
    solvers = {"список и in": count_list,
               "своя таблица (цепочки)": lambda lg: count_hash(lg, ChainHashTable, 131_072),
               "своя таблица (квадр.)": lambda lg: count_hash(lg, QuadHashTable, 131_072),
               "dict": count_dict, "Counter": count_counter}
    runs, list_skipped = [], False
    for size in sizes:
        part = log[:size]
        row, reference = {}, None
        for name, func in solvers.items():
            if name == "список и in" and (list_skipped or (
                    runs and "список и in" in runs[-1]["t"] and
                    runs[-1]["t"]["список и in"] * (size / runs[-1]["N"]) ** 2 > LIMIT)):
                list_skipped = True
                continue
            t0 = time.perf_counter()
            freq = func(part)
            row[name] = time.perf_counter() - t0
            if reference is None:
                reference = (len(freq), top10(freq))
            else:
                assert (len(freq), top10(freq)) == reference, f"результаты расходятся: {name}"
        runs.append({"N": size, "t": row, "unique": reference[0]})
        print(f"N = {size}: уникальных {reference[0]}, " +
              ", ".join(f"{k} {v:.3f} с" for k, v in row.items()), flush=True)
    freq = count_dict(log)
    res["v7"] = {"total": TOTAL, "clients": CLIENTS, "unique": len(freq),
                 "top10": top10(freq), "runs": runs}
    # экстраполяция времени решения через список степенной зависимостью по двум последним точкам
    meas = [(r["N"], r["t"]["список и in"]) for r in runs if "список и in" in r["t"]]
    if meas[-1][0] < TOTAL:
        (n1, t1), (n2, t2) = meas[-2], meas[-1]
        p = math.log(t2 / t1) / math.log(n2 / n1)
        res["v7"]["list_exponent"] = p
        res["v7"]["list_extrapolated"] = [
            {"N": s, "t": t2 * (s / n2) ** p} for s in sizes if s >= n2]
        print(f"показатель степени {p:.2f}; оценка для N = 10^6: {t2 * (TOTAL / n2) ** p:.0f} с")

    with open("res/lr5.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
