
from __future__ import annotations

import csv
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Vertex:
    '''
    Вершина графа

    Attributes:
        idx: Порядковый индекс вершины [0..n-1]
        name: Читаемое имя (необязательно)
    '''
    idx: int
    name: str | None = None

    def label(self) -> str:
        return self.name or f"v{self.idx}"


class Graph:
    '''
    Взвешенный ориентированный граф для задачи коммивояжёра

    Хранит матрицу смежности `w[i][j]` (стоимость перехода i->j)
    Допускает асимметрию (w[i][j] != w[j][i])
    Требование: для i != j, вес > 0; для i == j, вес = +inf или 0 (не используется)
    '''

    def __init__(self, weights: list[list[float]], names: Sequence[str] | None = None) -> None:
        n = len(weights)
        if n == 0 or any(len(row) != n for row in weights):
            raise ValueError("Матрица весов должна быть квадратной и непустой.")
        self.n: int = n
        self.w: list[list[float]] = [[float(x) for x in row] for row in weights]
        self.vertices: list[Vertex] = [Vertex(i, (names[i] if names else None)) for i in range(n)]
        # Нормализуем диагональ
        for i in range(n):
            if i < len(self.w[i]):
                self.w[i][i] = math.inf

    # ---- Основные операции -------------------------------------------------
    def cost(self, i: int, j: int) -> float:
        return self.w[i][j]

    def tour_cost(self, tour: Sequence[int]) -> float:
        '''Стоимость маршрута (замкнутого)'''
        if len(tour) < 2:
            return math.inf
        total = 0.0
        for a, b in zip(tour, tour[1:], strict=False):
            total += self.cost(a, b)
        return total

    def is_complete(self) -> bool:
        for i in range(self.n):
            for j in range(self.n):
                if i != j and not math.isfinite(self.w[i][j]):
                    return False
        return True

    # ---- Загрузка / сохранение --------------------------------------------
    @staticmethod
    def from_csv(path: str, directed: bool = True) -> Graph:
        '''Загружает граф из CSV-файла с квадратной матрицей

        В ячейках допускаются целые/вещественные числа.
        Если directed=False, матрица симметризуется: w = (w + w^T)/2
        '''
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [[float(x) for x in row] for row in reader if row]
        if not rows or any(len(r) != len(rows) for r in rows):
            raise ValueError("CSV должен содержать квадратную матрицу.")
        n = len(rows)
        # Заменяем нули вне диагонали на +inf (нет ребра)
        for i in range(n):
            for j in range(n):
                if i != j and rows[i][j] == 0:
                    rows[i][j] = math.inf
        if not directed:
            for i in range(n):
                for j in range(i + 1, n):
                    v = (rows[i][j] + rows[j][i]) / 2.0
                    rows[i][j] = rows[j][i] = v
        return Graph(rows)

    def to_csv(self, path: str) -> None:
        '''Сохраняет граф в CSV-файл'''
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row in self.w:
                writer.writerow(["" if not math.isfinite(x) else (int(x) if float(x).is_integer() else x) for x in row])


class GraphFactory:
    '''Фабрика для гибкой генерации графов'''

    @staticmethod
    def random_complete(n: int, *, directed: bool = True, low: int = 1, high: int = 100, seed: int | None = None) -> Graph:  # noqa: E501
        '''Создаёт полный граф размера n со случайными весами в [low, high]'''
        if n < 2:
            raise ValueError("n >= 2")
        rng = random.Random(seed)
        w = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    w[i][j] = float("inf")
                else:
                    w[i][j] = float(rng.randint(low, high))
        if not directed:
            for i in range(n):
                for j in range(i + 1, n):
                    v = (w[i][j] + w[j][i]) / 2.0
                    w[i][j] = w[j][i] = v
        return Graph(w)

    @staticmethod
    def random_sparse(n: int, m: int, *, directed: bool = True, low: int = 1, high: int = 100, seed: int | None = None) -> Graph:  # noqa: E501
        '''Создаёт разреженный граф: n вершин, ~m ориентированных дуг (без самопетель)

        Пустые рёбра получают вес +inf. Если граф несвязный, ACO может не найти валидный цикл
        Для TSP обычно используют полный граф; этот метод полезен для экспериментов
        '''
        if n < 2:
            raise ValueError("n >= 2")
        max_m = n * (n - 1) if directed else n * (n - 1) // 2
        if m > max_m:
            raise ValueError(f"слишком много рёбер: m <= {max_m}")
        rng = random.Random(seed)
        w = [[float("inf") for _ in range(n)] for _ in range(n)]
        for i in range(n):
            w[i][i] = float("inf")
        edges = set()
        while len(edges) < m:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if directed:
                e = (i, j)
                if e in edges:
                    continue
                edges.add(e)
                w[i][j] = float(rng.randint(low, high))
            else:
                e = (min(i, j), max(i, j))
                if e in edges:
                    continue
                edges.add(e)
                val = float(rng.randint(low, high))
                w[i][j] = w[j][i] = val
        return Graph(w)
