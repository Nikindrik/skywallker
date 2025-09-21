from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence  # noqa: UP035

from .graph import Graph


@dataclass(slots=True)
class ACOParams:
    '''
    Параметры алгоритма муравьиной колонии

    Attributes:
        alpha: важность феромона
        beta: важность эвристической информации (обратной стоимости)
        rho: коэффициент испарения феромона (0 < rho < 1)
        q: количество феромона, откладываемого муравьем
        elitist_weight: вес элитного муравья (0 - без элитного муравья)
        tau0: начальное значение феромона на ребрах
    '''
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.5
    q: float = 1.0
    elitist_weight: float = 0.0
    tau0: float = 0.1

@dataclass(slots=True)
class RunResult:
    '''
    Результат выполнения алгоритма муравьиной колонии

    Attributes:
        best_tour: лучший найденный тур (последовательность вершин)
        best_cost: стоимость лучшего тура
        iterations: количество выполненных итераций
    '''
    best_tour: list[int]
    best_cost: float
    iterations: int

class AntColony:
    '''
    Алгоритм муравьиной колонии для задачи коммивояжера

    Attributes:
        graph: объект графа (матрица стоимостей)
        params: параметры алгоритма (ACOParams)
        n_ants: количество муравьев (по умолчанию max(10, n), где n - число вершин)
        n_iterations: количество итераций (по умолчанию 200)
        start_vertex: начальная вершина для всех муравьев (по умолчанию None - случайная)
        seed: зерно для генератора случайных чисел (по умолчанию None - системное время)
        early_stop: количество итераций без улучшения для ранней остановки (по умолчанию None - не использовать)
        verbose: вывод отладочной информации (по умолчанию True)
    '''
    def __init__(self, graph: Graph, params: ACOParams | None=None, *, n_ants: int | None=None,
                 n_iterations:int=200, start_vertex:int | None=0, seed:int | None=None,
                 early_stop:int | None=None, verbose:bool=True) -> None:
        self.g=graph
        self.params=params or ACOParams()
        self.n_ants=n_ants or max(10, graph.n)
        self.n_iterations=n_iterations
        self.start_vertex=start_vertex
        self.rng=random.Random(seed)
        self.early_stop=early_stop
        self.verbose=verbose
        n=graph.n
        self.tau=[[self.params.tau0 if i!=j else 0.0 for j in range(n)] for i in range(n)]
        self.eta = [[0.0 if i == j else 1.0/graph.cost(i,j)
            if math.isfinite(graph.cost(i,j)) and graph.cost(i,j) > 0
            else 0.0 for j in range(n)] for i in range(n)]

    def run(self) -> RunResult:
        '''
        Запуск алгоритма муравьиной колонии
        Возвращает объект RunResult с результатами
        '''
        best_tour = []
        best_cost = math.inf
        n_no_improve = 0

        if self.verbose:
            print("=== Параметры ===", self.params)
            print("Матрица расстояний:")
            [print(["∞" if not math.isfinite(x) else round(x, 2) for x in row]) for row in self.g.w]
            print("Начальные феромоны:")
            [print([round(x, 3) for x in row]) for row in self.tau]
            print()

        for it in range(1, self.n_iterations + 1):
            all_tours = []
            all_costs = []

            #if self.verbose and (it <= 3 or it > self.n_iterations - 3):
            #    print(f"\n--- Итерация {it} ---")

            # Обычные муравьи
            for ant_id in range(self.n_ants):
                s = self.start_vertex if self.start_vertex is not None else self.rng.randrange(self.g.n)
                tour = self._construct_tour(start=s)
                cost = self.g.tour_cost(tour)
                all_tours.append(tour)
                all_costs.append(cost)

                if self.verbose and (it <= 3 or it > self.n_iterations - 3):
                    route_str = " -> ".join(map(str, tour))
                    print(f"Муравей {ant_id+1}, маршрут {route_str}, стоимость={cost}")

                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour
                    n_no_improve = 0

            # Элитный муравей
            if self.params.elitist_weight > 0 and best_tour:
                all_tours.append(best_tour)
                all_costs.append(best_cost)
                if self.verbose and (it <= 3 or it > self.n_iterations - 3):
                    route_str = " -> ".join(map(str, best_tour))
                    print(f"Элитный муравей, маршрут {route_str}, стоимость={best_cost}")

            n_no_improve += 1

            # Испарение и откладывание феромона
            self._evaporate()
            self._deposit(all_tours, all_costs, best_tour, best_cost)

            if self.verbose and (it <= 3 or it > self.n_iterations - 3):
                if it != self.n_iterations - 1:
                    print(f"Итерация {it}: лучший {best_tour}, стоимость={best_cost}")
                else:
                    print(f"Итерация {it} итоговая: лучший {best_tour}, стоимость={best_cost}")
                print("Феромоны:")
                [print([round(x, 3) for x in row]) for row in self.tau]

            if self.early_stop and n_no_improve >= self.early_stop:
                break

        return RunResult(best_tour=best_tour, best_cost=best_cost, iterations=it)

    def _construct_tour(self, *, start: int) -> list[int]:
        '''
        Построение тура одним муравьем, начиная с вершины start
        Attributes:
            start: начальная вершина
        Returns:
            построенный тур (список вершин)
        '''
        n = self.g.n
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        cur = start

        while unvisited:
            # Фильтр только достижимые вершины
            reachable = {j for j in unvisited 
                        if math.isfinite(self.g.cost(cur, j)) and self.g.cost(cur, j) > 0}
            if not reachable:
                #if self.verbose:
                #    print(f"Предупреждение: вершина {cur} не имеет достижимых соседей")
                break
            nxt = self._choose_next(cur, reachable)
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        # Цикл замыкания
        if (math.isfinite(self.g.cost(tour[-1], start)) and
            self.g.cost(tour[-1], start) > 0):
            tour.append(start)
        return tour

    def _choose_next(self, i: int, candidates: set[int]) -> int:
        '''
        Выбор следующей вершины из множества кандидатов, находясь в вершине i
        Добавлено "разведывательное" поведение: с вероятностью epsilon
        муравей выбирает случайную вершину, игнорируя феромоны

        Attributes:
            i: текущая вершина
            candidates: множество доступных вершин для перехода
        Returns:
            выбранная вершина
        '''
        alpha = self.params.alpha
        beta = self.params.beta
        rng = self.rng

        epsilon = 0.1  # вероятность разведки 10%

        if rng.random() < epsilon:  # Разведка: случайный выбор
            return rng.choice(list(candidates))

        # Нормальное вероятностное правило
        valid_candidates = []
        probabilities = []

        for j in candidates:
            if not math.isfinite(self.g.cost(i, j)) or self.g.cost(i, j) <= 0:
                continue
            tau, eta = self.tau[i][j], self.eta[i][j]
            score = (tau ** alpha) * (eta ** beta) if tau > 0 and eta > 0 else 0.0
            valid_candidates.append(j)
            probabilities.append(score)

        if not valid_candidates:
            return rng.choice(list(candidates))

        total = sum(probabilities)
        if total <= 0:
            return rng.choice(valid_candidates)

        normalized_probs = [p / total for p in probabilities]
        return rng.choices(valid_candidates, weights=normalized_probs, k=1)[0]


    def _evaporate(self)->None:
        '''Испарение феромонов на всех ребрах'''
        rho=self.params.rho
        n=self.g.n
        for i in range(n):
            for j in range(n):
                if i!=j:
                    self.tau[i][j]*=(1.0-rho)

    def _deposit(self, tours: Sequence[Sequence[int]], costs: Sequence[float],
             best_tour: Sequence[int], best_cost: float) -> None:
        '''
        Откладывание феромонов муравьями на всех ребрах

        Attributes:
            tours: список туров всех муравьев
            costs: список стоимостей туров всех муравьев
            best_tour: лучший тур за все время
            best_cost: стоимость лучшего тура за все время
        '''
        q = self.params.q
        elite = self.params.elitist_weight

        def add(path, amount):
            for a, b in zip(path, path[1:], strict=False):
                if a != b and math.isfinite(self.g.cost(a, b)):
                    self.tau[a][b] += amount

        for tour, cost in zip(tours, costs, strict=False):
            if math.isfinite(cost) and cost > 0:
                add(tour, q / cost)
        if elite > 0 and math.isfinite(best_cost) and best_cost > 0:
            add(best_tour, elite * q / best_cost)
