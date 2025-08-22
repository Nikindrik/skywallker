
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import random
from .graph import Graph


@dataclass(slots=True)
class ACOParams:
    """Параметры ACO.

    alpha: влияние феромона (>=0)
    beta: влияние эвристики (1/дистанции) (>=0)
    rho: коэффициент испарения (0..1)
    q: масштаб депонирования феромона (>0)
    elitist_weight: дополнительное усиление лучшего маршрута (0 - отключено)
    tau0: начальная концентрация феромона (>0)
    """
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.5
    q: float = 1.0
    elitist_weight: float = 0.0
    tau0: float = 0.1


@dataclass(slots=True)
class RunResult:
    best_tour: List[int]
    best_cost: float
    iterations: int


class AntColony:
    """Муравьиный алгоритм для (A)TSP.

    Поддерживает асимметричный граф, произвольную стартовую вершину или случайный старт.
    """

    def __init__(
        self,
        graph: Graph,
        params: Optional[ACOParams] = None,
        *,
        n_ants: Optional[int] = None,
        n_iterations: int = 200,
        start_vertex: Optional[int] = 0,
        seed: Optional[int] = None,
        early_stop: Optional[int] = None,
    ) -> None:
        if graph.n < 2:
            raise ValueError("Граф слишком мал.")
        self.g = graph
        self.params = params or ACOParams()
        self.n_ants = n_ants or max(10, graph.n)  # по умолчанию >= числу вершин
        self.n_iterations = n_iterations
        self.start_vertex = start_vertex  # None -> случайный старт каждого муравья
        self.rng = random.Random(seed)
        self.early_stop = early_stop  # кол-во итераций без улучшения для остановки

        n = graph.n
        self.tau = [[self.params.tau0 if i != j else 0.0 for j in range(n)] for i in range(n)]
        self.eta = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.eta[i][j] = 0.0
                else:
                    c = self.g.cost(i, j)
                    self.eta[i][j] = 0.0 if not math.isfinite(c) or c <= 0 else 1.0 / c

    # ---------------------------- Основной цикл -----------------------------
    def run(self) -> RunResult:
        best_tour: List[int] = []
        best_cost: float = math.inf
        n_no_improve = 0

        for it in range(1, self.n_iterations + 1):
            all_tours: List[List[int]] = []
            all_costs: List[float] = []
            for _ in range(self.n_ants):
                s = self.start_vertex if self.start_vertex is not None else self.rng.randrange(self.g.n)
                tour = self._construct_tour(start=s)
                cost = self.g.tour_cost(tour)
                all_tours.append(tour)
                all_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour
                    n_no_improve = 0
            n_no_improve += 1

            self._evaporate()
            self._deposit(all_tours, all_costs, best_tour, best_cost)

            if self.early_stop and n_no_improve >= self.early_stop:
                break

        return RunResult(best_tour=best_tour, best_cost=best_cost, iterations=it)

    # --------------------------- Построение тура ----------------------------
    def _construct_tour(self, *, start: int) -> List[int]:
        n = self.g.n
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        cur = start
        while unvisited:
            nxt = self._choose_next(cur, unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        tour.append(start)  # возврат в начало
        return tour

    def _choose_next(self, i: int, candidates: set[int]) -> int:
        alpha = self.params.alpha
        beta = self.params.beta
        rng = self.rng

        scores = []
        total = 0.0
        for j in candidates:
            tau_ij = self.tau[i][j]
            eta_ij = self.eta[i][j]
            if tau_ij <= 0.0 or eta_ij <= 0.0:
                score = 0.0
            else:
                score = (tau_ij ** alpha) * (eta_ij ** beta)
            scores.append((j, score))
            total += score

        if total <= 0.0:
            # fallback: равновероятный выбор из достижимых кандидатов
            return rng.choice(list(candidates))

        r = rng.random()
        cum = 0.0
        for j, score in scores:
            if score == 0.0:
                continue
            p = score / total
            cum += p
            if r <= cum:
                return j
        return scores[-1][0]

    # -------------------------- Обновление феромона ------------------------
    def _evaporate(self) -> None:
        rho = self.params.rho
        n = self.g.n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.tau[i][j] *= (1.0 - rho)

    def _deposit(self, tours: Sequence[Sequence[int]], costs: Sequence[float], best_tour: Sequence[int], best_cost: float) -> None:
        q = self.params.q
        elite = self.params.elitist_weight
        n = self.g.n

        def add_to_pheromone(path: Sequence[int], amount: float) -> None:
            for a, b in zip(path, path[1:]):
                if a == b:
                    continue
                self.tau[a][b] += amount

        for tour, cost in zip(tours, costs):
            if not math.isfinite(cost) or cost <= 0:
                continue
            add_to_pheromone(tour, q / cost)

        if elite > 0.0 and math.isfinite(best_cost) and best_cost > 0:
            add_to_pheromone(best_tour, elite * q / best_cost)
