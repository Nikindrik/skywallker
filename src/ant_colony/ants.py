from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence  # noqa: UP035

from .graph import Graph


@dataclass(slots=True)
class ACOParams:
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.5
    q: float = 1.0
    elitist_weight: float = 0.0
    tau0: float = 0.1

@dataclass(slots=True)
class RunResult:
    best_tour: list[int]
    best_cost: float
    iterations: int

class AntColony:
    def __init__(self, graph: Graph, params: ACOParams | None=None, *, n_ants: int | None=None,
                 n_iterations:int=200, start_vertex:int | None=0, seed:int | None=None,
                 early_stop:int | None=None, verbose:bool=True) -> None:
        self.g=graph
        self.params=params or ACOParams()
        self.n_ants=n_ants or max(10,graph.n)
        self.n_iterations=n_iterations
        self.start_vertex=start_vertex
        self.rng=random.Random(seed)
        self.early_stop=early_stop
        self.verbose=verbose
        n=graph.n
        self.tau=[[self.params.tau0 if i!=j else 0.0 for j in range(n)] for i in range(n)]
        self.eta=[[0.0 if i==j else (0.0 if not math.isfinite(graph.cost(i,j)) or graph.cost(i,j)<=0 else 1.0/graph.cost(i,j)) for j in range(n)] for i in range(n)]  # noqa: E501

    def run(self)->RunResult:
        best_tour=[]
        best_cost=math.inf
        n_no_improve=0
        if self.verbose:
            print("=== Параметры ===",self.params)
            print("Матрица расстояний:")
            [print(["∞" if not math.isfinite(x) else round(x,2) for x in row]) for row in self.g.w]
            print("Начальные феромоны:")
            [print([round(x,3) for x in row]) for row in self.tau]
        for it in range(1,self.n_iterations+1):
            all_tours=[]
            all_costs=[]
            for ant_id in range(self.n_ants):
                s=self.start_vertex if self.start_vertex is not None else self.rng.randrange(self.g.n)
                tour=self._construct_tour(start=s)
                cost=self.g.tour_cost(tour)
                all_tours.append(tour)
                all_costs.append(cost)
                if it==1 and self.verbose:
                    print(f"Муравей {ant_id+1}: {tour}, стоимость={cost}")
                if cost<best_cost:
                    best_cost=cost
                    best_tour=tour
                    n_no_improve=0
            n_no_improve+=1
            self._evaporate()
            self._deposit(all_tours,all_costs,best_tour,best_cost)
            if self.verbose and it % 100 == 0:
                print(f"Итерация {it}: лучший {best_tour}, стоимость={best_cost}")
                print("Феромоны:")
                [print([round(x,3) for x in row]) for row in self.tau]
            if self.early_stop and n_no_improve>=self.early_stop:
                break
        if self.verbose:
            print("=== Итог ===")
            print(f"Лучший: {best_tour}, стоимость={best_cost}")
            print("Финальные феромоны:")
            [print([round(x,3) for x in row]) for row in self.tau]
        return RunResult(best_tour=best_tour,best_cost=best_cost,iterations=it)

    def _construct_tour(self,*,start:int)->list[int]:
        n=self.g.n
        tour=[start]
        unvisited=set(range(n))
        unvisited.remove(start)
        cur=start
        while unvisited:
            nxt=self._choose_next(cur,unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
            cur=nxt
        tour.append(start)
        return tour

    def _choose_next(self,i:int,candidates:set[int])->int:
        alpha=self.params.alpha
        beta=self.params.beta
        rng=self.rng
        scores=[]
        total=0.0
        for j in candidates:
            tau,eta=self.tau[i][j],self.eta[i][j]
            score=(tau**alpha)*(eta**beta) if tau>0 and eta>0 else 0.0
            scores.append((j,score))
            total+=score
        if total<=0:
            return rng.choice(list(candidates))
        r=rng.random()
        cum=0.0
        for j,score in scores:
            if score==0:
                continue
            cum+=score/total
            if r<=cum:
                return j
        return scores[-1][0]

    def _evaporate(self)->None:
        rho=self.params.rho
        n=self.g.n
        for i in range(n):
            for j in range(n):
                if i!=j:
                    self.tau[i][j]*=(1.0-rho)

    def _deposit(self,tours:Sequence[Sequence[int]],costs:Sequence[float],best_tour:Sequence[int],best_cost:float)->None:  # noqa: E501
        q=self.params.q
        elite=self.params.elitist_weight
        def add(path,amount):
            for a,b in zip(path,path[1:], strict=False):
                if a!=b:
                    self.tau[a][b]+=amount
        for tour,cost in zip(tours,costs, strict=False):
            if math.isfinite(cost) and cost>0:
                add(tour,q/cost)
        if elite>0 and math.isfinite(best_cost) and best_cost>0:
            add(best_tour,elite*q/best_cost)
