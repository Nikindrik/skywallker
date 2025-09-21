
from __future__ import annotations

import argparse
from pathlib import Path

from .ants import ACOParams, AntColony
from .graph import Graph, GraphFactory


def build_argparser() -> argparse.ArgumentParser:
    '''Создаёт парсер аргументов командной строки'''
    p = argparse.ArgumentParser(
        prog="ant-colony-tsp",
        description="Муравьиный алгоритм (ACO) для TSP/ATSP: загрузка матрицы или генерация графа.",
    )
    src = p.add_argument_group("Источник графа")
    src.add_argument("--csv", type=str, help="Путь к CSV с квадратной матрицей весов", default=None)
    src.add_argument("--directed", action="store_true", help="Рассматривать входную матрицу как ориентированную (по умолчанию да)")
    src.add_argument("--undirected", action="store_true", help="Симметризовать матрицу (игнорируется, если указан --directed)")

    rnd = p.add_argument_group("Случайный граф")
    rnd.add_argument("--random", action="store_true", help="Сгенерировать случайный граф вместо CSV")
    rnd.add_argument("--n", type=int, default=10, help="Количество вершин (для --random)")
    rnd.add_argument("--sparse", action="store_true", help="Создать разреженный граф (по умолчанию полный)")
    rnd.add_argument("--edges", type=int, default=None, help="Количество рёбер для --sparse")
    rnd.add_argument("--low", type=int, default=1, help="Минимальный вес")
    rnd.add_argument("--high", type=int, default=100, help="Максимальный вес")
    rnd.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")

    aco = p.add_argument_group("Параметры ACO")
    aco.add_argument("--alpha", type=float, default=1.0, help="Влияние феромона")
    aco.add_argument("--beta", type=float, default=3.0, help="Влияние эвристики 1/d")
    aco.add_argument("--rho", type=float, default=0.5, help="Испарение (0..1)")
    aco.add_argument("--q", type=float, default=1.0, help="Масштаб депонирования")
    aco.add_argument("--elitist", type=float, default=0.0, help="Вес элитного доосаждения (0 отключить)")
    aco.add_argument("--tau0", type=float, default=0.1, help="Начальная концентрация феромона")
    aco.add_argument("--ants", type=int, default=None, help="Количество муравьёв (по умолчанию >= n)")
    aco.add_argument("--iters", type=int, default=200, help="Число итераций")
    aco.add_argument("--start", type=int, default=0, help="Стартовая вершина (None = случайный старт)")
    aco.add_argument("--early-stop", type=int, default=None, help="Ранний стоп после N итераций без улучшения")

    out = p.add_argument_group("Вывод")
    out.add_argument("--save-best", type=str, default=None, help="Сохраняет лучший тур в файл (txt)")

    return p


def main(argv: list[str] | None = None) -> int:
    '''Точка входа для ant_colony_tsp'''
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Построение графа
    if args.csv and not args.random:
        g = Graph.from_csv(args.csv, directed=(not args.undirected))
    else:
        if not args.random:
            args.random = True
        if args.sparse:
            if args.edges is None:
                parser.error("--edges обязательно для --sparse")
            g = GraphFactory.random_sparse(args.n, args.edges, directed=True, low=args.low, high=args.high, seed=args.seed)
        else:
            g = GraphFactory.random_complete(args.n, directed=True, low=args.low, high=args.high, seed=args.seed)

    params = ACOParams(alpha=args.alpha, beta=args.beta, rho=args.rho, q=args.q, elitist_weight=args.elitist, tau0=args.tau0)
    start_vertex = None if args.start is None else int(args.start)
    colony = AntColony(g, params=params, n_ants=args.ants, n_iterations=args.iters, start_vertex=start_vertex, seed=args.seed, early_stop=args.early_stop)
    res = colony.run()

    print("\nЛучший маршрут:", " -> ".join(map(str, res.best_tour)))
    print("Стоимость:", int(res.best_cost) if float(res.best_cost).is_integer() else res.best_cost)
    print("Итераций:", res.iterations)

    if args.save_best:
        Path(args.save_best).write_text(" ".join(map(str, res.best_tour)), encoding="utf-8")
        print("Сохранено:", args.save_best)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
