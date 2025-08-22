
"""Ант-колони оптимизация для TSP (симметричного и асимметричного).

Пакет предоставляет:
- ant_colony.graph: классы Graph, Vertex, утилиты генерации/загрузки графов
- ant_colony.ants: классы Ant, AntColony
- ant_colony.cli: CLI для запуска из терминала

Совместим с Python 3.10+.
"""
from .graph import Graph, Vertex, GraphFactory
from .ants import AntColony, ACOParams, RunResult
__all__ = ["Graph", "Vertex", "GraphFactory", "AntColony", "ACOParams", "RunResult"]
