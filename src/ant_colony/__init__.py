"""Оптимизация для TSP (симметричного и асимметричного).

Пакет предоставляет:
- ant_colony.graph: классы Graph, Vertex, утилиты генерации/загрузки графов
- ant_colony.ants: классы Ant, AntColony
- ant_colony.cli: CLI для запуска из терминала
"""
from .ants import ACOParams, AntColony, RunResult
from .graph import Graph, GraphFactory, Vertex

__all__ = ["Graph", "Vertex", "GraphFactory", "AntColony", "ACOParams", "RunResult"]
