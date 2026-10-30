from collections import deque
from typing import Dict, List, Generic, TypeVar

from flatland.core.entry_point_distance_map import EntryPointDistanceMap
from flatland.core.transition_map import TransitionMap

EntryPointDistanceMapT = TypeVar('EntryPointDistanceMapT', bound=EntryPointDistanceMap)
TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
EntryPointT = TypeVar('EntryPointT')


class DistanceMapWalker(Generic[EntryPointDistanceMapT, TransitionMapT, EntryPointT]):
    """
    "All-to-any-one-in-cluster": utility class to compute distance maps from each entry point in the rail network (cell and each possible orientation within it in grid case)
     to any one in the set of target entry points using backwards BFS. Agnostic of any agent/target_nr - operates
     purely in terms of entry points and returns the distances it finds.
    """

    def __init__(self, distance_map: EntryPointDistanceMap):
        self.distance_map = distance_map

    def _distance_map_walker(self,
                             rail: TransitionMapT,
                             target_entry_points: List[EntryPointT]
                             ) -> Dict[EntryPointT, int]:
        """
        Distance from every entry point that can reach any of the target entry points to the nearest of
        them, by one backward breadth-first search started from all target entry points at distance 0.

        A multi-source search gives the same distances as one search per target entry point followed by
        the minimum over them: a path to one target through another target is never shorter than the path to
        that other target. Each entry point is visited at most once, so this also terminates on cyclic/looped
        rail layouts.

        Parameters
        ----------
        rail
            the transition map to walk backwards
        target_entry_points
            the target entry points (distance 0)

        Returns
        -------
        Dict[EntryPointT, int]
            distance to the nearest target entry point, for every entry point backwards-reachable from any of
            them (the target entry points included).
        """
        distances: Dict[EntryPointT, int] = {}
        for target_entry_point in target_entry_points:
            distances[target_entry_point] = 0
        nodes_queue = deque(distances)
        while nodes_queue:
            entry_point = nodes_queue.popleft()
            next_distance = distances[entry_point] + 1
            for predecessor_entry_point in rail.get_predecessor_entry_points(entry_point):
                if predecessor_entry_point not in distances:
                    distances[predecessor_entry_point] = next_distance
                    nodes_queue.append(predecessor_entry_point)
        return distances
