import math
from typing import List, Optional, Generic, TypeVar, Callable

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
DistanceMapT = TypeVar('DistanceMapT')
EntryPointT = TypeVar('EntryPointT')
WaypointT = TypeVar('WaypointT')


def _infinite_distance():
    # kept so that pickles holding the former `distances` defaultdict (which used it as factory) still load
    return math.inf


class EntryPointDistanceMap(Generic[TransitionMapT, DistanceMapT, EntryPointT, WaypointT]):
    """
    Base distance map: the agents and the rail the distances refer to. `DistanceMapWalker` computes distances to a
    set of target entry points - agnostic of any numeric target_nr (agent handle), which it has no notion of.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[EntryPointT], WaypointT]):
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None
        self.waypoint_init = waypoint_init

    def reset(self, agents: List[EnvAgent], rail: TransitionMapT):
        """
        Reset the distance map
        """
        self.agents: List[EnvAgent] = agents
        self.rail = rail
