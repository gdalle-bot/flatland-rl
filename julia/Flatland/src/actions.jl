"""
    RailEnvAction

Action of an agent in a step, numbered as `flatland.envs.rail_env_action.RailEnvActions` in Python:
`DO_NOTHING = 0`, `MOVE_LEFT = 1`, `MOVE_FORWARD = 2`, `MOVE_RIGHT = 3`, `STOP_MOVING = 4`.
"""
@enum RailEnvAction::UInt8 begin
    DO_NOTHING = 0
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4
end

const ALL_ACTIONS = (DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING)
