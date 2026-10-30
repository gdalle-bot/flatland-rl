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

@doc "Keep the current speed; when leaving the cell, go where `MOVE_FORWARD` would (action `0`)." DO_NOTHING
@doc "Take the left branch when leaving the cell, or go forward where there is none (action `1`)." MOVE_LEFT
@doc "Accelerate, and go straight on when leaving the cell (action `2`)." MOVE_FORWARD
@doc "Take the right branch when leaving the cell, or go forward where there is none (action `3`)." MOVE_RIGHT
@doc "Brake (action `4`)." STOP_MOVING
