"""Repository-wide pytest configuration, applied to every collected test (tests/, benchmarks/, flatland/)."""
import os

# RailEnv's step() pre/postcondition checks are off by default outside tests (see
# flatland.envs.rail_env.default_check_step_pre_post_conditions). Run the whole test suite with them on, unless the
# environment variable is already set explicitly, e.g. to 0 for a timing run.
os.environ.setdefault("FLATLAND_CHECK_STEP_PRE_POST_CONDITIONS", "1")
