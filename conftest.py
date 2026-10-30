"""Repository-wide pytest configuration, applied to every test collected under the repository root."""
import os

CHECK_STEP_PRE_POST_CONDITIONS_ENV_VAR = "FLATLAND_CHECK_STEP_PRE_POST_CONDITIONS"

# RailEnv's step() pre/postcondition checks are off by default outside tests (see
# flatland.envs.rail_env.default_check_step_pre_post_conditions). Run the tests with them on, unless the environment
# variable is set to a non-empty value already, e.g. 0 for a timing run.
if not os.environ.get(CHECK_STEP_PRE_POST_CONDITIONS_ENV_VAR):
    os.environ[CHECK_STEP_PRE_POST_CONDITIONS_ENV_VAR] = "1"


def pytest_report_header(config):
    return f"{CHECK_STEP_PRE_POST_CONDITIONS_ENV_VAR}={os.environ[CHECK_STEP_PRE_POST_CONDITIONS_ENV_VAR]}"
