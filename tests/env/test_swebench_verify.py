"""Drift guard for the SWE-bench report-schema constants that src/env/swebench_verify.py
hardcodes (to keep the real, datasets-pulling swebench import out of module load).

Marked ``integration`` because it imports the real swebench package; it runs under
``pytest -m integration``, not in the fast unit suite.
"""

import pytest

from src.env import swebench_verify as swe_verify


@pytest.mark.integration
def test_swebench_report_constants_match_installed_package():
    """The report-schema strings hardcoded in the harness must track the pinned
    swebench package; a version bump that changes them would break grading."""
    from swebench.harness import constants

    assert swe_verify.KEY_INSTANCE_ID == constants.KEY_INSTANCE_ID
    assert swe_verify.KEY_MODEL == constants.KEY_MODEL
    assert swe_verify.KEY_PREDICTION == constants.KEY_PREDICTION
    assert swe_verify._LOG_REPORT == constants.LOG_REPORT
    assert swe_verify._LOG_TEST_OUTPUT == constants.LOG_TEST_OUTPUT
    assert constants.FAIL_TO_PASS == "FAIL_TO_PASS"
    assert constants.PASS_TO_PASS == "PASS_TO_PASS"
