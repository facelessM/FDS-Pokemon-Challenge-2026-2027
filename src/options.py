"""Run-wide switches.

FEATURE_FIXES turns on four corrections we found after the first selected
submission (see docs/methodology.md, "Feature corrections"):

1. Move accuracy is stored as a fraction (0.85), not a percentage (85).
2. P2 knockout scores are recalibrated, because P2 knockouts are mostly unrecorded.
3. Knockout risk is passed to the tactical view for every P2 species, not only common ones.
4. Validation and test rows average the inner models instead of one model refitted
   on the whole training fold, so they see estimates of the same quality as training rows.

Without the switch the code reproduces the original submission exactly.
run.py sets the variable with --feature-fixes before any module is imported.
"""

import os

FEATURE_FIXES = os.environ.get("FDS_FEATURE_FIXES") == "1"
