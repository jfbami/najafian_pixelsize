"""Physical constants for the calibration standard.

The grid is a TedPella Prod. No. 607 cross-grating replica: 2160 lines/mm,
which is 0.463 micrometers per single grid space. This value is the anchor of
every measurement and must not be changed without a new physical standard.
"""

GRID_STANDARD_NAME = "TedPella Prod. No. 607"
GRID_LINES_PER_MM = 2160
GRID_PITCH_UM = 0.463
NM_PER_UM = 1000.0


def nm_per_pixel(pixels_per_space: float) -> float:
    """Convert measured grid spacing in pixels to nanometers per pixel."""
    return (GRID_PITCH_UM / pixels_per_space) * NM_PER_UM
