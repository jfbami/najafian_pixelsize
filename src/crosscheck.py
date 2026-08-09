"""Independent checks on a measured nm/pixel.

The FFT measurement is self-contained, which means a mistake in it has nothing
to contradict it. Two cheap, independent references already exist in the data
and were previously parsed but never used:

1. **The microscope's own embedded scale** (`XpixCal` in the TIFF description).
   Its unit encoding is unreliable, so it cannot be trusted as a value -- but
   after normalising away powers of ten, the *ratio* to the measured value
   should be ~1. A ratio near a small integer (2, 3, 4) is the signature of a
   harmonic mis-identification, which is precisely the failure mode that is
   otherwise silent.

2. **The magnification scaling law.** For one camera on one scope,
   nm/pixel is inversely proportional to magnification, so
   `nm_per_pixel * magnification` is a constant. A frame that breaks that
   relation by a large factor is wrong regardless of how confident its
   spectrum looked.

Neither check can prove a measurement right; both are good at catching the
integer-factor blunder. They return findings, not verdicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# Ratios this close to a small integer are reported as a suspected harmonic error.
_INTEGER_RATIO_TOLERANCE = 0.06
_SUSPECT_FACTORS = (2, 3, 4, 5)
# Agreement within this fraction counts as confirmation.
_AGREEMENT_TOLERANCE = 0.05
# Fractional deviation from the fitted magnification law that trips a warning.
_MAGNIFICATION_LAW_TOLERANCE = 0.25


@dataclass
class CrossCheck:
    name: str
    passed: bool
    detail: str
    suspected_factor: Optional[float] = None


def check_embedded_scale(
    measured_nm_per_pixel: float,
    embedded_value: Optional[float],
    embedded_unit: Optional[str],
    trusted: bool = False,
) -> CrossCheck:
    """Compare against the microscope's own recorded pixel scale.

    Disabled unless `trusted` is set, because on this lab's data the field is
    not a scale at all. Every frame carries `XpixCal=150.000000`,
    `YpixCal=150.000000`, `Unit=um` -- identical across magnifications, and
    matching no unit reading of the true value (a frame measured at 4.234
    nm/pixel is 35x, 2.4x or 1.6x off depending on how 150 is interpreted).
    It is an unconfigured template default.

    Left enabled it was worse than useless: the 2.8x residual reads as a
    suspected 3x harmonic error, so *every correct frame* would be flagged as
    a harmonic blunder. Turn it on with `crosscheck.trust_embedded_scale` only
    after confirming the field tracks magnification on your instrument.

    The unit is treated as unknown: the embedded value is rescaled by whatever
    power of ten brings it closest to the measurement, so only a non-decimal
    discrepancy -- an integer factor -- can fail the check.
    """
    if not trusted:
        return CrossCheck(
            "embedded_scale", True, "embedded scale not trusted on this instrument"
        )
    if embedded_value is None or not math.isfinite(measured_nm_per_pixel):
        return CrossCheck("embedded_scale", True, "no embedded scale to compare")
    if embedded_value <= 0.0 or measured_nm_per_pixel <= 0.0:
        return CrossCheck("embedded_scale", True, "embedded scale unusable")

    ratios = _candidate_ratios(measured_nm_per_pixel, embedded_value)
    unit_note = f" (unit '{embedded_unit}' ignored)" if embedded_unit else ""

    agreement = min(ratios, key=lambda value: abs(math.log(value)))
    if abs(agreement - 1.0) <= _AGREEMENT_TOLERANCE:
        return CrossCheck(
            "embedded_scale",
            True,
            f"agrees with embedded scale within {abs(agreement - 1.0) * 100:.1f}%{unit_note}",
        )

    # An integer factor may appear at a different decade than the closest one:
    # 4x looks like 0.4x once rounded to the nearest power of ten, which would
    # hide exactly the error this check exists to find.
    factor = next(
        (f for f in (_nearest_integer_factor(r) for r in ratios) if f is not None), None
    )
    ratio = agreement
    if factor is not None:
        return CrossCheck(
            "embedded_scale",
            False,
            f"measured value is {factor:g}x the microscope's own scale{unit_note}; "
            f"this is the signature of a harmonic mis-identification",
            suspected_factor=factor,
        )
    return CrossCheck(
        "embedded_scale",
        False,
        f"disagrees with embedded scale by {(ratio - 1.0) * 100:+.1f}%{unit_note}",
    )


def check_magnification_law(
    measured_nm_per_pixel: float,
    magnification: Optional[int],
    reference: Sequence[tuple[int, float]],
) -> CrossCheck:
    """Compare against nm/pixel * magnification from other accepted frames.

    `reference` is (magnification, nm_per_pixel) pairs from measurements already
    trusted -- typically the calibration cache for the same instrument.
    """
    if magnification is None or magnification <= 0:
        return CrossCheck("magnification_law", True, "no magnification recorded")
    if not math.isfinite(measured_nm_per_pixel) or measured_nm_per_pixel <= 0.0:
        return CrossCheck("magnification_law", True, "no measurement to check")

    constants = [
        mag * nm
        for mag, nm in reference
        if mag and nm and mag > 0 and math.isfinite(nm) and nm > 0
    ]
    if not constants:
        return CrossCheck("magnification_law", True, "no reference frames yet")

    expected = _median(constants)
    observed = magnification * measured_nm_per_pixel
    ratio = observed / expected

    if abs(ratio - 1.0) <= _MAGNIFICATION_LAW_TOLERANCE:
        return CrossCheck(
            "magnification_law",
            True,
            f"consistent with {len(constants)} reference frames "
            f"({(ratio - 1.0) * 100:+.1f}%)",
        )

    factor = _nearest_integer_factor(ratio)
    detail = (
        f"nm/pixel x magnification = {observed:.0f} vs {expected:.0f} expected "
        f"from {len(constants)} reference frames ({(ratio - 1.0) * 100:+.1f}%)"
    )
    if factor is not None:
        detail += f"; ratio is ~{factor:g}x, suggesting a harmonic error"
    return CrossCheck("magnification_law", False, detail, suspected_factor=factor)


def summarize(checks: Sequence[CrossCheck]) -> tuple[bool, str]:
    """Collapse checks into (all_passed, human-readable summary)."""
    failures = [check for check in checks if not check.passed]
    if not failures:
        return True, "; ".join(check.detail for check in checks) or "no checks run"
    return False, "; ".join(f"{check.name}: {check.detail}" for check in failures)


def _candidate_ratios(measured: float, embedded: float) -> list[float]:
    """measured / embedded under every plausible unit rescaling.

    The embedded unit is unreliable, so each power-of-ten interpretation is a
    candidate. Only the decades that put the two within a couple of orders of
    magnitude are worth considering.
    """
    if measured <= 0.0 or embedded <= 0.0:
        return [1.0]
    centre = round(math.log10(measured / embedded))
    return [measured / (embedded * 10.0**decade) for decade in (centre - 1, centre, centre + 1)]


def _nearest_integer_factor(ratio: float) -> Optional[float]:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None
    for factor in _SUSPECT_FACTORS:
        if abs(ratio - factor) <= _INTEGER_RATIO_TOLERANCE * factor:
            return float(factor)
        if abs(ratio - 1.0 / factor) <= _INTEGER_RATIO_TOLERANCE / factor:
            return round(1.0 / factor, 4)
    return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
