"""Orchestrator agent: an Anthropic tool-use loop over the calibration tools.

The model decides, per folder, which frame is the calibration grid, measures it,
cross-checks the result, matches it to tissue by magnification and date, and
either records the result or flags the case for review with a written
explanation.

Folders are processed in batches, each in its own conversation. A single
conversation covering hundreds of folders would grow past the context window
part-way through a run and fail after hours of work; batching also means a
failure costs one batch rather than everything.
"""

from __future__ import annotations

import json
from typing import Iterator

from .tools import TOOL_DEFINITIONS, ToolBox

DEFAULT_BATCH_SIZE = 10
MAX_TURNS_PER_BATCH = 120

SYSTEM_PROMPT = """\
You are a calibration measurement agent for electron microscopy kidney biopsy data.

For each specimen acquisition folder:
1. Identify which TIFF frames are calibration diffraction grating images (vs tissue).
2. Measure the calibration grid spacing and obtain nm/pixel. The measurement tool
   applies the standard: nm/pixel = (0.4629630 / D) * 1000, where 0.4629630 um is
   the TedPella 607 grating pitch (2160 lines/mm) and D is pixels per grid space.
3. Cross-check the measurement, then match the calibration to the tissue by
   magnification and acquisition date.
4. Save the result, or flag the case for review.

MEASUREMENT RULES - these are not negotiable:
- Thumbnail detection scores tell you WHICH frame is the grid. They are never a
  measurement. nm/pixel must always come from measure_grid on the full TIFF.
- If measure_grid returns valid=false, there is no measurement. Do not derive a
  number from the confidence, the thumbnail, or the embedded metadata. Flag it.
- Always call cross_check_measurement before save_result. If it reports a
  suspected_harmonic_factor, the measurement is wrong by that integer factor  - 
  flag the case as harmonic_suspect with high priority. Never "correct" the
  number yourself by multiplying or dividing it.
- If measure_grid warns that the fundamental was inferred, flag for review even
  when the cross-checks pass.

MATCHING RULES (lab SOP):
- The calibration frame must have the same resolution as the tissue frame.
  nm/pixel describes a pixel grid, so a calibration measured on a 2512px frame
  does not apply to a 1024px frame even at the same magnification. Use
  choose_calibration_frame, which enforces this; never pair them by hand.
- Always pass image_width and image_height to save_result.
- Use choose_calibration_frame to apply the SOP; do not do the date arithmetic
  yourself.
- Prefer a calibration frame from the same folder at the same magnification.
- Otherwise accept one from the same date at the same magnification.
- If none is close in date, flag it ("ask Frida/Zour").

DECISION GUIDELINES:
- One candidate with score > 0.90: download the full TIFF, measure, cross-check, save.
- Multiple candidates: match magnification to the tissue frames, measure that one.
- Scores disagree with the FFT measurement, or the measurement warns about
  non-orthogonal axes or unequal spacings: flag as low_confidence.
- No candidate in the folder: search cross-account by magnification and date;
  use a match within 7 days, flag if 7-30 days, flag high if none within 30 days.

For every case write a clear agent_notes string: which frame was used, the
measured D and nm/pixel with its uncertainty, the cross-check outcome, and any
concerns. Prefer flagging with a good explanation over guessing. After each
case, call cleanup_temp_files. When every folder in the batch is done, reply
with a short plain-text summary.
"""


def run_agent(
    client,
    toolbox: ToolBox,
    folder_paths: list[str],
    model: str,
    max_tokens: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> str:
    """Process every folder, one batch per conversation. Returns a run summary."""
    summaries: list[str] = []
    for index, batch in enumerate(_batches(folder_paths, batch_size), start=1):
        try:
            summaries.append(
                f"[batch {index}] " + _run_batch(client, toolbox, batch, model, max_tokens)
            )
        except Exception as error:
            # A failed batch leaves its cases 'pending', so a rerun retries them.
            summaries.append(f"[batch {index}] FAILED: {type(error).__name__}: {error}")
    return "\n".join(summaries)


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


def _run_batch(
    client, toolbox: ToolBox, folder_paths: list[str], model: str, max_tokens: int
) -> str:
    messages = [
        {
            "role": "user",
            "content": "Process these specimen folders:\n" + json.dumps(folder_paths, indent=2),
        }
    ]

    for _ in range(MAX_TURNS_PER_BATCH):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return _final_text(response)

        messages.append({"role": "user", "content": _run_tools(toolbox, response.content)})

    return f"stopped after {MAX_TURNS_PER_BATCH} turns without finishing {folder_paths}"


def _run_tools(toolbox: ToolBox, blocks: list) -> list[dict]:
    results = []
    for block in blocks:
        if block.type == "tool_use":
            output = toolbox.dispatch(block.name, dict(block.input))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, default=str),
                    "is_error": bool(isinstance(output, dict) and output.get("error")),
                }
            )
    return results


def _final_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
