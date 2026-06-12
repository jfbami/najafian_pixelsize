"""Orchestrator agent: an Anthropic tool-use loop over the calibration tools.

The model decides, per folder, which frame is the calibration grid, measures it,
matches it to tissue by magnification and date, and either records the result or
flags the case for review with a written explanation.
"""

from __future__ import annotations

import json

from .tools import TOOL_DEFINITIONS, ToolBox

SYSTEM_PROMPT = """\
You are a calibration measurement agent for electron microscopy kidney biopsy data.

For each specimen acquisition folder:
1. Identify which TIFF frames are calibration diffraction grating images (vs tissue).
2. Measure the calibration grid spacing and obtain nm/pixel. The measurement tool
   applies the standard: nm/pixel = (0.463 / D) * 1000, where 0.463 um is the
   TedPella 607 grating pitch (2160 lines/mm) and D is pixels per grid space.
3. Match the calibration to the tissue by magnification and acquisition date.
4. Save the result, or flag the case for review.

MATCHING RULES (lab SOP):
- Prefer a calibration frame from the same folder at the same magnification.
- Otherwise accept one from the same date at the same magnification.
- If none is close in date, flag it ("ask Frida/Zour").

DECISION GUIDELINES:
- One candidate with score > 0.90: download the full TIFF, measure, save.
- Multiple candidates: match magnification to the tissue frames, measure that one.
- Scores disagree with the FFT measurement, or the measurement warns about
  non-orthogonal axes or unequal spacings: flag as low_confidence.
- No candidate in the folder: search cross-account by magnification and date;
  use a match within 7 days, flag if 7-30 days, flag high if none within 30 days.

For every case write a clear agent_notes string: which frame was used, the
measured D and nm/pixel, and any concerns. Prefer flagging with a good
explanation over guessing. After each case, call cleanup_temp_files.
"""


def run_agent(client, toolbox: ToolBox, folder_paths: list[str], model: str, max_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": "Process these specimen folders:\n" + json.dumps(folder_paths, indent=2),
        }
    ]

    while True:
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
                }
            )
    return results


def _final_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
