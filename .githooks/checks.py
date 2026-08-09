"""Enforce the repository conventions from AGENTS.md at commit time.

Run by the git hooks in this directory. Two independent checks:

    python .githooks/checks.py --staged
        Rejects em dashes (U+2014) and en dashes (U+2013) in staged text
        files, and rejects hand edits to generated files.

    python .githooks/checks.py --message .git/COMMIT_EDITMSG
        Rejects an AI agent named in a Co-Authored-By trailer, and rejects
        the banned dashes in the commit message itself.

Both exit 0 when clean and 1 with an explanation when not.
Bypass a hook with `git commit --no-verify` when you genuinely need to.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Built from codepoints so this file does not contain the characters it bans,
# which would otherwise make the hook reject its own source.
BANNED_DASHES = {chr(0x2014): "em dash (U+2014)", chr(0x2013): "en dash (U+2013)"}

TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".yaml", ".yml", ".sql", ".txt", ".cfg", ".toml", ".ini", ".json"}
)

GENERATED_FILES = frozenset({"CHANGELOG.md"})
GENERATED_MARKER = re.compile(r"auto-?generated|do not edit", re.IGNORECASE)

AGENT_COAUTHOR = re.compile(
    r"^\s*co-authored-by:.*\b(claude|anthropic|gpt|openai|copilot|cursor|codex|gemini)\b",
    re.IGNORECASE | re.MULTILINE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--message", metavar="FILE")
    arguments = parser.parse_args()

    if arguments.staged:
        return check_staged_files()
    if arguments.message:
        return check_commit_message(arguments.message)
    parser.error("give --staged or --message FILE")
    return 1


def check_staged_files() -> int:
    problems: list[str] = []
    for path in staged_paths():
        content = staged_content(path)
        if content is None:
            continue
        if is_generated(path, content):
            problems.append(
                f"{path}: generated file edited by hand; regenerate it instead "
                f"(AGENTS.md, 'Files that must not be edited by hand')"
            )
        problems.extend(dash_problems(path, content))

    return report(problems, "Commit rejected by .githooks/pre-commit")


def check_commit_message(message_path: str) -> int:
    try:
        with open(message_path, encoding="utf-8", errors="replace") as handle:
            message = handle.read()
    except OSError as error:
        print(f"could not read commit message: {error}", file=sys.stderr)
        return 1

    body = strip_comments(message)
    problems = list(dash_problems("commit message", body))
    for match in AGENT_COAUTHOR.finditer(body):
        problems.append(
            f"commit message:{line_of(body, match.start())}: "
            f"AI agent named as co-author: {match.group(0).strip()!r} "
            f"(AGENTS.md, 'Commit conventions')"
        )

    return report(problems, "Commit rejected by .githooks/commit-msg")


def staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def staged_content(path: str) -> str | None:
    """Content as staged, which may differ from the working tree."""
    if not any(path.lower().endswith(suffix) for suffix in TEXT_SUFFIXES):
        return None
    result = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_generated(path: str, content: str) -> bool:
    if path.split("/")[-1] in GENERATED_FILES:
        return True
    return bool(GENERATED_MARKER.search(content[:500]))


def dash_problems(path: str, content: str):
    for number, line in enumerate(content.splitlines(), start=1):
        for character, name in BANNED_DASHES.items():
            column = line.find(character)
            if column >= 0:
                yield (
                    f"{path}:{number}:{column + 1}: {name}; use a plain '-' "
                    f"(AGENTS.md, 'Writing conventions')\n    {line.strip()}"
                )


def strip_comments(message: str) -> str:
    return "\n".join(
        line for line in message.splitlines() if not line.startswith("#")
    )


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def report(problems: list[str], heading: str) -> int:
    if not problems:
        return 0
    print(f"\n{heading}:\n", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        f"\n{len(problems)} problem(s). Fix them, or bypass with "
        f"'git commit --no-verify' if you are certain.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
