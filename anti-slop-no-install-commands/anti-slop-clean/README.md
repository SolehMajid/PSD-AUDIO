# Anti-Slop — Script-Free Skills Pack

This package contains the human-readable AI skill instructions and Markdown reference files only.

## Deliberately excluded
- JavaScript/Node.js files (`.js`, `.mjs`, `.cjs`)
- Python files (`.py`)
- CLI installers, package manifests, plugin launchers, and automation scripts
- GitHub workflows and repository maintenance scripts

## Contents
The `skills/` directory contains the skill definitions (`SKILL.md`), Markdown references, license notices, and the version marker. The contrast-checking skill has been adapted to use its documented WCAG formula and reference table instead of running Python.

## Usage
Read the relevant `SKILL.md` and Markdown reference files as guidance. This archive does not install anything, modify your environment, or execute code.

## Security note
Removing scripts reduces executable-code exposure; it does not guarantee that the natural-language instructions are appropriate for every project. Review the skill instructions before enabling them.
