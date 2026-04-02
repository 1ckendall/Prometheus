---
name: Schema versioning discipline
description: Rules for when to update schema version numbers in file formats
type: feedback
---

Never increment a schema version number unless the user explicitly asks for it.

**Why:** The user wants to control when schema versions change — bumping a version is a deliberate decision that may affect backwards compatibility and user tooling.

**How to apply:** When making changes to a file format (JSON, TOML, CSV, etc.), describe the structural changes to the user and ask whether a schema version update is needed, rather than deciding unilaterally. Present the diff of the format change and let the user decide.
