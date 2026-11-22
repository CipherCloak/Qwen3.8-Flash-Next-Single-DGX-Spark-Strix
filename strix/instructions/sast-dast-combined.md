# Combined white-box assessment: source code + running application

<!--
Template for `strix --instruction-file`. Fill the bracketed fields, delete what
does not apply. Written for a self-hosted model: it asks for short, targeted
tool calls because every tool output stays in the agent's context until the
next compaction, and context length is what costs time on one DGX Spark.
-->

## Authorization and scope

- This is an authorized assessment of an application we own: [APP NAME].
- Source code: the local directory target ([REPO PATH]). Running instance: the URL
  target ([STAGING URL]). Both are the same application at [GIT REF / VERSION].
- In scope: [hosts, ports, path prefixes]. Out of scope: [third-party services,
  payment providers, production, anything not listed above].
- Do not run denial-of-service, brute-force beyond [N] attempts per account,
  mass scanning of other hosts, or anything that deletes or corrupts data.
  Prefer read-only proofs (read a record, echo a marker) over destructive ones.
- Test accounts: [user / password per role]. Do not create other accounts unless
  a finding requires it, and name them `strix-test-*`.

## How to work (SAST and DAST together)

1. **Map the code first.** Find the framework, entry points, routes/handlers,
   auth and session handling, data access, and every place user input reaches a
   sink (SQL, shell, file paths, templates, deserialization, outbound requests,
   redirects). Build a route list with the handler file:line for each.
2. **Confirm against the running app.** For each candidate from the code, send
   the smallest request that proves or disproves it on [STAGING URL]. A finding
   is confirmed only when the running app shows the behaviour; a code-only
   finding is reported as "unconfirmed (code)" with the reason it could not be
   exercised.
3. **Chain.** When one finding unlocks another (a leaked secret, a forged
   session, a file read), follow it, and report the chain as one finding plus
   its parts.
4. **Cover the usual classes** for this stack: injection (SQL/NoSQL/command/
   template), broken access control and IDOR, authentication and session flaws,
   path traversal and file handling, SSRF, XSS, deserialization, secrets in code
   or config, weak crypto, security misconfiguration, vulnerable dependencies.

## Keep the context small (important on a local model)

- Read files with line ranges or `grep -n` with a pattern; never `cat` whole
  large files, lockfiles, minified bundles, binaries or logs.
- Cap command output (`| head -n 100`, `--max-count`, `-q`), and do not repeat a
  read whose result you already have.
- Prefer one precise request over a sweep; no full-wordlist fuzzing unless a
  specific parameter justifies it.
- Delegate a whole area (e.g. "auth and sessions", "file handling") to a
  sub-agent with a precise brief instead of exploring everything in one agent.
- Run at most 4 sub-agents at the same time; prefer delegating in sequence over
  spawning more. The model server runs 4 requests at once and queues the rest.
- Write intermediate notes (route map, candidates, confirmed findings) to files
  in the workspace and re-read them, rather than carrying them in conversation.

## Report

For every finding: title, severity (CVSS 3.1 vector), affected route and
code location (file:line), root cause, the exact request(s) that reproduce it
and the observed response, impact, and a concrete fix (code-level where
possible). Mark each one confirmed (dynamic) or unconfirmed (code). End with a
short list of what was checked and found clean, and what could not be tested.
