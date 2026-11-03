# Rules for coding agents working on SETU

- Never run git commands, and never read git history or other branches.
- Only edit the files your task lists. If another file needs a change,
  describe it in your summary instead.
- Never read or print .env or any API key.
- Run `python -m pytest tests -m "not slow"` (and `npm test` /
  `npm run build` in frontend/ for frontend work) before reporting,
  and paste the real output.
- Honesty rules for SETU:
  - Never invent numbers, results, or sample data presented as real.
  - Never label a fallback or heuristic result as coming from SAM or
    Muse Spark.
  - A "weak" or "failed" verdict is never described as an anomaly.
  - SETU never says a bridge is safe; it flags readings for engineer
    review.
- Keep public function names and required arguments unchanged unless
  the task says otherwise.

  - When running pytest, always add --basetemp=.tmp/<name>, where
  <name> is a short random name you pick once per session (for
  example .tmp/k7q2). Never share a folder with another session.