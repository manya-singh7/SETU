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

- At the end of every task, give a short report with:
    1. What you changed (list every file).
    2. Real test output (paste the final pytest / npm lines).
    3. Anything not done, not working, or needing another session.
    4. Any assumption you made or anything I should check by eye.

- Several sessions share this laptop. Run only the tests for the files
  you changed. Don't run the full suite unless the task asks for it.