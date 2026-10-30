# My Agent Project Rules

## Architecture

- Keep provider adapters responsible for vendor request/response translation and error normalization.
- Keep retries, context assembly, tool dispatch, approvals, and turn lifecycle in the harness.
- CLI and Web clients use AgentService/AppServer and typed events; do not add a separate agent loop to either UI.
- Preserve checkpoint-before-action semantics and explicit recovery of unknown tool outcomes.
- Project rules are guidance, not permissions. Do not bypass permission checks or OS sandboxing.

## Validation

- Run `.venv/bin/python -m unittest discover -s tests` for Python changes.
- Run `node --check my_agent/web/app.js` for Web JavaScript changes.
- The browser smoke test is `node scripts/check_web_ui.cjs`; it needs Playwright available on `NODE_PATH` and Chrome.
- Use isolated workspaces and scripted providers in regressions. Do not call paid model APIs without explicit authorization.
- A successful command is evidence, not proof that the whole task is complete.

## Data and Scope

- Never commit API keys, credentials, or `.my-agent` runtime data.
- Do not delete, reset, or silently migrate user conversations as a UI workaround.
- Keep changes focused and independently testable. Preserve unrelated local work.
