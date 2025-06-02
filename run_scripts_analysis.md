# Tekton Run Scripts Analysis

## Summary

All Tekton component run scripts have been checked for irregularities in launch patterns. The standard pattern is `python -m module.api.app`.

## Findings

### Components Following Standard Pattern ✅

1. **Apollo** - `python -m apollo.api.app`
2. **Athena** - `python -m athena.api.app --port $ATHENA_PORT`
3. **Budget** - `python -m budget.api.app`
4. **Ergon** - `python -m ergon.api.app`
5. **Harmonia** - `python -m harmonia.api.app --port $HARMONIA_PORT`
6. **Hermes** - `python -m hermes.api.app`
7. **Metis** - `python -m metis.api.app`
8. **Prometheus** - `python -m prometheus.api.app`
9. **Rhetor** - `python -m rhetor.api.app`
10. **Sophia** - `python -m sophia.api.app --port $SOPHIA_PORT`
11. **Synthesis** - `python -m synthesis.api.app --port $SYNTHESIS_PORT`
12. **Telos** - `python -m telos.api.app`
13. **Terma** - `python -m terma.api.app --port $TERMA_PORT`
14. **Tekton-Core** - `python -m tekton.api.app`

### Non-Standard Launch Patterns ❌

1. **Engram** - Uses `python -m engram.api.consolidated_server --port $ENGRAM_PORT`
   - Not following the standard `app.py` pattern
   - Uses `consolidated_server` instead

2. **Hephaestus** - Uses `python3 ui/server/server.py --port "$PORT"`
   - Uses `python3` instead of `python`
   - Directly runs a script file instead of using module notation
   - Server is in `ui/server/` directory, not following standard structure

## Observations

### Port Handling Consistency
Some components pass the port as a command-line argument (`--port $PORT`), while others don't:
- With --port flag: Athena, Engram, Harmonia, Sophia, Synthesis, Terma
- Without --port flag: Apollo, Budget, Ergon, Hermes, Metis, Prometheus, Rhetor, Telos, Tekton-Core

### No Components Found Using:
- `uvicorn` directly in shell scripts
- `socket_server` module
- Any other non-standard server implementations

## Recommendations

1. **Engram**: Consider refactoring to use standard `engram.api.app` pattern
2. **Hephaestus**: 
   - Change to use `python -m` notation
   - Consider moving server to standard location following `component.api.app` pattern
3. **Port Handling**: Standardize whether ports should be passed as command-line arguments or rely solely on environment variables

All other components are following the standard launch pattern correctly.