"""
simulation/app.py
──────────────────────────────────────────────────────────────────────────────
FastAPI application layer for the Nexus-Omni Simulator.

Start from the PROJECT ROOT (one level above simulation/):
  uvicorn simulation.app:app --host 0.0.0.0 --port 8080 --reload

Endpoints
─────────
  GET  /            – health check & live engine metrics
  POST /tick        – advance all four domains by one dt; returns full state
  POST /reset       – rebuild engine from scratch, with optional param overrides
  GET  /docs        – custom cyberpunk Swagger UI
  GET  /openapi.json – OpenAPI schema (served automatically by FastAPI)

FIX applied here
─────────────────
  NexusOmniEngine.__init__ requires a SimulationParams argument.
  Previous crash: NexusOmniEngine()               ← missing required positional arg
  Corrected:      NexusOmniEngine(SimulationParams())   ← see §2 below
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Relative import — works because simulation/ is a package (has __init__.py)
# and uvicorn is launched as `uvicorn simulation.app:app` from the project root.
from core_engine import NexusOmniEngine, SimulationParams


# ══════════════════════════════════════════════════════════════════════════════
# §1  APP INSTANCE
#     docs_url=None  →  we serve our own cyberpunk /docs (§6)
#     redoc_url=None →  disable the default ReDoc page
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Nexus-Omni Simulator",
    description=(
        "Multi-domain mathematical modelling dashboard.\n\n"
        "Four coupled simulation engines running in lock-step:\n"
        "- **Domain 1** — SEIR compartmental population dynamics\n"
        "- **Domain 2** — Spatial diffusion + advection vector field\n"
        "- **Domain 3** — Supply-chain inventory optimisation\n"
        "- **Domain 4** — Composite statistical risk scoring"
    ),
    version="0.1.0",
    docs_url=None,   # disabled — replaced by custom /docs route in §6
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down to your Replit domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# §2  ENGINE SINGLETON
#
#   THE FIX:  NexusOmniEngine expects a SimulationParams as its first argument.
#             We create a default SimulationParams() instance first, then pass
#             it in.  The engine is re-created (not mutated) on every /reset.
# ══════════════════════════════════════════════════════════════════════════════

_params:     SimulationParams = SimulationParams()         # ← default knobs
_engine:     NexusOmniEngine  = NexusOmniEngine(_params)  # ← THE FIX
_tick_count: int               = 0
_last_state                    = None   # caches the most recent SimulationState


# ══════════════════════════════════════════════════════════════════════════════
# §3  PYDANTIC MODELS
#     All request bodies and response shapes are typed here.
#     Every field is Optional in ParamsIn — omitted fields keep their defaults.
# ══════════════════════════════════════════════════════════════════════════════

class ParamsIn(BaseModel):
    """Body for POST /reset.  Only supply the fields you want to override."""
    # Domain 1 — SEIR
    beta:               Optional[float] = None
    gamma:              Optional[float] = None
    sigma:              Optional[float] = None
    population:         Optional[int]   = None
    # Domain 2 — Spatial
    grid_size:          Optional[int]   = None
    diffusion_coeff:    Optional[float] = None
    wind_vx:            Optional[float] = None
    wind_vy:            Optional[float] = None
    # Domain 3 — Supply chain
    num_nodes:          Optional[int]   = None
    base_capacity:      Optional[float] = None
    inflow_rates:       Optional[List[float]] = None
    outflow_rates:      Optional[List[float]] = None
    capacity_threshold: Optional[float] = None
    # Domain 4 — Risk
    risk_weights:       Optional[Tuple[float, float, float]] = None
    # Simulation global
    dt:                 Optional[float] = None
    seed:               int = 42   # always has a value; controls engine RNG


class SEIROut(BaseModel):
    S: float
    E: float
    I: float
    R: float


class VelocityOut(BaseModel):
    vx: List[List[float]]
    vy: List[List[float]]


class TickResponse(BaseModel):
    tick:                int
    seir:                SEIROut
    r_naught:            float
    dominant_eigenvalue: float
    diffusion_grid:      List[List[float]]
    velocity_field:      VelocityOut
    inventory:           List[float]
    bottleneck_flags:    List[bool]
    supply_stress_score: float
    risk_score:          float
    risk_breakdown:      List[float]   # [infection_contribution, spatial, supply]


class HealthResponse(BaseModel):
    status:        str
    tick:          int
    r_naught:      float
    risk_score:    float
    active_params: dict


class ResetResponse(BaseModel):
    status:        str
    tick_reset_to: int
    r_naught:      float
    message:       str


# ══════════════════════════════════════════════════════════════════════════════
# §4  SERIALISATION HELPER
#     SimulationState stores NumPy arrays → convert every field to plain Python
#     types so FastAPI/Pydantic can serialise them to JSON without errors.
# ══════════════════════════════════════════════════════════════════════════════

def _build_response(state, tick: int) -> TickResponse:
    S, E, I, R = state.seir_vector
    vx, vy     = state.velocity_field

    return TickResponse(
        tick                = tick,
        seir                = SEIROut(
                                S=float(S), E=float(E),
                                I=float(I), R=float(R),
                              ),
        r_naught            = round(float(state.r_naught), 6),
        dominant_eigenvalue = round(float(state.dominant_eigenvalue), 6),
        diffusion_grid      = state.diffusion_grid.tolist(),
        velocity_field      = VelocityOut(vx=vx.tolist(), vy=vy.tolist()),
        inventory           = [round(float(v), 2)  for v in state.inventory_vector],
        bottleneck_flags    = state.bottleneck_flags.tolist(),
        supply_stress_score = round(float(state.supply_stress_score), 6),
        risk_score          = round(float(state.risk_score), 6),
        risk_breakdown      = [round(float(v), 6) for v in state.risk_breakdown],
    )


# ══════════════════════════════════════════════════════════════════════════════
# §5  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Health check — engine liveness and current simulation step",
)
async def health() -> HealthResponse:
    """
    Returns engine liveness, the current tick counter, and the full set of
    active `SimulationParams`.  Does **not** advance the simulation.
    """
    r0 = _params.beta / _params.gamma
    rs = float(_last_state.risk_score) if _last_state else 0.0

    return HealthResponse(
        status        = "ok",
        tick          = _tick_count,
        r_naught      = round(r0, 4),
        risk_score    = round(rs, 4),
        active_params = dataclasses.asdict(_params),
    )


@app.post(
    "/tick",
    response_model=TickResponse,
    tags=["Simulation"],
    summary="Advance all four simulation domains by one time step (dt)",
)
async def tick() -> TickResponse:
    """
    Calls `NexusOmniEngine.tick()` once.

    The engine simultaneously advances:
    - SEIR state vector via Euler integration
    - Spatial concentration grid via diffusion + advection
    - Supply-chain inventory via flow balance equations
    - Composite risk score via weighted domain signals

    Returns the full serialised `SimulationState` snapshot.
    """
    global _tick_count, _last_state
    _last_state  = _engine.tick()
    _tick_count += 1
    return _build_response(_last_state, _tick_count)


@app.post(
    "/reset",
    response_model=ResetResponse,
    tags=["Simulation"],
    summary="Reset the engine to t = 0, optionally with new parameter overrides",
)
async def reset(body: ParamsIn = None) -> ResetResponse:
    """
    Rebuilds the engine from scratch.

    - Call with an empty body `{}` to reset to factory defaults.
    - Supply any subset of `ParamsIn` fields to override only those knobs.
    - Unspecified fields retain their current values.

    `seed` controls the engine's internal RNG (default 42).
    """
    global _engine, _tick_count, _last_state, _params

    body = body or ParamsIn()

    # Merge: start from current params dict, apply non-None overrides
    base_dict = dataclasses.asdict(_params)
    overrides  = {
        k: v for k, v in body.model_dump().items()
        if v is not None and k != "seed"
    }

    # Pydantic may give risk_weights as a list; SimulationParams wants a tuple
    if "risk_weights" in overrides and isinstance(overrides["risk_weights"], list):
        overrides["risk_weights"] = tuple(overrides["risk_weights"])

    base_dict.update(overrides)
    new_params = SimulationParams(**base_dict)

    # Replace singleton
    _params      = new_params
    _engine      = NexusOmniEngine(new_params, seed=body.seed)
    _tick_count  = 0
    _last_state  = None

    r0 = new_params.beta / new_params.gamma
    return ResetResponse(
        status        = "reset",
        tick_reset_to = 0,
        r_naught      = round(r0, 4),
        message       = (
            f"Engine rebuilt with {len(overrides)} override(s). "
            f"R\u2080 = {r0:.4f} \u2014 epidemic "
            f"{'grows \u2191' if r0 > 1 else 'dies out \u2193'}."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §6  CUSTOM SWAGGER UI  —  dark neon / cyberpunk theme
#
#     FastAPI's default /docs is disabled (docs_url=None in §1).
#     We serve our own HTML that loads Swagger UI 5 from CDN and injects CSS.
# ══════════════════════════════════════════════════════════════════════════════

_CYBERPUNK_CSS = """
  :root {
    --bg:      #03010a;
    --surface: #0d0b1e;
    --surf2:   #120f28;
    --border:  #1e1a40;
    --cyan:    #00f5ff;
    --violet:  #bf5fff;
    --green:   #39ff14;
    --orange:  #ff9f43;
    --red:     #ff3860;
    --text:    #d8d8f8;
    --dim:     #6868a0;
    --glow-c:  0 0 8px rgba(0,245,255,.7);
    --glow-v:  0 0 8px rgba(191,95,255,.7);
  }
  *, *::before, *::after { box-sizing: border-box; }
  body {
    background: var(--bg);
    margin: 0;
    font-family: 'JetBrains Mono','Fira Code','Courier New',monospace;
  }

  /* TOPBAR */
  .swagger-ui .topbar {
    background: linear-gradient(135deg,#0a0820,#160e30);
    border-bottom: 1px solid var(--cyan);
    box-shadow: 0 2px 24px rgba(0,245,255,.25);
    padding: 10px 0;
  }
  .swagger-ui .topbar a span {
    font-family: 'Orbitron',sans-serif;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(90deg,var(--cyan),var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .swagger-ui .topbar-wrapper img { display:none; }
  .swagger-ui .topbar .download-url-wrapper input[type=text] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text); font-family: inherit;
  }
  .swagger-ui .topbar .download-url-wrapper input[type=text]:focus {
    border-color: var(--cyan); box-shadow: var(--glow-c); outline: none;
  }
  .swagger-ui .topbar .download-url-wrapper .download-url-button {
    background: var(--cyan); color: var(--bg); font-weight: 700;
    border: none; border-radius: 4px; font-family: inherit;
  }

  /* BASE */
  .swagger-ui { background: var(--bg); color: var(--text); }

  /* INFO */
  .swagger-ui .info {
    margin: 28px 0; padding: 24px 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    border-radius: 8px;
  }
  .swagger-ui .info .title {
    font-family: 'Orbitron',sans-serif; font-size: 2rem; font-weight: 900;
    color: var(--cyan); text-shadow: var(--glow-c); letter-spacing: 4px;
  }
  .swagger-ui .info .title small.version-stamp {
    background: var(--violet); color: var(--bg); border-radius: 4px;
    font-size: .6rem; padding: 3px 8px; margin-left: 12px;
    font-family: inherit; vertical-align: middle;
  }
  .swagger-ui .info p,
  .swagger-ui .info li { color: var(--dim); font-size: .85rem; }
  .swagger-ui .info a { color: var(--cyan); text-decoration: none; }
  .swagger-ui .info a:hover { text-shadow: var(--glow-c); }

  /* TAG HEADERS */
  .swagger-ui .opblock-tag {
    border-bottom: 1px solid var(--border);
    color: var(--violet);
    font-family: 'Orbitron',sans-serif; font-size: .9rem; letter-spacing: 2px;
    padding: 14px 0; text-shadow: var(--glow-v);
  }
  .swagger-ui .opblock-tag:hover { background: rgba(191,95,255,.04); }
  .swagger-ui .opblock-tag svg   { fill: var(--violet); }

  /* OPERATION BLOCKS */
  .swagger-ui .opblock {
    border-radius: 6px; margin-bottom: 8px;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    transition: border-color .2s, box-shadow .2s;
  }
  .swagger-ui .opblock:hover   { border-color: var(--cyan) !important; }
  .swagger-ui .opblock.is-open {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 18px rgba(0,245,255,.12);
  }
  .swagger-ui .opblock-summary-method {
    border-radius: 4px;
    font-family: 'Orbitron',sans-serif; font-size: .65rem; font-weight: 900;
    letter-spacing: 1px; min-width: 72px; text-align: center;
  }
  .swagger-ui .opblock-summary-path {
    color: var(--text); font-family: 'JetBrains Mono',monospace; font-size: .9rem;
  }
  .swagger-ui .opblock-summary-description { color: var(--dim); font-size: .8rem; }

  /* GET  → cyan  */
  .swagger-ui .opblock.opblock-get {
    border-left: 3px solid var(--cyan) !important;
    background: rgba(0,245,255,.025) !important;
  }
  .swagger-ui .opblock.opblock-get .opblock-summary-method {
    background: var(--cyan); color: var(--bg);
  }
  /* POST → violet */
  .swagger-ui .opblock.opblock-post {
    border-left: 3px solid var(--violet) !important;
    background: rgba(191,95,255,.025) !important;
  }
  .swagger-ui .opblock.opblock-post .opblock-summary-method {
    background: var(--violet); color: var(--bg);
  }
  /* DELETE → red */
  .swagger-ui .opblock.opblock-delete {
    border-left: 3px solid var(--red) !important;
  }
  .swagger-ui .opblock.opblock-delete .opblock-summary-method {
    background: var(--red); color: #fff;
  }
  /* PUT → orange */
  .swagger-ui .opblock.opblock-put {
    border-left: 3px solid var(--orange) !important;
  }
  .swagger-ui .opblock.opblock-put .opblock-summary-method {
    background: var(--orange); color: var(--bg);
  }

  /* EXPANDED BODY */
  .swagger-ui .opblock-body { background: var(--surf2); }
  .swagger-ui .opblock-description-wrapper p { color: var(--dim); }

  /* PARAMETERS */
  .swagger-ui .parameter__name,
  .swagger-ui .parameter__type,
  .swagger-ui .parameters-col_description p { color: var(--text); font-family: inherit; }
  .swagger-ui .parameter__in { color: var(--dim); font-size: .75rem; }
  .swagger-ui table thead tr th {
    border-bottom: 1px solid var(--border); color: var(--cyan);
    font-family: inherit; font-size: .75rem; letter-spacing: 1px; text-transform: uppercase;
  }
  .swagger-ui table tbody tr td { border-bottom: 1px solid var(--border); }

  /* BUTTONS */
  .swagger-ui .btn.execute {
    background: transparent; border: 1px solid var(--cyan); border-radius: 4px;
    color: var(--cyan); font-family: 'Orbitron',sans-serif; font-size: .65rem;
    letter-spacing: 2px; transition: all .2s;
  }
  .swagger-ui .btn.execute:hover {
    background: var(--cyan); color: var(--bg); box-shadow: var(--glow-c);
  }
  .swagger-ui .btn.cancel {
    background: transparent; border: 1px solid var(--red);
    border-radius: 4px; color: var(--red); font-family: inherit;
  }
  .swagger-ui .try-out__btn {
    background: transparent; border: 1px solid var(--violet);
    border-radius: 4px; color: var(--violet); font-family: inherit; transition: all .2s;
  }
  .swagger-ui .try-out__btn:hover {
    background: var(--violet); color: var(--bg); box-shadow: var(--glow-v);
  }

  /* INPUTS */
  .swagger-ui input[type=text],
  .swagger-ui input[type=number],
  .swagger-ui textarea,
  .swagger-ui select {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 4px; color: var(--text) !important;
    font-family: 'JetBrains Mono',monospace !important; font-size: .85rem;
  }
  .swagger-ui input:focus,
  .swagger-ui textarea:focus,
  .swagger-ui select:focus {
    border-color: var(--cyan) !important; box-shadow: var(--glow-c) !important; outline: none;
  }

  /* RESPONSES */
  .swagger-ui .response-col_status { color: var(--green); font-family: inherit; font-weight: 700; }
  .swagger-ui .response-col_description { color: var(--dim); }
  .swagger-ui .responses-inner h4,
  .swagger-ui .responses-inner h5 { color: var(--text); }
  .swagger-ui .highlight-code > .microlight {
    background: var(--surf2) !important; border: 1px solid var(--border);
    border-radius: 4px; color: var(--green) !important;
    font-family: 'JetBrains Mono',monospace; font-size: .8rem;
  }

  /* SCHEMAS */
  .swagger-ui section.models {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  }
  .swagger-ui section.models h4 {
    color: var(--violet); font-family: 'Orbitron',sans-serif; letter-spacing: 1px;
  }
  .swagger-ui .model-box { background: var(--surf2); }
  .swagger-ui .model-title { color: var(--cyan); }
  .swagger-ui .model { color: var(--text); font-family: 'JetBrains Mono',monospace; }
  .swagger-ui span.prop-name { color: var(--cyan); }
  .swagger-ui .prop-type     { color: var(--violet) !important; }
  .swagger-ui .prop-format   { color: var(--dim); }

  /* SCHEME SELECTOR */
  .swagger-ui .scheme-container {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; box-shadow: none; padding: 16px;
  }
  .swagger-ui .servers > label { color: var(--dim); font-family: inherit; }

  /* MISC */
  .swagger-ui svg   { fill: var(--dim); }
  .swagger-ui .arrow { fill: var(--cyan); }

  /* SCROLLBAR */
  ::-webkit-scrollbar              { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track        { background: var(--bg); }
  ::-webkit-scrollbar-thumb        { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover  { background: var(--cyan); box-shadow: var(--glow-c); }
"""

_SWAGGER_HTML = (
    "<!DOCTYPE html>\n"
    "<html lang=\"en\">\n"
    "<head>\n"
    "  <meta charset=\"UTF-8\" />\n"
    "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
    "  <title>Nexus-Omni \u2014 API Docs</title>\n"
    "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\n"
    "  <link href=\"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Orbitron:wght@700;900&display=swap\"\n"
    "        rel=\"stylesheet\" />\n"
    "  <link rel=\"stylesheet\" type=\"text/css\"\n"
    "        href=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui.css\" />\n"
    "  <style>\n"
    + _CYBERPUNK_CSS +
    "  </style>\n"
    "</head>\n"
    "<body>\n"
    "  <div id=\"swagger-ui\"></div>\n"
    "  <script src=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js\"></script>\n"
    "  <script src=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js\"></script>\n"
    "  <script>\n"
    "    window.onload = function () {\n"
    "      window.ui = SwaggerUIBundle({\n"
    "        url:             \"/openapi.json\",\n"
    "        dom_id:          \"#swagger-ui\",\n"
    "        presets:         [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],\n"
    "        layout:          \"StandaloneLayout\",\n"
    "        deepLinking:     true,\n"
    "        displayRequestDuration: true,\n"
    "        defaultModelsExpandDepth: 1,\n"
    "        defaultModelExpandDepth:  1,\n"
    "        filter:          true,\n"
    "        syntaxHighlight: { theme: \"monokai\" },\n"
    "      });\n"
    "    };\n"
    "  </script>\n"
    "</body>\n"
    "</html>\n"
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    """Serves the custom cyberpunk-themed Swagger UI."""
    return HTMLResponse(_SWAGGER_HTML)
