"""
simulation/app.py
──────────────────────────────────────────────────────────────────────────────
FastAPI application for the Nexus-Omni Simulator.

Run from the PROJECT ROOT (the folder that contains the simulation/ package):
    uvicorn simulation.app:app --host 0.0.0.0 --port 8080 --reload

Endpoints
─────────
  GET  /               → health-check / welcome JSON
  POST /configure      → replace simulation parameters, reset engine state
  POST /tick           → advance one time step, return full SimulationState
  POST /tick/{n}       → advance n steps at once, return list of states
  GET  /params         → return the current parameter set
  POST /reset          → reset engine to initial conditions (keep params)
──────────────────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

app = FastAPI(
    title="Nexus-Omni Simulator API",
    version="0.1.0",
    description="Multi-domain mathematical modelling backend.",
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Neon UI",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
    )

# ══════════════════════════════════════════════════════════════════════════════
# §1  REQUEST / RESPONSE SCHEMAS  (Pydantic)
#     These are the JSON shapes the frontend will send and receive.
# ══════════════════════════════════════════════════════════════════════════════

class ParamsRequest(BaseModel):
    """
    All fields are optional — only send what you want to change.
    Omitted fields keep their current (or default) values.
    """
    # Domain 1
    beta:       Optional[float] = Field(None, gt=0, le=5,   description="Transmission rate β")
    gamma:      Optional[float] = Field(None, gt=0, le=1,   description="Recovery rate γ")
    sigma:      Optional[float] = Field(None, gt=0, le=1,   description="Incubation rate σ")
    population: Optional[int]   = Field(None, gt=0,          description="Total population N")

    # Domain 2
    grid_size:       Optional[int]   = Field(None, ge=4, le=64)
    diffusion_coeff: Optional[float] = Field(None, ge=0.0, le=1.0)
    wind_vx:         Optional[float] = Field(None, ge=-1.0, le=1.0)
    wind_vy:         Optional[float] = Field(None, ge=-1.0, le=1.0)

    # Domain 3
    num_nodes:          Optional[int]   = Field(None, ge=1, le=20)
    base_capacity:      Optional[float] = Field(None, gt=0)
    capacity_threshold: Optional[float] = Field(None, ge=0.5, le=1.0)

    # Simulation global
    dt: Optional[float] = Field(None, gt=0, le=5.0, description="Time step in days")


class SEIROut(BaseModel):
    S: float
    E: float
    I: float
    R: float


class VelocityFieldOut(BaseModel):
    """Flattened velocity arrays (easier to JSON-serialise than 2-D arrays)."""
    Vx: list[float]
    Vy: list[float]
    shape: list[int]   # [rows, cols]


class TickResponse(BaseModel):
    """
    Complete output of one simulation tick.
    All numpy arrays are serialised to plain Python lists.
    """
    # Metadata
    tick_number: int

    # Domain 1
    seir:               SEIROut
    r_naught:           float
    dominant_eigenvalue: float

    # Domain 2
    diffusion_grid: list[list[float]]   # N×N nested list
    velocity_field: VelocityFieldOut

    # Domain 3
    inventory_vector:    list[float]
    bottleneck_flags:    list[bool]
    supply_stress_score: float

    # Domain 4
    risk_score:     float
    risk_breakdown: list[float]   # [infection_contribution, spatial, supply]


class MultiTickResponse(BaseModel):
    ticks: list[TickResponse]


# ══════════════════════════════════════════════════════════════════════════════
# §2  SERVER STATE
#     One engine instance lives for the life of the server process.
#     /configure replaces it; /reset re-initialises it with the same params.
# ══════════════════════════════════════════════════════════════════════════════

_params: SimulationParams = SimulationParams()
_engine: NexusOmniEngine  = NexusOmniEngine(_params)
_tick_count: int          = 0


def _state_to_response(state: SimulationState, tick_number: int) -> TickResponse:
    """Convert a SimulationState (numpy-heavy) to a JSON-safe TickResponse."""
    S, E, I, R = state.seir_vector.tolist()
    Vx, Vy     = state.velocity_field
    rows, cols  = Vx.shape

    return TickResponse(
        tick_number=tick_number,
        seir=SEIROut(S=S, E=E, I=I, R=R),
        r_naught=round(state.r_naught, 6),
        dominant_eigenvalue=round(state.dominant_eigenvalue, 6),
        diffusion_grid=state.diffusion_grid.tolist(),
        velocity_field=VelocityFieldOut(
            Vx=Vx.flatten().tolist(),
            Vy=Vy.flatten().tolist(),
            shape=[rows, cols],
        ),
        inventory_vector=[round(v, 4) for v in state.inventory_vector.tolist()],
        bottleneck_flags=state.bottleneck_flags.tolist(),
        supply_stress_score=round(state.supply_stress_score, 6),
        risk_score=round(state.risk_score, 6),
        risk_breakdown=[round(v, 6) for v in state.risk_breakdown.tolist()],
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["meta"])
def health_check():
    """Server health check and welcome message."""
    return {
        "status":  "ok",
        "service": "Nexus-Omni Simulator",
        "version": "0.1.0",
        "ticks_run": _tick_count,
        "hint": "POST /tick to advance the simulation one step.",
    }


@app.get("/params", tags=["configuration"])
def get_params():
    """Return the current simulation parameter set."""
    return _params.__dict__


@app.post("/configure", tags=["configuration"])
def configure(req: ParamsRequest):
    """
    Update one or more simulation parameters and reset the engine.
    Only the fields you include in the JSON body are changed.
    """
    global _params, _engine, _tick_count

    # Merge: start from current params, apply only the non-None fields
    current = _params.__dict__.copy()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    current.update(updates)

    # Rebuild param object and reset engine
    _params      = SimulationParams(**current)
    _engine      = NexusOmniEngine(_params)
    _tick_count  = 0

    log.info(f"[/configure] params updated: {updates}")
    return {"status": "reconfigured", "applied": updates, "params": _params.__dict__}


@app.post("/reset", tags=["configuration"])
def reset():
    """Re-initialise the engine to t=0 without changing parameters."""
    global _engine, _tick_count
    _engine     = NexusOmniEngine(_params)
    _tick_count = 0
    log.info("[/reset] engine reset to t=0")
    return {"status": "reset", "tick_count": _tick_count}


@app.post("/tick", response_model=TickResponse, tags=["simulation"])
def tick_once():
    """Advance the simulation by one time step (dt days) and return the state."""
    global _tick_count
    state        = _engine.tick()
    _tick_count += 1
    log.info(f"[/tick] t={_tick_count}  risk={state.risk_score:.4f}")
    return _state_to_response(state, _tick_count)


@app.post("/tick/{n}", response_model=MultiTickResponse, tags=["simulation"])
def tick_n(n: int):
    """
    Advance the simulation by n steps and return every intermediate state.
    Useful for fast-forwarding or batch data collection.
    Max n = 500 to prevent timeouts.
    """
    global _tick_count
    if not (1 <= n <= 500):
        raise HTTPException(status_code=422, detail="n must be between 1 and 500.")

    responses = []
    for _ in range(n):
        state        = _engine.tick()
        _tick_count += 1
        responses.append(_state_to_response(state, _tick_count))

    log.info(f"[/tick/{n}] ran {n} steps, now at t={_tick_count}")
    return MultiTickResponse(ticks=responses)
