from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from core_engine import NexusOmniEngine, SimulationParams, SimulationState

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

engine = NexusOmniEngine()

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Nexus-Omni Simulator",
        "version": "0.1.0",
        "ticks_run": engine.ticks_run,
        "hint": "POST /tick to advance the simulation one step."
    }

@app.post("/configure")
def configure_simulation(params: SimulationParams):
    engine.configure(params)
    return {"status": "configured", "params": engine.params}

@app.post("/tick", response_model=SimulationState)
def tick_once():
    return engine.tick()

@app.post("/tick/{n}", response_model=list[SimulationState])
def tick_n(n: int = Field(..., ge=1, le=1000)):
    states = []
    for _ in range(n):
        states.append(engine.tick())
    return states

@app.get("/params", response_model=SimulationParams)
def get_params():
    return engine.params

@app.post("/reset")
def reset_engine():
    engine.reset()
    return {"status": "reset", "ticks_run": engine.ticks_run}
