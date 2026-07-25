from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
import numpy as np

from core_engine import NexusOmniEngine, SimulationParams, SimulationState

# ---------------------------------------------------------------------------
# APP INSTANCE
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nexus-Omni Simulator",
    version="0.1.0",
    description="Multi-domain mathematical modelling backend with live visual simulation dashboard.",
    docs_url=None, # Custom Swagger UI endpoint configured below
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simulation engine
_params = SimulationParams()
engine = NexusOmniEngine(_params)

# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS (Fixed for clean JSON Schema serialization - no raw np.ndarray)
# ---------------------------------------------------------------------------
class ParamsIn(BaseModel):
    beta: Optional[float] = Field(None, description="SEIR transmission rate")
    gamma: Optional[float] = Field(None, description="SEIR recovery rate")
    sigma: Optional[float] = Field(None, description="SEIR incubation rate")
    population: Optional[int] = Field(None, description="Total population scale")
    grid_size: Optional[int] = Field(None, description="Spatial grid dimension")
    diffusion_coeff: Optional[float] = Field(None, description="Diffusion coefficient")
    wind_vx: Optional[float] = Field(None, description="Wind velocity X vector")
    wind_vy: Optional[float] = Field(None, description="Wind velocity Y vector")
    num_nodes: Optional[int] = Field(None, description="Supply chain inventory nodes")
    base_capacity: Optional[float] = Field(None, description="Base inventory capacity")
    capacity_threshold: Optional[float] = Field(None, description="Risk scoring threshold")

class TickResponse(BaseModel):
    t: int
    dt: float
    seir_summary: dict
    spatial_peak: float
    supply_shortages: int
    composite_risk_score: float

class ResetResponse(BaseModel):
    status: str
    t: int
    message: str

# ---------------------------------------------------------------------------
# CUSTOM CYBERPUNK SWAGGER UI WITH LIVE VISUAL FRONTEND TAB
# ---------------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Nexus-Omni Simulator - Live Visual Dashboard</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <style>
            :root {
                --bg-color: #0b0f19;
                --panel-bg: #111827;
                --neon-cyan: #00f0ff;
                --neon-purple: #b000ff;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --border-color: #1f2937;
            }
            body {
                background-color: var(--bg-color) !important;
                color: var(--text-main) !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                margin: 0;
                padding: 0;
            }
            /* Custom Visual Control Deck Banner for Recruiters */
            .recruiter-banner {
                background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(176, 0, 255, 0.1));
                border-bottom: 2px solid var(--neon-cyan);
                padding: 30px 20px;
                text-align: center;
            }
            .recruiter-banner h1 {
                font-size: 2.2rem;
                margin: 0 0 10px 0;
                color: var(--neon-cyan);
                text-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
                letter-spacing: 1px;
            }
            .recruiter-banner p {
                color: var(--text-muted);
                font-size: 1.1rem;
                max-width: 700px;
                margin: 0 auto 20px auto;
                line-height: 1.5;
            }
            .live-viz-box {
                background: var(--panel-bg);
                border: 1px solid var(--neon-purple);
                border-radius: 12px;
                max-width: 900px;
                margin: 20px auto;
                padding: 20px;
                box-shadow: 0 0 30px rgba(176, 0, 255, 0.15);
                text-align: left;
            }
            .live-viz-box h3 {
                color: var(--neon-purple);
                margin-top: 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .metric-card {
                background: #06080f;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }
            .metric-card .val {
                font-size: 1.4rem;
                font-weight: bold;
                color: var(--neon-cyan);
                margin-top: 5px;
            }
            .action-row {
                margin-top: 15px;
                display: flex;
                gap: 10px;
                justify-content: center;
            }
            .cyber-btn {
                background: transparent;
                border: 1px solid var(--neon-cyan);
                color: var(--neon-cyan);
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.2s ease;
            }
            .cyber-btn:hover {
                background: var(--neon-cyan);
                color: var(--bg-color);
                box-shadow: 0 0 15px var(--neon-cyan);
            }
            /* Swagger Custom Theme overrides */
            .swagger-ui .topbar { display: none !important; }
            .swagger-ui .scheme-container { background: var(--panel-bg) !important; box-shadow: none !important; }
            .swagger-ui .info h1, .swagger-ui .info p, .swagger-ui .info table { color: var(--text-main) !important; }
            .swagger-ui .opblock { background: var(--panel-bg) !important; border-color: var(--border-color) !important; border-radius: 8px; }
            .swagger-ui .opblock.opblock-get { border-color: var(--neon-cyan) !important; }
            .swagger-ui .opblock.opblock-post { border-color: var(--neon-purple) !important; }
            .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-description { color: var(--text-main) !important; }
            .swagger-ui .btn.execute { background-color: var(--neon-cyan) !important; color: #000 !important; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="recruiter-banner">
            <h1>Nexus-Omni Visual Control Deck</h1>
            <p>Interactive multi-domain simulation core running SEIR dynamics, spatial vector fields, and composite risk metrics.</p>
            
            <div class="live-viz-box">
                <h3>
                    <span>Live Telemetry Visualizer</span>
                    <span id="sim-status" style="font-size: 0.85rem; color: #10b981;">● ENGINE ACTIVE</span>
                </h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div style="color:var(--text-muted); font-size:0.85rem;">Simulation Step (t)</div>
                        <div class="val" id="val-t">0</div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--text-muted); font-size:0.85rem;">Composite Risk Score</div>
                        <div class="val" id="val-risk">0.000</div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--text-muted); font-size:0.85rem;">Spatial Peak Density</div>
                        <div class="val" id="val-peak">0.000</div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--text-muted); font-size:0.85rem;">Supply Shortages</div>
                        <div class="val" id="val-shortages">0</div>
                    </div>
                </div>
                <div class="action-row">
                    <button class="cyber-btn" onclick="triggerTick()">▶ Step Simulation (/tick)</button>
                    <button class="cyber-btn" style="border-color:var(--neon-purple); color:var(--neon-purple);" onclick="triggerReset()">⟳ Reset Engine</button>
                </div>
            </div>
        </div>

        <div id="swagger-ui"></div>

        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                window.ui = SwaggerUIBundle({
                    url: "/openapi.json",
                    dom_id: '#swagger-ui',
                    presets: [SwaggerUIBundle.presets.apis],
                    layout: "BaseLayout",
                    deepLinking: true
                });
            };

            async function triggerTick() {
                try {
                    const res = await fetch('/tick', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('val-t').innerText = data.t;
                    document.getElementById('val-risk').innerText = data.composite_risk_score.toFixed(3);
                    document.getElementById('val-peak').innerText = data.spatial_peak.toFixed(3);
                    document.getElementById('val-shortages').innerText = data.supply_shortages;
                } catch (err) {
                    console.error("Tick error:", err);
                }
            }

            async function triggerReset() {
                try {
                    const res = await fetch('/reset', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('val-t').innerText = data.t;
                    document.getElementById('val-risk').innerText = "0.000";
                    document.getElementById('val-peak').innerText = "0.000";
                    document.getElementById('val-shortages').innerText = "0";
                } catch (err) {
                    console.error("Reset error:", err);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/", tags=["Meta"])
def health_check():
    """Health check endpoint confirming engine liveness and current step."""
    return {
        "status": "ok",
        "service": "Nexus-Omni Simulator",
        "version": "0.1.0",
        "ticks_run": engine.ticks_run,
        "hint": "Navigate to /docs for the interactive visual control deck."
    }

@app.post("/tick", response_model=TickResponse, tags=["Simulation"])
def tick_once():
    """Advance all four simulation domains by one time step (dt)."""
    try:
        state = engine.step()
        return {
            "t": state.t,
            "dt": state.dt,
            "seir_summary": state.seir_summary,
            "spatial_peak": float(np.max(state.spatial_grid)) if state.spatial_grid is not None else 0.0,
            "supply_shortages": int(np.sum(state.supply_shortages)) if state.supply_shortages is not None else 0,
            "composite_risk_score": float(state.composite_risk)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", response_model=ResetResponse, tags=["Simulation"])
def reset_simulation(params_override: Optional[ParamsIn] = None):
    """Reset the simulation engine back to t = 0 with optional parameter overrides."""
    try:
        if params_override:
            current_dict = dataclasses.asdict(engine.params)
            for k, v in params_override.model_dump(exclude_unset=True).items():
                if v is not None:
                    current_dict[k] = v
            new_params = SimulationParams(**current_dict)
            engine.configure(new_params)
        else:
            engine.reset()
        
        return {
            "status": "success",
            "t": engine.ticks_run,
            "message": "Simulation engine successfully re-initialized."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
