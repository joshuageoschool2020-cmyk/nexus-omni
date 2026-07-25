from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np

from core_engine import NexusOmniEngine, SimulationParams, SimulationState

# ---------------------------------------------------------------------------
# APP INSTANCE
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nexus-Omni Simulator",
    version="0.1.0",
    description="Multi-domain mathematical modelling backend with NASA-grade telemetry command deck.",
    docs_url=None, 
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
# PYDANTIC SCHEMAS (Validated for clean JSON Schema serialization)
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
# NASA-GRADE ADVANCED TELEMETRY CONTROL DECK UI
# ---------------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NEXUS-OMNI // DEEP-SPACE TELEMETRY & COMMAND DECK</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <style>
            :root {
                --nasa-bg: #050b14;
                --nasa-panel: #0d1726;
                --nasa-border: #1e3a5f;
                --nasa-amber: #ffb703;
                --nasa-cyan: #00b4d8;
                --nasa-green: #2ec4b6;
                --nasa-red: #e71d36;
                --nasa-text: #e2e8f0;
                --nasa-muted: #64748b;
            }
            body {
                background-color: var(--nasa-bg) !important;
                color: var(--nasa-text) !important;
                font-family: 'Courier New', Courier, monospace, sans-serif;
                margin: 0;
                padding: 0;
            }
            /* NASA Mission Control Header */
            .mission-header {
                background: linear-gradient(180deg, #091324 0%, #050b14 100%);
                border-bottom: 2px solid var(--nasa-cyan);
                padding: 20px;
                font-family: 'Courier New', Courier, monospace;
            }
            .mission-grid-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1400px;
                margin: 0 auto;
                border-bottom: 1px dashed var(--nasa-border);
                padding-bottom: 15px;
                margin-bottom: 15px;
            }
            .mission-title {
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--nasa-cyan);
                letter-spacing: 2px;
            }
            .mission-badge {
                background: rgba(0, 180, 216, 0.1);
                border: 1px solid var(--nasa-cyan);
                color: var(--nasa-cyan);
                padding: 4px 10px;
                font-size: 0.8rem;
                letter-spacing: 1px;
            }
            
            /* Command Console Container */
            .command-deck {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            @media (max-width: 900px) {
                .command-deck { grid-template-columns: 1fr; }
            }
            
            .console-panel {
                background: var(--nasa-panel);
                border: 1px solid var(--nasa-border);
                border-radius: 4px;
                padding: 15px;
                box-shadow: inset 0 0 15px rgba(0, 180, 216, 0.05);
            }
            .panel-header {
                font-size: 0.9rem;
                color: var(--nasa-amber);
                text-transform: uppercase;
                letter-spacing: 1.5px;
                border-bottom: 1px solid var(--nasa-border);
                padding-bottom: 8px;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
            }
            
            /* Telemetry Data Grid */
            .telemetry-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }
            .telemetry-item {
                background: #060e18;
                border: 1px solid var(--nasa-border);
                padding: 10px;
            }
            .telemetry-label {
                font-size: 0.75rem;
                color: var(--nasa-muted);
                text-transform: uppercase;
            }
            .telemetry-value {
                font-size: 1.4rem;
                font-weight: bold;
                color: var(--nasa-green);
                margin-top: 4px;
            }
            
            /* Vector Canvas Simulation Visualiser */
            #sim-canvas {
                width: 100%;
                height: 180px;
                background: #03070e;
                border: 1px solid var(--nasa-border);
                display: block;
            }
            
            /* Console Action Controls */
            .action-panel-btns {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            .nasa-btn {
                background: transparent;
                border: 1px solid var(--nasa-cyan);
                color: var(--nasa-cyan);
                padding: 8px 14px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.85rem;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
                text-transform: uppercase;
            }
            .nasa-btn:hover {
                background: var(--nasa-cyan);
                color: var(--nasa-bg);
                box-shadow: 0 0 10px var(--nasa-cyan);
            }
            .nasa-btn-amber {
                border-color: var(--nasa-amber);
                color: var(--nasa-amber);
            }
            .nasa-btn-amber:hover {
                background: var(--nasa-amber);
                color: var(--nasa-bg);
                box-shadow: 0 0 10px var(--nasa-amber);
            }
            
            /* Live Terminal Logs */
            .terminal-log {
                background: #020509;
                border: 1px solid var(--nasa-border);
                height: 120px;
                overflow-y: auto;
                padding: 8px;
                font-size: 0.75rem;
                color: #38bdf8;
                line-height: 1.4;
            }

            /* Swagger UI Restyling to match NASA Control Theme */
            .swagger-ui .topbar { display: none !important; }
            .swagger-ui .scheme-container { background: var(--nasa-panel) !important; box-shadow: none !important; border: 1px solid var(--nasa-border); }
            .swagger-ui .info h1, .swagger-ui .info p, .swagger-ui .info table, .swagger-ui .base-url { color: var(--nasa-text) !important; font-family: 'Courier New', Courier, monospace !important; }
            .swagger-ui .opblock { background: var(--nasa-panel) !important; border-color: var(--nasa-border) !important; border-radius: 0px !important; }
            .swagger-ui .opblock.opblock-get { border-left: 4px solid var(--nasa-cyan) !important; }
            .swagger-ui .opblock.opblock-post { border-left: 4px solid var(--nasa-amber) !important; }
            .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-description { color: var(--nasa-text) !important; font-family: 'Courier New', Courier, monospace !important; }
            .swagger-ui .btn.execute { background-color: var(--nasa-cyan) !important; color: #000 !important; font-weight: bold; font-family: 'Courier New', Courier, monospace !important; }
            .swagger-ui .tab li { color: var(--nasa-text) !important; }
        </style>
    </head>
    <body>
        <div class="mission-header">
            <div class="mission-grid-top">
                <div class="mission-title">SYSTEMS // NEXUS-OMNI FLIGHT DYNAMICS & SIMULATION ENGINE</div>
                <div class="mission-badge" id="telemetry-status">● TELEMETRY ONLINE [STABLE]</div>
            </div>

            <div class="command-deck">
                <!-- Left Column: Primary Telemetry & Visualizer -->
                <div class="console-panel">
                    <div class="panel-header">
                        <span>Real-Time Vector Field & State Telemetry</span>
                        <span>[DOMAINS: 1-4 LOCKED]</span>
                    </div>
                    
                    <div class="telemetry-grid" style="margin-bottom: 15px;">
                        <div class="telemetry-item">
                            <div class="telemetry-label">Simulation Epoch (t)</div>
                            <div class="telemetry-value" id="disp-t">0</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Composite Risk Index</div>
                            <div class="telemetry-value" id="disp-risk" style="color: var(--nasa-amber);">0.0000</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Spatial Diffusion Peak</div>
                            <div class="telemetry-value" id="disp-peak">0.0000</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Supply Shortages</div>
                            <div class="telemetry-value" id="disp-shortages" style="color: var(--nasa-red);">0</div>
                        </div>
                    </div>

                    <!-- Visual Vector Grid Rendering Canvas -->
                    <canvas id="sim-canvas" width="600" height="180"></canvas>

                    <div class="action-panel-btns">
                        <button class="nasa-btn" onclick="executeTick()">▶ ADVANCE EPOCH (/tick)</button>
                        <button class="nasa-btn nasa-btn-amber" onclick="executeReset()">⟳ RE-INITIALIZE ENGINE</button>
                    </div>
                </div>

                <!-- Right Column: System Logs & Execution Output -->
                <div class="console-panel">
                    <div class="panel-header">
                        <span>Mission Control Event Stream</span>
                    </div>
                    <div class="terminal-log" id="terminal-stream">
                        [SYS_INIT] Core engine linked successfully.<br>
                        [SYS_INIT] Spatial matrix buffers allocated.<br>
                        [READY] Awaiting operator command input...<br>
                    </div>
                </div>
            </div>
        </div>

        <div id="swagger-ui" style="max-width: 1400px; margin: 20px auto; padding: 0 20px;"></div>

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
                initCanvas();
            };

            // Canvas Vector Visualizer Simulation Animation Loop
            let canvasPhase = 0;
            function initCanvas() {
                const canvas = document.getElementById('sim-canvas');
                const ctx = canvas.getContext('2d');
                
                function draw() {
                    ctx.fillStyle = '#03070e';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw grid matrix
                    ctx.strokeStyle = '#1e3a5f';
                    ctx.lineWidth = 0.5;
                    for(let x = 0; x < canvas.width; x += 30) {
                        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                    }
                    for(let y = 0; y < canvas.height; y += 30) {
                        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                    }

                    // Draw dynamic simulated vector field wave
                    ctx.strokeStyle = '#00b4d8';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    for(let x = 0; x < canvas.width; x++) {
                        let y = canvas.height / 2 + Math.sin((x + canvasPhase) * 0.03) * 35 * Math.cos((x * 0.01));
                        if(x === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();

                    // Secondary amber wave overlay
                    ctx.strokeStyle = '#ffb703';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    for(let x = 0; x < canvas.width; x++) {
                        let y = canvas.height / 2 + Math.cos((x - canvasPhase) * 0.02) * 25;
                        if(x === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();

                    canvasPhase += 1.5;
                    requestAnimationFrame(draw);
                }
                draw();
            }

            function logMessage(msg) {
                const term = document.getElementById('terminal-stream');
                term.innerHTML += `[${new Date().toLocaleTimeString()}] ${msg}<br>`;
                term.scrollTop = term.scrollHeight;
            }

            async function executeTick() {
                try {
                    const res = await fetch('/tick', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('disp-t').innerText = data.t;
                    document.getElementById('disp-risk').innerText = data.composite_risk_score.toFixed(4);
                    document.getElementById('disp-peak').innerText = data.spatial_peak.toFixed(4);
                    document.getElementById('disp-shortages').innerText = data.supply_shortages;
                    logMessage(`Epoch advanced to t=${data.t} | Risk: ${data.composite_risk_score.toFixed(3)}`);
                } catch (err) {
                    logMessage(`ERROR: Tick execution failed.`);
                }
            }

            async function executeReset() {
                try {
                    const res = await fetch('/reset', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('disp-t').innerText = data.t;
                    document.getElementById('disp-risk').innerText = "0.0000";
                    document.getElementById('disp-peak').innerText = "0.0000";
                    document.getElementById('disp-shortages').innerText = "0";
                    logMessage(`SYSTEM RE-INITIALIZED: Epoch reset to t=0.`);
                } catch (err) {
                    logMessage(`ERROR: Reset sequence failed.`);
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
        "hint": "Navigate to /docs for the NASA-grade command telemetry deck."
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
            "message": "Flight simulation engine successfully re-calibrated."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
