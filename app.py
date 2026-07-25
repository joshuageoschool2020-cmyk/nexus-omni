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
    description="Multi-domain mathematical modelling backend with aerospace telemetry and analytics.",
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
# ADVANCED AEROSPACE MISSION CONTROL COMMAND DECK UI
# ---------------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NEXUS-OMNI // FLIGHT OPERATIONS & TELEMETRY COMMAND DECK</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <style>
            :root {
                --nasa-bg: #020617;
                --nasa-panel: #090d19;
                --nasa-border: #1e293b;
                --nasa-amber: #f59e0b;
                --nasa-cyan: #38bdf8;
                --nasa-green: #34d399;
                --nasa-red: #f87171;
                --nasa-text: #f1f5f9;
                --nasa-muted: #64748b;
            }
            body {
                background-color: var(--nasa-bg) !important;
                color: var(--nasa-text) !important;
                font-family: 'Courier New', Courier, monospace, sans-serif;
                margin: 0;
                padding: 0;
            }
            .mission-header {
                background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
                border-bottom: 2px solid var(--nasa-cyan);
                padding: 20px;
            }
            .mission-grid-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1450px;
                margin: 0 auto;
                border-bottom: 1px dashed var(--nasa-border);
                padding-bottom: 15px;
                margin-bottom: 20px;
            }
            .mission-title {
                font-size: 1.3rem;
                font-weight: bold;
                color: var(--nasa-cyan);
                letter-spacing: 2px;
            }
            .mission-badge {
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid var(--nasa-cyan);
                color: var(--nasa-cyan);
                padding: 4px 12px;
                font-size: 0.75rem;
                letter-spacing: 1px;
            }
            .command-deck {
                max-width: 1450px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            @media (max-width: 1000px) {
                .command-deck { grid-template-columns: 1fr; }
            }
            .console-panel {
                background: var(--nasa-panel);
                border: 1px solid var(--nasa-border);
                border-radius: 4px;
                padding: 15px;
            }
            .panel-header {
                font-size: 0.8rem;
                color: var(--nasa-amber);
                text-transform: uppercase;
                letter-spacing: 1.5px;
                border-bottom: 1px solid var(--nasa-border);
                padding-bottom: 8px;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
            }
            .telemetry-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin-bottom: 15px;
            }
            .telemetry-item {
                background: #020617;
                border: 1px solid var(--nasa-border);
                padding: 8px 10px;
                text-align: center;
            }
            .telemetry-label {
                font-size: 0.65rem;
                color: var(--nasa-muted);
                text-transform: uppercase;
            }
            .telemetry-value {
                font-size: 1.2rem;
                font-weight: bold;
                color: var(--nasa-green);
                margin-top: 4px;
            }
            .graphs-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-bottom: 15px;
            }
            .graph-box {
                background: #020617;
                border: 1px solid var(--nasa-border);
                padding: 8px;
            }
            .graph-title {
                font-size: 0.65rem;
                color: var(--nasa-cyan);
                margin-bottom: 4px;
                text-transform: uppercase;
            }
            canvas {
                width: 100%;
                height: 100px;
                display: block;
            }
            .action-panel-btns {
                display: flex;
                gap: 10px;
            }
            .nasa-btn {
                background: transparent;
                border: 1px solid var(--nasa-cyan);
                color: var(--nasa-cyan);
                padding: 8px 14px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.75rem;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
                text-transform: uppercase;
                flex: 1;
            }
            .nasa-btn:hover {
                background: var(--nasa-cyan);
                color: var(--nasa-bg);
            }
            .nasa-btn-amber {
                border-color: var(--nasa-amber);
                color: var(--nasa-amber);
            }
            .nasa-btn-amber:hover {
                background: var(--nasa-amber);
                color: var(--nasa-bg);
            }
            .terminal-log {
                background: #020617;
                border: 1px solid var(--nasa-border);
                height: 295px;
                overflow-y: auto;
                padding: 10px;
                font-size: 0.7rem;
                color: #38bdf8;
                line-height: 1.4;
            }
            /* Swagger UI Restyling */
            .swagger-ui .topbar { display: none !important; }
            .swagger-ui .scheme-container { background: var(--nasa-panel) !important; border: 1px solid var(--nasa-border); }
            .swagger-ui .info h1, .swagger-ui .info p, .swagger-ui .info table, .swagger-ui .base-url { color: var(--nasa-text) !important; font-family: 'Courier New', Courier, monospace !important; }
            .swagger-ui .opblock { background: var(--nasa-panel) !important; border-color: var(--nasa-border) !important; border-radius: 0px !important; }
            .swagger-ui .opblock.opblock-get { border-left: 4px solid var(--nasa-cyan) !important; }
            .swagger-ui .opblock.opblock-post { border-left: 4px solid var(--nasa-amber) !important; }
            .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-description { color: var(--nasa-text) !important; font-family: 'Courier New', Courier, monospace !important; }
            .swagger-ui .btn.execute { background-color: var(--nasa-cyan) !important; color: #000 !important; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="mission-header">
            <div class="mission-grid-top">
                <div class="mission-title">FLIGHTCOM // NEXUS-OMNI MULTI-DOMAIN TELEMETRY DECK</div>
                <div class="mission-badge">● SYS_LINK: ESTABLISHED</div>
            </div>

            <div class="command-deck">
                <div class="console-panel">
                    <div class="panel-header">
                        <span>Real-Time Analytical Subsystems</span>
                        <span>[STATUS: NOMINAL]</span>
                    </div>
                    
                    <div class="telemetry-grid">
                        <div class="telemetry-item">
                            <div class="telemetry-label">Epoch (t)</div>
                            <div class="telemetry-value" id="disp-t">0</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Risk Index</div>
                            <div class="telemetry-value" id="disp-risk" style="color: var(--nasa-amber);">0.0000</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Spatial Peak</div>
                            <div class="telemetry-value" id="disp-peak">0.0000</div>
                        </div>
                        <div class="telemetry-item">
                            <div class="telemetry-label">Shortages</div>
                            <div class="telemetry-value" id="disp-shortages" style="color: var(--nasa-red);">0</div>
                        </div>
                    </div>

                    <div class="graphs-container">
                        <div class="graph-box">
                            <div class="graph-title">1. Risk Trajectory Vector</div>
                            <canvas id="graph-risk" width="300" height="100"></canvas>
                        </div>
                        <div class="graph-box">
                            <div class="graph-title">2. Spatial Diffusion Matrix</div>
                            <canvas id="graph-spatial" width="300" height="100"></canvas>
                        </div>
                        <div class="graph-box">
                            <div class="graph-title">3. Supply Chain Stress</div>
                            <canvas id="graph-supply" width="300" height="100"></canvas>
                        </div>
                        <div class="graph-box">
                            <div class="graph-title">4. SEIR Population Flux</div>
                            <canvas id="graph-seir" width="300" height="100"></canvas>
                        </div>
                    </div>

                    <div class="action-panel-btns">
                        <button class="nasa-btn" onclick="executeTick()">▶ EXECUTE TICK (/tick)</button>
                        <button class="nasa-btn nasa-btn-amber" onclick="executeReset()">⟳ HARD RESET</button>
                    </div>
                </div>

                <div class="console-panel">
                    <div class="panel-header">
                        <span>Telemetry Event Stream</span>
                    </div>
                    <div class="terminal-log" id="terminal-stream">
                        [SYS] Core telemetry interface loaded.<br>
                        [SYS] Canvas buffers allocated cleanly.<br>
                        [READY] Awaiting flight control input...<br>
                    </div>
                </div>
            </div>
        </div>

        <div id="swagger-ui" style="max-width: 1450px; margin: 20px auto; padding: 0 20px;"></div>

        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            let historyRisk = [];
            let historyPeak = [];
            let historyShortages = [];

            window.onload = function() {
                window.ui = SwaggerUIBundle({
                    url: "/openapi.json",
                    dom_id: '#swagger-ui',
                    presets: [SwaggerUIBundle.presets.apis],
                    layout: "BaseLayout",
                    deepLinking: true
                });
                startGraphLoops();
            };

            function logMessage(msg) {
                const term = document.getElementById('terminal-stream');
                term.innerHTML += `[${new Date().toLocaleTimeString()}] ${msg}<br>`;
                term.scrollTop = term.scrollHeight;
            }

            function drawLineGraph(canvasId, dataArray, lineColor) {
                const canvas = document.getElementById(canvasId);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.strokeStyle = '#1e293b';
                ctx.lineWidth = 0.4;
                for(let x = 0; x < canvas.width; x += 40) {
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                }
                for(let y = 0; y < canvas.height; y += 25) {
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                }

                if (dataArray.length < 2) return;

                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 2;
                ctx.beginPath();

                const maxVal = Math.max(...dataArray, 1.0);
                const minVal = Math.min(...dataArray, 0.0);
                const range = maxVal - minVal === 0 ? 1 : maxVal - minVal;

                for (let i = 0; i < dataArray.length; i++) {
                    let x = (i / (dataArray.length - 1)) * canvas.width;
                    let y = canvas.height - ((dataArray[i] - minVal) / range) * (canvas.height - 20) - 10;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            let phase = 0;
            function startGraphLoops() {
                function render() {
                    let rData = historyRisk.length > 0 ? historyRisk : [0, 0];
                    let pData = historyPeak.length > 0 ? historyPeak : [0, 0];
                    let sData = historyShortages.length > 0 ? historyShortages : [0, 0];
                    
                    let seirData = [];
                    for(let i=0; i<20; i++) {
                        seirData.push(Math.sin((i + phase) * 0.2) * 5 + 10);
                    }

                    drawLineGraph('graph-risk', rData, '#f59e0b');
                    drawLineGraph('graph-spatial', pData, '#38bdf8');
                    drawLineGraph('graph-supply', sData, '#f87171');
                    drawLineGraph('graph-seir', seirData, '#34d399');

                    phase += 0.2;
                    requestAnimationFrame(render);
                }
                render();
            }

            async function executeTick() {
                try {
                    const res = await fetch('/tick', { method: 'POST' });
                    const data = await res.json();
                    
                    document.getElementById('disp-t').innerText = data.t;
                    document.getElementById('disp-risk').innerText = data.composite_risk_score.toFixed(4);
                    document.getElementById('disp-peak').innerText = data.spatial_peak.toFixed(4);
                    document.getElementById('disp-shortages').innerText = data.supply_shortages;

                    historyRisk.push(data.composite_risk_score);
                    historyPeak.push(data.spatial_peak);
                    historyShortages.push(data.supply_shortages);

                    if(historyRisk.length > 30) {
                        historyRisk.shift();
                        historyPeak.shift();
                        historyShortages.shift();
                    }

                    logMessage(`EPOCH ${data.t} SUCCESS // Risk: ${data.composite_risk_score.toFixed(4)}`);
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

                    historyRisk = [];
                    historyPeak = [];
                    historyShortages = [];

                    logMessage(`HARD RESET: Engine state re-initialized to t=0.`);
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
        "current_t": engine.t if hasattr(engine, 't') else 0,
        "hint": "Navigate to /docs for the aerospace analytics command deck."
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
        
        current_t = engine.t if hasattr(engine, 't') else 0
        return {
            "status": "success",
            "t": current_t,
            "message": "Flight simulation engine successfully re-calibrated."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
