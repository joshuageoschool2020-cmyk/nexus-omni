"""
nexus_omni/core_engine.py
──────────────────────────────────────────────────────────────────────────────
Core mathematical engine for the Nexus-Omni Simulator.
Computes one synchronised simulation tick across four coupled domains:

  Domain 1 │ State-Transition Population Dynamics
            │   SEIR compartmental ODE system, integrated via Euler's method.
            │   Outputs: state vector [S, E, I, R], R₀, dominant eigenvalue.

  Domain 2 │ Spatial / Fluid Vector Field
            │   2-D concentration field advanced by discrete diffusion
            │   (5-point Laplacian) + first-order upwind advection.
            │   Outputs: N×N concentration grid, (Vx, Vy) velocity arrays.

  Domain 3 │ Resource Supply Chain Optimisation
            │   Per-node inventory balance with capacity constraints.
            │   Outputs: inventory vector, bottleneck flags, stress score.

  Domain 4 │ Statistical Risk Matrix Scoring
            │   Weighted composite of the three domain-level risk signals.
            │   Outputs: scalar risk score [0,1], per-domain breakdown.

Dependencies: numpy, scipy
Compatibility: Python 3.10+

Usage:
  python core_engine.py          # runs a self-contained smoke-test
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# §1  PARAMETER CONTAINER
#     One dataclass holds every tunable knob for the whole engine.
#     The FastAPI layer will later deserialise JSON payloads into this type.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationParams:
    """
    All adjustable parameters for one simulation run.
    Defaults represent a plausible mid-severity scenario.

    Mathematical notation used in comments:
      β  – transmission rate (day⁻¹)
      σ  – incubation rate   (day⁻¹, = 1 / incubation_period)
      γ  – recovery rate     (day⁻¹, = 1 / infectious_period)
      D  – diffusion coefficient (grid-units² / day)
      dt – time step (days)
    """

    # ── Domain 1: SEIR Compartmental Model ───────────────────────────────────
    beta:       float = 0.30   # β: contacts × P(infection per contact)
    gamma:      float = 0.05   # γ: recovery rate  → avg infectious period = 20 d
    sigma:      float = 0.20   # σ: incubation rate → avg latent period    =  5 d
    population: int   = 10_000 # N: total (closed) population

    # ── Domain 2: Spatial Diffusion + Advection ───────────────────────────────
    grid_size:       int   = 12    # produces a 12×12 concentration grid
    diffusion_coeff: float = 0.08  # D: aerosol / particle diffusion constant
    wind_vx:         float = 0.05  # advection velocity, x-axis (grid-units/day)
    wind_vy:         float = 0.03  # advection velocity, y-axis (grid-units/day)

    # ── Domain 3: Supply Chain ────────────────────────────────────────────────
    num_nodes:          int   = 5
    base_capacity:      float = 1_000.0  # max inventory per node (units)
    inflow_rates:       list  = field(default_factory=lambda: [120, 90, 150, 80, 110])
    outflow_rates:      list  = field(default_factory=lambda: [100, 110, 130, 70, 100])
    capacity_threshold: float = 0.85     # fraction; above this ⟹ bottleneck

    # ── Domain 4: Risk Scoring ────────────────────────────────────────────────
    # Weights sum to 1.0; order: [infection_pressure, spatial_spread, supply_stress]
    risk_weights: Tuple[float, ...] = (0.45, 0.30, 0.25)

    # ── Global ────────────────────────────────────────────────────────────────
    dt: float = 1.0  # time step in days


# ══════════════════════════════════════════════════════════════════════════════
# §2  SIMULATION STATE  (output snapshot for one tick)
#     Downstream: every field maps directly to one visualisation panel.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationState:
    """
    Immutable snapshot produced by NexusOmniEngine.tick().
    The frontend will consume this as a JSON-serialisable dict.
    """

    # Domain 1 outputs
    seir_vector:        np.ndarray  # shape (4,)  → [S, E, I, R] counts
    r_naught:           float       # R₀ = β / γ  (dimensionless threshold)
    dominant_eigenvalue: float      # λ_max of next-generation Jacobian at DFE

    # Domain 2 outputs
    diffusion_grid:  np.ndarray                          # shape (N, N)
    velocity_field:  Tuple[np.ndarray, np.ndarray]      # (Vx, Vy) each (N, N)

    # Domain 3 outputs
    inventory_vector:    np.ndarray  # shape (num_nodes,)  current stock levels
    bottleneck_flags:    np.ndarray  # shape (num_nodes,)  bool: True ⟹ stressed
    supply_stress_score: float       # scalar ∈ [0, 1]

    # Domain 4 outputs
    risk_score:     float        # composite risk ∈ [0, 1]
    risk_breakdown: np.ndarray   # shape (3,)  weighted per-domain contributions

    def summary(self) -> str:
        """Human-readable one-liner for quick CLI inspection."""
        S, E, I, R = self.seir_vector
        bn = int(np.sum(self.bottleneck_flags))
        return (
            f"SEIR=[S:{S:.0f} E:{E:.0f} I:{I:.0f} R:{R:.0f}]  "
            f"R₀={self.r_naught:.3f}  λ={self.dominant_eigenvalue:.4f}  "
            f"SupplyStress={self.supply_stress_score:.3f}  "
            f"Bottlenecks={bn}/{len(self.bottleneck_flags)}  "
            f"Risk={self.risk_score:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §3  CORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class NexusOmniEngine:
    """
    Master simulation engine.

    Lifecycle:
      1. engine = NexusOmniEngine(params)   # initialise internal state
      2. state  = engine.tick()              # advance one time step
      3. repeat step 2 for time-series data

    Internal mutable state:
      _seir      – current SEIR count vector (shape 4)
      _grid      – current concentration field (shape N×N)
      _inventory – current inventory per supply node (shape num_nodes)
    """

    def __init__(self, params: SimulationParams, seed: int = 42):
        self.params = params
        self.rng    = np.random.default_rng(seed)
        self._init_state()

    # ── Private: initialise mutable state ────────────────────────────────────

    def _init_state(self) -> None:
        """Set meaningful starting conditions for every domain."""
        p = self.params
        N = p.population

        # Domain 1 — seed 1 % of population as Exposed
        e0 = max(1, int(0.01 * N))
        self._seir = np.array([N - e0, e0, 0, 0], dtype=float)

        # Domain 2 — Gaussian concentration blob centred on grid
        g = p.grid_size
        cx, cy = g // 2, g // 2
        xs, ys = np.meshgrid(np.arange(g), np.arange(g))
        sigma_g = g / 6.0
        self._grid = np.exp(
            -((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma_g ** 2)
        )

        # Domain 3 — start every node at 60 % of capacity
        self._inventory = np.full(p.num_nodes, 0.60 * p.base_capacity)

    # ── Domain 1: SEIR Euler step ─────────────────────────────────────────────

    def _seir_tick(self) -> Tuple[np.ndarray, float, float]:
        """
        Advance the SEIR compartmental ODE system one step using Euler's method.

        Governing equations:
          dS/dt = -β·(I/N)·S
          dE/dt =  β·(I/N)·S  -  σ·E
          dI/dt =  σ·E         -  γ·I
          dR/dt =  γ·I

        Threshold analysis (disease-free equilibrium, S ≈ N):
          R₀   = β / γ          (basic reproduction number)
          λ_max = σ·(R₀ - 1)   (dominant eigenvalue of next-generation matrix)
                                 λ > 0 ⟹ epidemic grows; λ < 0 ⟹ dies out

        Returns
        -------
        seir  : updated state vector [S, E, I, R]
        r0    : R₀ scalar
        dom_ev: dominant eigenvalue λ_max
        """
        p = self.params
        S, E, I, R = self._seir
        N = float(p.population)

        # Force of infection (λ = β·I/N  →  dimensionless per-day hazard)
        lam = p.beta * I / N

        # ODE right-hand side
        dS = -lam * S
        dE =  lam * S  - p.sigma * E
        dI =  p.sigma * E  - p.gamma * I
        dR =  p.gamma * I

        # Euler integration
        delta = np.array([dS, dE, dI, dR]) * p.dt
        self._seir = np.clip(self._seir + delta, 0.0, N)

        # Threshold metrics (independent of current state — always computed at DFE)
        r0      = p.beta / p.gamma
        dom_ev  = p.sigma * (r0 - 1.0)   # sign determines epidemic fate

        return self._seir.copy(), r0, dom_ev

    # ── Domain 2: Spatial diffusion + advection ───────────────────────────────

    def _spatial_tick(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Advance the 2-D concentration field u(x,y,t) one step.

        PDE (advection-diffusion):
          ∂u/∂t = D·∇²u  −  (vx·∂u/∂x + vy·∂u/∂y)

        Discretisation:
          ∇²u  → 5-point Laplacian stencil (finite differences)
          ∂u/∂x → central difference (first-order upwind)
          Boundary → periodic wrap (roll), simplifies edge handling

        Returns
        -------
        grid        : updated N×N concentration field
        (Vx, Vy)   : uniform velocity field arrays (extended to N×N for visualisation)
        """
        p  = self.params
        u  = self._grid
        D  = p.diffusion_coeff
        dt = p.dt

        # ── Diffusion: 5-point stencil Laplacian ──────────────────────────
        # ∇²u ≈ u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] − 4·u[i,j]
        laplacian = (
            np.roll(u, +1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, +1, axis=1) + np.roll(u, -1, axis=1) -
            4.0 * u
        )

        # ── Advection: central-difference gradient ─────────────────────────
        du_dx = (np.roll(u, -1, axis=1) - np.roll(u, +1, axis=1)) / 2.0
        du_dy = (np.roll(u, -1, axis=0) - np.roll(u, +1, axis=0)) / 2.0
        advection = p.wind_vx * du_dx + p.wind_vy * du_dy

        # ── Combined update (no-negative-concentration clamp) ─────────────
        self._grid = np.clip(u + dt * (D * laplacian - advection), 0.0, None)

        # ── Velocity field arrays (uniform background + small curl stub) ──
        g  = p.grid_size
        Vx = np.full((g, g), p.wind_vx)
        Vy = np.full((g, g), p.wind_vy)
        # TODO (Step 3): add curl / divergence-free noise for richer visualisation

        return self._grid.copy(), (Vx, Vy)

    # ── Domain 3: Supply chain inventory balance ──────────────────────────────

    def _supply_tick(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Update per-node inventory using a linear flow balance model.

        Balance equation (per node i):
          inventory_i[t+1] = clip( inventory_i[t] + (inflow_i − outflow_i)·dt,
                                   0, base_capacity )

        Bottleneck detection:
          Node is flagged if utilisation ≥ capacity_threshold (over-stressed)
          OR utilisation ≤ 5 % (near-empty / stockout risk).

        Stress score:
          Mean absolute deviation of utilisation from the "ideal" 50 % midpoint,
          rescaled to [0, 1].  Score = 1 means all nodes are at extreme states.

        Returns
        -------
        inventory       : shape (num_nodes,) updated stock levels
        bottleneck_flags: shape (num_nodes,) bool array
        stress_score    : scalar [0, 1]
        """
        p       = self.params
        inflow  = np.array(p.inflow_rates,  dtype=float)
        outflow = np.array(p.outflow_rates, dtype=float)

        # Flow balance with hard capacity limits
        self._inventory = np.clip(
            self._inventory + (inflow - outflow) * p.dt,
            0.0,
            p.base_capacity,
        )

        utilisation      = self._inventory / p.base_capacity
        bottleneck_flags = (utilisation >= p.capacity_threshold) | (utilisation <= 0.05)

        # Deviation from ideal 50 % → rescale to [0, 1]
        stress_score = float(np.mean(np.abs(utilisation - 0.5))) * 2.0
        stress_score = min(stress_score, 1.0)

        return self._inventory.copy(), bottleneck_flags, stress_score

    # ── Domain 4: Composite risk scoring ─────────────────────────────────────

    def _risk_tick(
        self,
        seir:      np.ndarray,
        grid:      np.ndarray,
        stress:    float,
    ) -> Tuple[float, np.ndarray]:
        """
        Combine three domain-level risk signals into one composite score.

        Risk signals:
          infection_risk  = I / N            ∈ [0, 1]
          spatial_risk    = mean(u) / (1 + max(u))  ∈ [0, 1]  (normalised spread)
          supply_risk     = stress_score              ∈ [0, 1]

        Composite:
          risk = Σ  weight_k · raw_k          ∈ [0, 1]  (capped)

        Returns
        -------
        composite  : scalar risk score
        breakdown  : shape (3,) weighted per-domain contributions
        """
        p = self.params
        N = float(p.population)
        w = np.array(p.risk_weights)

        infection_risk = seir[2] / N
        spatial_risk   = float(np.mean(grid) / (1.0 + np.max(grid) + 1e-9))
        supply_risk    = stress

        raw       = np.array([infection_risk, spatial_risk, supply_risk])
        breakdown = raw * w
        composite = float(np.sum(breakdown))
        composite = min(composite, 1.0)

        return composite, breakdown

    # ── Public: single simulation tick ───────────────────────────────────────

    def tick(self) -> SimulationState:
        """
        Advance all four coupled domains by one time step (dt).

        Coupling topology:
          Domain 1 → Domain 4  (infection count drives infection_risk)
          Domain 2 → Domain 4  (spatial spread drives spatial_risk)
          Domain 3 → Domain 4  (supply stress drives supply_risk)
          Domain 1 → Domain 2  (TODO Step 4: source term coupling)

        Returns
        -------
        SimulationState: complete snapshot of the current system state
        """
        seir,      r0,    dom_ev = self._seir_tick()
        grid,      vel           = self._spatial_tick()
        inventory, flags, stress = self._supply_tick()
        risk,      breakdown     = self._risk_tick(seir, grid, stress)

        return SimulationState(
            seir_vector         = seir,
            r_naught            = r0,
            dominant_eigenvalue = dom_ev,
            diffusion_grid      = grid,
            velocity_field      = vel,
            inventory_vector    = inventory,
            bottleneck_flags    = flags,
            supply_stress_score = stress,
            risk_score          = risk,
            risk_breakdown      = breakdown,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §4  STANDALONE SMOKE TEST
#     Run:  python core_engine.py
#     Expected: 30 ticks of console output with no errors.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  Nexus-Omni │ Core Engine │ Smoke Test (30 ticks)")
    print("=" * 72)

    params = SimulationParams()
    engine = NexusOmniEngine(params)

    # Quick parameter sanity print
    print(f"\n[Config]  N={params.population}  β={params.beta}  "
          f"σ={params.sigma}  γ={params.gamma}  "
          f"R₀(expected)={params.beta / params.gamma:.2f}\n")

    for t in range(30):
        state = engine.tick()
        if t % 5 == 0 or t == 29:
            print(f"  t={t+1:>2}  {state.summary()}")

    # Domain 2: final grid stats
    g = state.diffusion_grid
    print(f"\n[Spatial]  grid min={g.min():.5f}  max={g.max():.5f}  "
          f"mean={g.mean():.5f}")

    # Domain 3: final inventory
    print(f"[Supply ]  inventory={np.round(state.inventory_vector, 1)}")
    print(f"           bottlenecks={state.bottleneck_flags}")

    # Domain 4: risk breakdown labels
    labels = ["infection", "spatial ", "supply  "]
    print(f"\n[Risk   ]  composite={state.risk_score:.4f}")
    for lbl, val in zip(labels, state.risk_breakdown):
        bar = "█" * int(val * 40)
        print(f"           {lbl}: {val:.4f}  {bar}")

    print("\n✓  Smoke test passed — engine is ready for FastAPI integration.\n")
