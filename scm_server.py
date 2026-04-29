import json
import math
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

# --------------- Optional imports ---------------
_HAS_FCI = False
_HAS_PC = False
_HAS_SK = False

try:
    from causallearn.search.ConstraintBased.FCI import fci
    _HAS_FCI = True
except Exception:
    pass

try:
    from causallearn.search.ConstraintBased.PC import pc
    _HAS_PC = True
except Exception:
    pass

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SK = True
except Exception:
    pass


def _json_response(handler: BaseHTTPRequestHandler, code: int, data: Dict[str, Any]) -> None:
    payload = json.dumps(data).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


# ================================================================
#  Observation record: (Yellow, Action, Accident, is_random, epoch)
# ================================================================
class Observation:
    """Single observation with metadata for valid causal inference."""
    __slots__ = ("y", "u", "a", "is_random", "epoch")

    def __init__(self, y: int, u: int, a: int, is_random: bool, epoch: int):
        self.y = y            # proxy (Yellow)
        self.u = u            # treatment: 1=fast, 0=slow
        self.a = a            # outcome (Accident)
        self.is_random = is_random  # was action randomized?
        self.epoch = epoch    # 0=naive, 1=causal


class CausalSCMEngine:
    """
    Core causal inference engine.

    Design principles:
      - FCI discovery uses ONLY randomized-action data (interventional validity)
      - Causal effect estimation uses proxy-stratified estimator (not IPW)
        because the confounder (Slippery) is unmeasured; Yellow is a proxy
      - Policy uses Thompson Sampling with stratified Beta posteriors
      - Rosenbaum sensitivity analysis quantifies robustness to unmeasured confounding
    """

    def __init__(self) -> None:
        self.obs: List[Observation] = []
        self.current_epoch: int = 0  # 0=naive, 1=causal

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def add_obs(self, y: int, action: str, a: int, is_random: bool = False) -> None:
        u = 1 if action == "fast" else 0
        self.obs.append(Observation(int(y), u, int(a), is_random, self.current_epoch))

    def reset(self) -> Dict[str, Any]:
        self.obs.clear()
        self.current_epoch = 0
        return {"status": "reset_ok", **self.counts()}

    # --------------- Counting helpers ---------------
    def counts(self) -> Dict[str, int]:
        if not self.obs:
            return {"n": 0, "n_fast": 0, "n_slow": 0, "n_random": 0}
        n_fast = sum(1 for o in self.obs if o.u == 1)
        n_slow = sum(1 for o in self.obs if o.u == 0)
        n_rand = sum(1 for o in self.obs if o.is_random)
        return {"n": len(self.obs), "n_fast": n_fast, "n_slow": n_slow, "n_random": n_rand}

    def _to_array(self, only_random: bool = False, epoch: Optional[int] = None) -> Optional[np.ndarray]:
        """Convert observations to numpy array with optional filters."""
        filtered = self.obs
        if only_random:
            filtered = [o for o in filtered if o.is_random]
        if epoch is not None:
            filtered = [o for o in filtered if o.epoch == epoch]
        if not filtered:
            return None
        return np.array([(o.y, o.u, o.a) for o in filtered], dtype=float)

    # ================================================================
    #  FIX #1: Discovery uses ONLY randomized data
    # ================================================================
    def discover(self, min_n: int = 50) -> Dict[str, Any]:
        """
        Causal structure discovery using ONLY randomized-action observations.

        Rationale: FCI/PC assume no selection bias. When the agent's policy
        makes Action depend on Yellow, this creates a spurious Y→U edge.
        Using only random-action data ensures Action ⊥ Yellow|∅, letting
        FCI detect the true latent structure.
        """
        c = self.counts()
        random_data = self._to_array(only_random=True)

        if random_data is None or len(random_data) < min_n:
            n_rand = 0 if random_data is None else len(random_data)
            return {
                "status_code": "insufficient",
                "status_text": (
                    f"Discovery: insufficient random data "
                    f"(n_random={n_rand}, min={min_n}). "
                    f"Epsilon-greedy exploration should continue."
                ),
                "method": "NONE",
                "n_random": n_rand,
                "min_required": min_n,
                **c,
            }

        if _HAS_FCI:
            try:
                G, edges = fci(random_data)
                result = self._interpret_graph(G, "FCI")
                result["n_random_used"] = len(random_data)
                return result
            except Exception as e:
                return {
                    "status_code": "error",
                    "status_text": f"Discovery(FCI) error: {e}",
                    "method": "FCI",
                    **c,
                }

        if _HAS_PC:
            try:
                cg = pc(random_data)
                result = self._interpret_graph(cg.G, "PC_DEGRADED")
                result["status_text"] += " [WARNING: FCI unavailable, PC cannot detect latent variables]"
                result["n_random_used"] = len(random_data)
                return result
            except Exception:
                pass

        return {
            "status_code": "no_fci",
            "status_text": "Discovery skipped: FCI library not installed.",
            "method": "NONE",
            **c,
        }

    def _interpret_graph(self, G: Any, method: str) -> Dict[str, Any]:
        """
        Interpret FCI PAG output using edge MARKS, not just adjacency.

        FCI edge encoding in causal-learn:
          -1 = tail (definite non-cause)
           1 = arrowhead (definite cause or latent)
           2 = circle (ambiguous: could be tail or arrowhead)
           0 = no edge

        Key edge patterns:
          (2, 1) = o→  : possible causal or latent common cause
          (1, 1) = ↔   : definite latent common cause (bidirected)
          (-1, 1) = →  : definite direct cause
          (2, 2) = o-o : fully ambiguous
          (0, 0) = no edge : d-separated

        Proxy detection logic:
          1. Y-A edge has circle or bidirected mark → latent structure possible
          2. Y-U has NO edge → Action is independent of Yellow (randomization OK)
          3. Combination of (1) + (2) → Yellow is proxy, not direct cause

        Column ordering: [Yellow=0, Action=1, Accident=2]
        """
        result = {"method": method}

        try:
            mat = getattr(G, "graph", None)
            if mat is None:
                result["status_code"] = "unknown"
                result["status_text"] = f"Discovery({method}): graph matrix not available."
                return result

            mat = np.array(mat)

            # Extract edge marks
            ya_marks = (int(mat[0, 2]), int(mat[2, 0]))  # Yellow-Accident
            yu_marks = (int(mat[0, 1]), int(mat[1, 0]))  # Yellow-Action
            ua_marks = (int(mat[1, 2]), int(mat[2, 1]))  # Action-Accident

            result["edge_marks"] = {
                "yellow_accident": ya_marks,
                "yellow_action": yu_marks,
                "action_accident": ua_marks,
            }

            # Classify Yellow-Accident edge
            ya_adjacent = (ya_marks[0] != 0) or (ya_marks[1] != 0)
            ya_has_circle = 2 in ya_marks         # o→ or o-o
            ya_bidirected = ya_marks == (1, 1)     # ↔
            ya_direct = ya_marks == (-1, 1)        # →  (Y causes A directly)

            # Classify Yellow-Action edge
            yu_independent = (yu_marks[0] == 0) and (yu_marks[1] == 0)

            # Classify Action-Accident edge
            ua_has_circle = 2 in ua_marks
            ua_bidirected = ua_marks == (1, 1)

            result["ya_adjacent"] = ya_adjacent
            result["yu_independent"] = yu_independent
            result["has_latent_indicator"] = ya_bidirected or ya_has_circle or ua_bidirected

            # --- Decision logic ---

            if not ya_adjacent:
                # No Y-A edge at all: d-separated (unlikely with enough data, but possible)
                result["status_code"] = "proxy"
                result["status_text"] = (
                    f"Discovery({method}): Y-A no edge (d-separated). "
                    f"Yellow is proxy. Latent confounder likely."
                )

            elif ya_direct:
                # Y → A: FCI thinks Yellow directly causes Accident
                result["status_code"] = "direct_cause"
                result["status_text"] = (
                    f"Discovery({method}): Y → A (direct cause detected). "
                    f"Yellow may directly influence Accident. "
                    f"Proxy interpretation not supported."
                )

            elif ya_bidirected:
                # Y ↔ A: Latent common cause confirmed by FCI
                result["status_code"] = "proxy"
                result["status_text"] = (
                    f"Discovery({method}): Y ↔ A (bidirected edge). "
                    f"Latent common cause confirmed. "
                    f"Yellow is proxy for latent confounder."
                )

            elif ya_has_circle and yu_independent:
                # Y o→ A with Y ⊥ U: Most common pattern with random actions
                # Circle mark means FCI cannot determine if Y→A or Y↔A
                # Combined with Y⊥U (confirmed by randomization), this is
                # strong evidence for latent structure (proxy interpretation)
                result["status_code"] = "proxy"
                result["status_text"] = (
                    f"Discovery({method}): Y o→ A with Y⊥Action. "
                    f"Circle mark indicates possible latent common cause. "
                    f"Yellow is proxy for latent confounder (high confidence)."
                )

            elif ya_has_circle:
                # Y o→ A but Y-U edge exists (shouldn't happen with random actions)
                result["status_code"] = "ambiguous_latent"
                result["status_text"] = (
                    f"Discovery({method}): Y o→ A with Y-U edge. "
                    f"Latent structure possible but action independence violated. "
                    f"Check randomization quality."
                )

            else:
                result["status_code"] = "adjacent"
                result["status_text"] = (
                    f"Discovery({method}): Y-A edge type ({ya_marks}). "
                    f"Structure unclear, more data needed."
                )

        except Exception as e:
            result["status_code"] = "error"
            result["status_text"] = f"Discovery({method}): interpretation error: {e}"

        return result

    # ================================================================
    #  FIX #2: Proxy-Stratified Estimator (replaces biased IPW)
    # ================================================================
    def proxy_stratified_estimate(self, min_per_cell: int = 5) -> Dict[str, Any]:
        """
        Proxy-stratified causal effect estimator.

        Since Yellow is a proxy for the unmeasured confounder (Cleaning/Slippery),
        standard IPW (conditioning only on Yellow) is BIASED.

        Instead, we compute stratified estimates within Y-strata and report:
          1. Stratified risk differences (informative bounds)
          2. Weighted average assuming proxy ≈ confounder (optimistic estimate)
          3. Worst-case bounds (no assumptions about proxy quality)

        This is honest about what we can and cannot identify.

        DAG: Cleaning → {Yellow, Slippery}, Slippery × Action → Accident
        Yellow is NOT a sufficient adjustment set for Action → Accident.
        But stratifying on Yellow provides PARTIAL information.
        """
        c = self.counts()
        arr = self._to_array()
        if arr is None or c["n"] < 10:
            return {"stable": False, "reason": "insufficient_data", **c}

        y_col = arr[:, 0].astype(int)
        u_col = arr[:, 1].astype(int)
        a_col = arr[:, 2].astype(int)

        strata = {}
        for yval in [0, 1]:
            for uval in [0, 1]:
                mask = (y_col == yval) & (u_col == uval)
                accidents = a_col[mask]
                label = f"y{yval}_{'fast' if uval == 1 else 'slow'}"
                strata[label] = {
                    "n": int(accidents.size),
                    "n_accidents": int(np.sum(accidents)),
                    "rate": float(np.mean(accidents)) if accidents.size > 0 else None,
                }

        # Check minimum cell sizes
        all_sufficient = all(
            strata[k]["n"] >= min_per_cell for k in strata
        )

        if not all_sufficient:
            return {
                "stable": False,
                "reason": "insufficient_cell_sizes",
                "strata": strata,
                "min_per_cell": min_per_cell,
                **c,
            }

        # --- Stratified estimates ---
        # Within each Y-stratum, compute P(Accident | Action, Y)
        # These are NOT causal effects (Yellow doesn't block the backdoor)
        # but they provide bounds and diagnostics.

        risk_fast_y0 = strata["y0_fast"]["rate"]
        risk_slow_y0 = strata["y0_slow"]["rate"]
        risk_fast_y1 = strata["y1_fast"]["rate"]
        risk_slow_y1 = strata["y1_slow"]["rate"]

        # Stratum-specific risk differences
        rd_y0 = risk_fast_y0 - risk_slow_y0  # RD when Yellow=0
        rd_y1 = risk_fast_y1 - risk_slow_y1  # RD when Yellow=1

        # Marginal P(Y=1)
        p_y1 = float(np.mean(y_col))
        p_y0 = 1.0 - p_y1

        # Proxy-weighted average (ASSUMPTION: Yellow captures most confounding)
        # This is an approximation, NOT an identified quantity
        risk_fast_adj = p_y0 * risk_fast_y0 + p_y1 * risk_fast_y1
        risk_slow_adj = p_y0 * risk_slow_y0 + p_y1 * risk_slow_y1
        rd_adjusted = risk_fast_adj - risk_slow_adj

        # --- Partial Identification / Manski-style Bounds ---
        # Worst-case: within each stratum, the hidden confounder could
        # completely reverse the within-stratum comparison.
        # Under monotonicity (higher Slippery → higher Accident):
        #   Lower bound on causal RD = min(rd_y0, rd_y1)
        #   Upper bound on causal RD = max(rd_y0, rd_y1)
        rd_lower = min(rd_y0, rd_y1)
        rd_upper = max(rd_y0, rd_y1)

        # Sign agreement = both strata agree on direction
        sign_agreement = (rd_y0 > 0 and rd_y1 > 0) or (rd_y0 < 0 and rd_y1 < 0)

        return {
            "stable": True,
            "method": "proxy_stratified",
            # Point estimates (with proxy-adjustment caveat)
            "risk_fast": risk_fast_adj,
            "risk_slow": risk_slow_adj,
            "diff": rd_adjusted,
            # Stratum-specific
            "rd_y0": rd_y0,
            "rd_y1": rd_y1,
            "sign_agreement": sign_agreement,
            # Bounds
            "rd_lower_bound": rd_lower,
            "rd_upper_bound": rd_upper,
            # Cell details
            "strata": strata,
            "p_y1": p_y1,
            # Diagnostics
            "WARNING": (
                "Yellow is a proxy, not a sufficient adjustment variable. "
                "Estimates are bounds/approximations, not point-identified. "
                "See /sensitivity for robustness analysis."
            ),
            **c,
        }

    # ================================================================
    #  FIX #3: Rosenbaum Sensitivity Analysis
    # ================================================================
    def sensitivity_analysis(
        self,
        gamma_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Rosenbaum-style sensitivity analysis.

        For each Γ ≥ 1, compute bounds on the treatment effect assuming
        unmeasured confounding could make one unit up to Γ times more likely
        to receive treatment than another with the same observed covariates.

        Γ = 1.0 → no unmeasured confounding (standard estimate)
        Γ > 1.0 → how much confounding would overturn our conclusion?

        The "tipping point" Γ* where the bound crosses zero tells us
        how robust our finding is.
        """
        if gamma_values is None:
            gamma_values = [1.0, 1.5, 2.0, 3.0, 5.0]

        arr = self._to_array()
        if arr is None or len(arr) < 20:
            return {"stable": False, "reason": "insufficient_data"}

        y_col = arr[:, 0].astype(int)
        u_col = arr[:, 1].astype(int)
        a_col = arr[:, 2].astype(int)

        results = []
        tipping_gamma = None

        for gamma in sorted(gamma_values):
            bounds = self._rosenbaum_bounds(y_col, u_col, a_col, gamma)
            results.append({"gamma": gamma, **bounds})

            # Find tipping point: smallest gamma where bound crosses zero
            if tipping_gamma is None:
                if bounds["rd_lower"] <= 0 <= bounds["rd_upper"]:
                    tipping_gamma = gamma

        return {
            "stable": True,
            "method": "rosenbaum_sensitivity",
            "gamma_results": results,
            "tipping_gamma": tipping_gamma,
            "interpretation": self._interpret_sensitivity(tipping_gamma),
            **self.counts(),
        }

    def _rosenbaum_bounds(
        self,
        y_col: np.ndarray,
        u_col: np.ndarray,
        a_col: np.ndarray,
        gamma: float,
    ) -> Dict[str, float]:
        """
        Compute bounds on stratified risk difference for a given Γ.

        For each Y-stratum, the propensity can be off by factor Γ:
          1/(1+Γ) ≤ P(U=1|Y, hidden) ≤ Γ/(1+Γ)

        We compute best-case and worst-case risk differences.
        """
        rd_lowers = []
        rd_uppers = []

        for yval in [0, 1]:
            mask = y_col == yval
            u_s = u_col[mask]
            a_s = a_col[mask]

            n_fast = np.sum(u_s == 1)
            n_slow = np.sum(u_s == 0)

            if n_fast < 2 or n_slow < 2:
                continue

            # Observed rates
            r_fast = float(np.mean(a_s[u_s == 1]))
            r_slow = float(np.mean(a_s[u_s == 0]))

            # Under Γ-confounding, the "true" rates could be shifted
            # The intuition: hidden confounder could make fast-choosers
            # inherently more/less accident-prone by factor Γ
            # Bounds on risk difference:
            shift = (gamma - 1.0) / (gamma + 1.0)

            rd_lower = (r_fast - shift) - (r_slow + shift)
            rd_upper = (r_fast + shift) - (r_slow - shift)

            # Clip to [-1, 1]
            rd_lower = max(rd_lower, -1.0)
            rd_upper = min(rd_upper, 1.0)

            rd_lowers.append(rd_lower)
            rd_uppers.append(rd_upper)

        if not rd_lowers:
            return {"rd_lower": -1.0, "rd_upper": 1.0}

        return {
            "rd_lower": float(min(rd_lowers)),
            "rd_upper": float(max(rd_uppers)),
        }

    def _interpret_sensitivity(self, tipping_gamma: Optional[float]) -> str:
        if tipping_gamma is None:
            return (
                "Results are robust across all tested Gamma values. "
                "High resilience to unmeasured confounding."
            )
        if tipping_gamma >= 3.0:
            return (
                f"Gamma*={tipping_gamma:.1f}: Results remain robust even "
                f"under strong unmeasured confounding."
            )
        if tipping_gamma >= 1.5:
            return (
                f"Gamma*={tipping_gamma:.1f}: Results are resilient to "
                f"moderate unmeasured confounding. Interpret with caution."
            )
        return (
            f"Gamma*={tipping_gamma:.1f}: Results may flip even under "
            f"weak confounding. High uncertainty."
        )

    # ================================================================
    #  FIX #4: Thompson Sampling — Causal Bandits Policy
    # ================================================================
    def thompson_policy(
        self,
        yellow: int,
        explore: bool = True,
        min_per_arm: int = 8,
    ) -> Dict[str, Any]:
        """
        Thompson Sampling policy with causal awareness.

        Phase 1 (explore=True):
          - Ensures both actions are sampled in each Y-stratum (positivity)
          - Under-sampled arms get priority
          - Otherwise, Thompson Sampling from Beta posteriors

        Phase 2 (explore=False):
          - Pure exploitation from posterior
          - Falls back to proxy-stratified risk comparison

        Reference: Lattimore et al. (2016) "Causal Bandits"
        We use Y-stratified posteriors because Yellow (proxy) is the
        only observed pre-treatment variable. This is a contextual
        bandit where context = Yellow value.
        """
        if not self.obs:
            return {"best_action": "slow", "reason": "no_data_default_slow"}

        arr = self._to_array()
        y_col = arr[:, 0].astype(int)
        u_col = arr[:, 1].astype(int)
        a_col = arr[:, 2].astype(int)

        # Use Y-stratum if enough data, else global
        mask = y_col == int(yellow)
        used_stratum = True
        if np.sum(mask) < 10:
            mask = np.ones(len(y_col), dtype=bool)
            used_stratum = False

        acc_fast = a_col[mask & (u_col == 1)]
        acc_slow = a_col[mask & (u_col == 0)]
        n_fast = int(acc_fast.size)
        n_slow = int(acc_slow.size)

        # Phase 1: ensure coverage (positivity for valid inference)
        if explore:
            if n_fast < min_per_arm and n_slow >= min_per_arm:
                return self._policy_response(
                    "fast", "explore_coverage_fast", n_fast, n_slow,
                    used_stratum, True
                )
            if n_slow < min_per_arm and n_fast >= min_per_arm:
                return self._policy_response(
                    "slow", "explore_coverage_slow", n_fast, n_slow,
                    used_stratum, True
                )
            if n_fast < min_per_arm and n_slow < min_per_arm:
                # Both under-sampled: alternate (coin flip)
                action = "fast" if np.random.random() < 0.5 else "slow"
                return self._policy_response(
                    action, "explore_both_under_sampled", n_fast, n_slow,
                    used_stratum, True
                )

        # Minimum data check for exploitation
        if n_fast < 1 or n_slow < 1:
            return self._policy_response(
                "slow", "insufficient_default_slow", n_fast, n_slow,
                used_stratum, explore
            )

        # --- Thompson Sampling ---
        # Beta posteriors: Beta(successes + 1, failures + 1)
        # Here "success" = NO accident (we want to MINIMIZE accidents)
        safe_fast = int(np.sum(acc_fast == 0))
        fail_fast = int(np.sum(acc_fast == 1))
        safe_slow = int(np.sum(acc_slow == 0))
        fail_slow = int(np.sum(acc_slow == 1))

        # Sample from Beta(safe + 1, fail + 1) — probability of being safe
        theta_fast = np.random.beta(safe_fast + 1, fail_fast + 1)
        theta_slow = np.random.beta(safe_slow + 1, fail_slow + 1)

        # Choose action with higher sampled safety probability
        if explore:
            best = "fast" if theta_fast >= theta_slow else "slow"
            reason = "thompson_sampling"
        else:
            # Exploitation: use posterior mean (deterministic)
            mean_fast = (safe_fast + 1.0) / (n_fast + 2.0)
            mean_slow = (safe_slow + 1.0) / (n_slow + 2.0)
            best = "fast" if mean_fast >= mean_slow else "slow"
            reason = "posterior_mean_exploit"
            theta_fast = mean_fast
            theta_slow = mean_slow

        return {
            "best_action": best,
            "reason": reason,
            "theta_fast": float(theta_fast),
            "theta_slow": float(theta_slow),
            "risk_fast": float(fail_fast + 1) / (n_fast + 2),
            "risk_slow": float(fail_slow + 1) / (n_slow + 2),
            "n_fast": n_fast,
            "n_slow": n_slow,
            "used_stratum": used_stratum,
            "explore": explore,
            "method": "thompson_sampling_causal_bandit",
        }

    def _policy_response(
        self, action: str, reason: str, n_fast: int, n_slow: int,
        used_stratum: bool, explore: bool
    ) -> Dict[str, Any]:
        return {
            "best_action": action,
            "reason": reason,
            "n_fast": n_fast,
            "n_slow": n_slow,
            "used_stratum": used_stratum,
            "explore": explore,
        }

    # ================================================================
    #  Phase readiness (overlap + weight diagnostics)
    # ================================================================
    def phase_ready(
        self,
        min_per_action_stratum: int = 8,
        min_total_random: int = 50,
    ) -> Dict[str, Any]:
        """
        Phase gating: checks whether we have enough randomized data
        AND sufficient overlap for valid causal inference.
        """
        c = self.counts()
        if c["n"] == 0:
            return {"ready": False, "reason": "no_data", **c}

        arr = self._to_array()
        y_col = arr[:, 0].astype(int)
        u_col = arr[:, 1].astype(int)

        # Overlap: both actions in both Y strata
        counts_stratum = {}
        for yval in [0, 1]:
            for uval in [0, 1]:
                label = f"y{yval}_{'fast' if uval == 1 else 'slow'}"
                counts_stratum[label] = int(
                    np.sum((y_col == yval) & (u_col == uval))
                )

        overlap_ok = all(
            counts_stratum[k] >= min_per_action_stratum
            for k in counts_stratum
        )

        # Randomized data sufficiency
        n_random = c["n_random"]
        random_ok = n_random >= min_total_random

        ready = bool(overlap_ok and random_ok)

        return {
            "ready": ready,
            "reason": "ok" if ready else "overlap_or_random_data_insufficient",
            **c,
            **counts_stratum,
            "n_random": n_random,
            "min_per_action_stratum": min_per_action_stratum,
            "min_total_random": min_total_random,
            "overlap_ok": overlap_ok,
            "random_ok": random_ok,
        }

    # ================================================================
    #  Legacy /ace endpoint (now calls proxy_stratified_estimate)
    # ================================================================
    def ace(self, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible ACE endpoint → delegates to proxy-stratified."""
        return self.proxy_stratified_estimate(**kwargs)


# ================================================================
#  Global engine instance
# ================================================================
scm_engine = CausalSCMEngine()


# ================================================================
#  HTTP Server
# ================================================================
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return  # Suppress default logging

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {
                "ok": True,
                **scm_engine.counts(),
                "has_fci": _HAS_FCI,
                "has_pc": _HAS_PC,
                "has_sklearn": _HAS_SK,
                "version": "2.0-causal-bdi",
            })
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}

            # --- /record_observation ---
            if self.path == "/record_observation":
                y = data.get("yellow", data.get("Y", data.get("y")))
                action = data.get("action", data.get("U", data.get("u")))
                acc = data.get("accident", data.get("A", data.get("a")))
                is_random = data.get("is_random", False)

                if isinstance(is_random, str):
                    is_random = is_random.strip().lower() in ("true", "1", "yes")

                if y is None or action is None or acc is None:
                    _json_response(self, 400, {"error": "missing_fields", "received": data})
                    return

                scm_engine.add_obs(int(y), str(action), int(acc), bool(is_random))
                _json_response(self, 200, {"status": "ok", **scm_engine.counts()})
                return

            # --- /reset ---
            if self.path == "/reset":
                _json_response(self, 200, scm_engine.reset())
                return

            # --- /set_epoch ---
            if self.path == "/set_epoch":
                epoch = int(data.get("epoch", 0))
                scm_engine.set_epoch(epoch)
                _json_response(self, 200, {"status": "ok", "epoch": epoch})
                return

            # --- /discover (uses ONLY random data) ---
            if self.path == "/discover":
                min_n = int(data.get("min_n", 50))
                _json_response(self, 200, scm_engine.discover(min_n=min_n))
                return

            # --- /ace (proxy-stratified, backward compatible) ---
            if self.path == "/ace":
                _json_response(self, 200, scm_engine.ace())
                return

            # --- /phase_ready ---
            if self.path == "/phase_ready":
                _json_response(self, 200, scm_engine.phase_ready(
                    min_per_action_stratum=8,
                    min_total_random=50,
                ))
                return

            # --- /optimize_policy (Thompson Sampling) ---
            if self.path == "/optimize_policy":
                y = int(data.get("yellow", 0))
                exp = data.get("explore", True)
                if isinstance(exp, str):
                    exp = exp.strip().lower() not in ("false", "0", "no")
                _json_response(
                    self, 200,
                    scm_engine.thompson_policy(y, explore=bool(exp), min_per_arm=8)
                )
                return

            # --- /sensitivity (NEW) ---
            if self.path == "/sensitivity":
                gammas = data.get("gamma_values", [1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
                _json_response(self, 200, scm_engine.sensitivity_analysis(gamma_values=gammas))
                return

            _json_response(self, 404, {"error": "not_found", "path": self.path})

        except Exception as e:
            _json_response(self, 500, {"error": "server_exception", "detail": str(e)})


if __name__ == "__main__":
    host, port = "127.0.0.1", 8008
    print(f"SCM Server v2 (Causal BDI) — http://{host}:{port}")
    print(f"  FCI: {'✓' if _HAS_FCI else '✗'}  |  PC: {'✓' if _HAS_PC else '✗'}  |  sklearn: {'✓' if _HAS_SK else '✗'}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()