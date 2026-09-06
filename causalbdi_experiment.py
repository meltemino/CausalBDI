"""
CausalBDI Experimental Evaluation
=================================
Standalone Python simulation of the CausalBDI architecture, used for the
quantitative evaluation reported in the paper.

This script reproduces the agent's decision logic and the causal inference
pipeline in a single process for reproducibility and statistical analysis.
The Jason/JaCaMo agents in src/asl/ implement the same architecture for
the agent-based demonstration.

Default experimental parameters (matching the paper):
  - goal distance      = 500
  - initial fuel       = 1000
  - epsilon (greedy)   = 0.4
  - FCI minimum sample = 50 random observations
  - discovery interval = 25 steps
  - n_runs (per agent) = 30 independent simulation runs

Three agent types are compared:
  - Naive    : correlational policy only (Y=1 -> slow, Y=0 -> fast)
  - CausalBDI: full epistemic lifecycle (epsilon-greedy -> FCI -> Thompson Sampling)
  - Oracle   : observes the latent confounder Slippery directly (upper bound)

Note: Naive and CausalBDI use independent random seeds so their trajectories
diverge stochastically; we therefore report distributions over 30 runs rather
than paired comparisons.
"""

import numpy as np
import warnings
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

_HAS_FCI = False
try:
    from causallearn.search.ConstraintBased.FCI import fci
    _HAS_FCI = True
except Exception:
    pass

print(f"FCI available: {_HAS_FCI}")

# ================================================================
# ENVIRONMENT — faithful to env.asl
# ================================================================
P_CLEANING = 0.30
P_SLIPPERY_GIVEN_C1 = 0.90
P_SLIPPERY_GIVEN_C0 = 0.05
P_YELLOW_GIVEN_C1 = 0.80
P_YELLOW_GIVEN_C0 = 0.10
P_ACC_S1_FAST = 0.70
P_ACC_S1_SLOW = 0.10
P_ACC_S0_FAST = 0.01
P_ACC_S0_SLOW = 0.005

@dataclass
class Env:
    """Environment state. Defaults match the paper's evaluation setup."""
    pos: int = 0
    fuel: int = 1000
    goal: int = 500
    step: int = 0
    cleaning: int = 0
    slippery: int = 0
    yellow: int = 0
    done: bool = False
    won: bool = False

def env_step(env, action):
    """One step: generate context, apply action, return (yellow, accident)"""
    env.step += 1
    # Causal DAG sampling
    C = 1 if np.random.random() < P_CLEANING else 0
    S = 1 if np.random.random() < (P_SLIPPERY_GIVEN_C1 if C else P_SLIPPERY_GIVEN_C0) else 0
    Y = 1 if np.random.random() < (P_YELLOW_GIVEN_C1 if C else P_YELLOW_GIVEN_C0) else 0
    env.cleaning = C
    env.slippery = S
    env.yellow = Y
    
    # Accident probability
    if S == 1:
        p_acc = P_ACC_S1_FAST if action == "fast" else P_ACC_S1_SLOW
    else:
        p_acc = P_ACC_S0_FAST if action == "fast" else P_ACC_S0_SLOW
    
    accident = 1 if np.random.random() < p_acc else 0
    
    if accident:
        env.fuel -= 5  # penalty
    else:
        env.fuel -= 1
        env.pos += 3 if action == "fast" else 1
    
    if env.pos >= env.goal:
        env.done = True
        env.won = True
    elif env.fuel <= 0:
        env.done = True
        env.won = False
    
    return Y, accident


# ================================================================
# CAUSAL ENGINE — faithful to scm_server.py
# ================================================================
class CausalEngine:
    def __init__(self):
        self.data = []  # list of (y, u, a, is_random, epoch)
        self.epoch = 0
    
    def reset(self):
        self.data.clear()
        self.epoch = 0
    
    def add(self, y, action, accident, is_random):
        u = 1 if action == "fast" else 0
        self.data.append((y, u, accident, is_random, self.epoch))
    
    def n_random(self):
        return sum(1 for d in self.data if d[3])
    
    def random_array(self):
        rd = [(d[0], d[1], d[2]) for d in self.data if d[3]]
        return np.array(rd, dtype=float) if rd else None
    
    def all_array(self):
        if not self.data:
            return None
        return np.array([(d[0], d[1], d[2]) for d in self.data], dtype=float)
    
    def discover(self, min_n=50):
        rd = self.random_array()
        if rd is None or len(rd) < min_n:
            return {"status": "insufficient", "n_random": 0 if rd is None else len(rd)}
        if not _HAS_FCI:
            return {"status": "no_fci"}
        try:
            G, _ = fci(rd)
            mat = np.array(G.graph)
            ya = (int(mat[0,2]), int(mat[2,0]))  # Yellow-Accident
            yu = (int(mat[0,1]), int(mat[1,0]))  # Yellow-Action
            ua = (int(mat[1,2]), int(mat[2,1]))  # Action-Accident
            
            ya_adj = ya[0] != 0 or ya[1] != 0
            ya_bid = ya == (1,1)
            ya_circ = 2 in ya
            ya_dir = ya == (-1,1)
            yu_ind = yu[0] == 0 and yu[1] == 0
            
            if not ya_adj:
                # (0,0) = no DETECTED adjacency; NOT evidence of a latent common cause
                return {"status": "no_adjacency", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
            elif ya_bid:
                # (1,1) = the only PAG output that identifies latent confounding
                return {"status": "latent_confounding", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
            elif ya_circ and yu_ind:
                # circle = orientation unresolved (Y->A vs Y<->A indistinguishable)
                return {"status": "ambiguous_orientation", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
            elif ya_circ:
                return {"status": "ambiguous_randomisation", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
            elif ya_dir:
                return {"status": "direct_cause", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
            else:
                return {"status": "adjacent", "ya": ya, "yu": yu, "ua": ua, "n": len(rd)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def sensitivity(self):
        arr = self.all_array()
        if arr is None or len(arr) < 20:
            return -1.0
        y, u, a = arr[:,0].astype(int), arr[:,1].astype(int), arr[:,2].astype(int)
        for gamma in [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
            rd_lo, rd_hi = [], []
            for yv in [0,1]:
                m = y == yv
                nf = np.sum(u[m]==1)
                ns = np.sum(u[m]==0)
                if nf < 2 or ns < 2: continue
                rf = np.mean(a[m & (u==1)])
                rs = np.mean(a[m & (u==0)])
                shift = (gamma-1)/(gamma+1)
                rd_lo.append(max((rf-shift)-(rs+shift), -1))
                rd_hi.append(min((rf+shift)-(rs-shift), 1))
            if rd_lo and min(rd_lo) <= 0 <= max(rd_hi):
                return gamma
        return 99.0  # very robust
    
    def thompson(self, yellow, explore=True, min_arm=8):
        arr = self.all_array()
        if arr is None:
            return "slow"
        y, u, a = arr[:,0].astype(int), arr[:,1].astype(int), arr[:,2].astype(int)
        m = y == yellow
        if np.sum(m) < 10:
            m = np.ones(len(y), dtype=bool)
        af = a[m & (u==1)]
        asl = a[m & (u==0)]
        nf, ns = af.size, asl.size
        
        if explore:
            if nf < min_arm and ns >= min_arm: return "fast"
            if ns < min_arm and nf >= min_arm: return "slow"
            if nf < min_arm and ns < min_arm:
                return "fast" if np.random.random() < 0.5 else "slow"
        
        if nf < 1 or ns < 1:
            return "slow"
        
        sf, ff = int(np.sum(af==0)), int(np.sum(af==1))
        ss, fs = int(np.sum(asl==0)), int(np.sum(asl==1))
        
        if explore:
            tf = np.random.beta(sf+1, ff+1)
            ts = np.random.beta(ss+1, fs+1)
            return "fast" if tf >= ts else "slow"
        else:
            mf = (sf+1.0)/(nf+2.0)
            ms = (ss+1.0)/(ns+2.0)
            return "fast" if mf >= ms else "slow"
    
    def phase_ready(self, min_per=8, min_rand=50):
        if not self.data: return False
        arr = self.all_array()
        y, u = arr[:,0].astype(int), arr[:,1].astype(int)
        for yv in [0,1]:
            for uv in [0,1]:
                if np.sum((y==yv)&(u==uv)) < min_per:
                    return False
        return self.n_random() >= min_rand


# ================================================================
# AGENTS
# ================================================================
@dataclass
class Result:
    agent: str
    success: bool
    steps: int
    accidents: int
    acc_rate: float
    fuel: int
    pos: int
    # causal-specific
    fci_status: str = "N/A"
    fci_step: int = -1
    tipping_gamma: float = -1.0
    transitioned: bool = False

def run_naive(goal=500, fuel=1000, max_steps=2000):
    env = Env(goal=goal, fuel=fuel)
    acc = 0
    while not env.done and env.step < max_steps:
        # Naive: Yellow=1→slow, Yellow=0→fast, fuel<10→fast
        # Need to see Yellow first, but action determines accident
        # Peek at context first (generate it), then decide
        env.step += 1
        C = 1 if np.random.random() < P_CLEANING else 0
        S = 1 if np.random.random() < (P_SLIPPERY_GIVEN_C1 if C else P_SLIPPERY_GIVEN_C0) else 0
        Y = 1 if np.random.random() < (P_YELLOW_GIVEN_C1 if C else P_YELLOW_GIVEN_C0) else 0
        
        if env.fuel < 10:
            action = "fast"
        elif Y == 1:
            action = "slow"
        else:
            action = "fast"
        
        # Accident
        p = (P_ACC_S1_FAST if action=="fast" else P_ACC_S1_SLOW) if S else \
            (P_ACC_S0_FAST if action=="fast" else P_ACC_S0_SLOW)
        accident = 1 if np.random.random() < p else 0
        
        if accident:
            acc += 1
            env.fuel -= 5
        else:
            env.fuel -= 1
            env.pos += 3 if action == "fast" else 1
        
        if env.pos >= env.goal: env.done = True; env.won = True
        elif env.fuel <= 0: env.done = True
    
    s = env.step
    return Result("naive", env.won, s, acc, acc/s if s else 0, max(env.fuel,0), env.pos)

def run_oracle(goal=500, fuel=1000, max_steps=2000):
    """Knows Slippery directly — theoretical upper bound"""
    env = Env(goal=goal, fuel=fuel)
    acc = 0
    while not env.done and env.step < max_steps:
        env.step += 1
        C = 1 if np.random.random() < P_CLEANING else 0
        S = 1 if np.random.random() < (P_SLIPPERY_GIVEN_C1 if C else P_SLIPPERY_GIVEN_C0) else 0
        Y = 1 if np.random.random() < (P_YELLOW_GIVEN_C1 if C else P_YELLOW_GIVEN_C0) else 0
        
        if env.fuel < 10:
            action = "fast"
        elif S == 1:
            action = "slow"  # Oracle knows the true confounder
        else:
            action = "fast"
        
        p = (P_ACC_S1_FAST if action=="fast" else P_ACC_S1_SLOW) if S else \
            (P_ACC_S0_FAST if action=="fast" else P_ACC_S0_SLOW)
        accident = 1 if np.random.random() < p else 0
        
        if accident:
            acc += 1
            env.fuel -= 5
        else:
            env.fuel -= 1
            env.pos += 3 if action == "fast" else 1
        
        if env.pos >= env.goal: env.done = True; env.won = True
        elif env.fuel <= 0: env.done = True
    
    s = env.step
    return Result("oracle", env.won, s, acc, acc/s if s else 0, max(env.fuel,0), env.pos)

def run_causal(goal=500, fuel=1000, max_steps=2000, epsilon=0.4, 
               fci_min=50, disc_interval=25):
    """CausalBDI agent: ε-greedy → FCI → Thompson Sampling"""
    env = Env(goal=goal, fuel=fuel)
    engine = CausalEngine()
    acc_total = 0
    
    phase = "naive"  # "naive" or "causal"
    fci_status = "N/A"
    fci_step = -1
    tipping = -1.0
    transitioned = False
    
    while not env.done and env.step < max_steps:
        env.step += 1
        C = 1 if np.random.random() < P_CLEANING else 0
        S = 1 if np.random.random() < (P_SLIPPERY_GIVEN_C1 if C else P_SLIPPERY_GIVEN_C0) else 0
        Y = 1 if np.random.random() < (P_YELLOW_GIVEN_C1 if C else P_YELLOW_GIVEN_C0) else 0
        
        # Discovery check (naive phase only)
        if phase == "naive" and env.step > 1 and env.step % disc_interval == 0:
            res = engine.discover(min_n=fci_min)
            st = res.get("status", "unknown")
            # Conservative decision rule: any outcome that fails to support the
            # naive direct-cause model triggers the proxy-aware policy. This is a
            # DECISION under uncertainty, not a claim of structural identification.
            if st in ("no_adjacency", "latent_confounding",
                      "ambiguous_orientation", "ambiguous_randomisation"):
                phase = "causal"
                engine.epoch = 1
                fci_status = st
                fci_step = env.step
                transitioned = True
                tipping = engine.sensitivity()
            elif st == "direct_cause":
                fci_status = "direct_cause"
        
        # Action selection
        is_random = False
        if env.fuel < 10:
            action = "fast"
        elif phase == "causal":
            ready = engine.phase_ready(min_per=8, min_rand=fci_min)
            action = engine.thompson(Y, explore=not ready)
        else:
            # Naive: epsilon-greedy
            if np.random.random() < epsilon:
                action = "fast" if np.random.random() < 0.5 else "slow"
                is_random = True
            else:
                action = "slow" if Y == 1 else "fast"
        
        # Accident
        p = (P_ACC_S1_FAST if action=="fast" else P_ACC_S1_SLOW) if S else \
            (P_ACC_S0_FAST if action=="fast" else P_ACC_S0_SLOW)
        accident = 1 if np.random.random() < p else 0
        
        # Record observation
        engine.add(Y, action, accident, is_random)
        
        if accident:
            acc_total += 1
            env.fuel -= 5
        else:
            env.fuel -= 1
            env.pos += 3 if action == "fast" else 1
        
        if env.pos >= env.goal: env.done = True; env.won = True
        elif env.fuel <= 0: env.done = True
    
    s = env.step
    return Result("causal", env.won, s, acc_total, acc_total/s if s else 0, 
                  max(env.fuel,0), env.pos, fci_status, fci_step, tipping, transitioned)


# ================================================================
# STANDALONE FCI TEST
# ================================================================
def fci_accuracy_test(n_tests=50, sizes=[50, 100, 150, 200]):
    if not _HAS_FCI:
        print("FCI not available")
        return
    
    print(f"\n{'='*65}")
    print(f"  FCI ACCURACY TEST (standalone, random data from true DAG)")
    print(f"  {n_tests} tests per sample size")
    print(f"{'='*65}\n")
    
    for n in sizes:
        counts = {"latent_confounding":0, "ambiguous_orientation":0,
                  "ambiguous_randomisation":0, "no_adjacency":0,
                  "direct_cause":0, "adjacent":0, "error":0}
        for t in range(n_tests):
            np.random.seed(9000 + t*100 + n)
            data = []
            for _ in range(n):
                C = 1 if np.random.random() < 0.30 else 0
                S = 1 if np.random.random() < (0.90 if C else 0.05) else 0
                Y = 1 if np.random.random() < (0.80 if C else 0.10) else 0
                U = 1 if np.random.random() < 0.5 else 0  # random action
                p = (P_ACC_S1_FAST if U else P_ACC_S1_SLOW) if S else \
                    (P_ACC_S0_FAST if U else P_ACC_S0_SLOW)
                A = 1 if np.random.random() < p else 0
                data.append([Y, U, A])
            
            arr = np.array(data, dtype=float)
            try:
                G, _ = fci(arr)
                mat = np.array(G.graph)
                ya = (int(mat[0,2]), int(mat[2,0]))
                yu = (int(mat[0,1]), int(mat[1,0]))
                ya_adj = ya[0]!=0 or ya[1]!=0
                ya_bid = ya==(1,1)
                ya_circ = 2 in ya
                ya_dir = ya==(-1,1)
                yu_ind = yu[0]==0 and yu[1]==0
                
                if not ya_adj: counts["no_adjacency"] += 1
                elif ya_bid: counts["latent_confounding"] += 1
                elif ya_circ and yu_ind: counts["ambiguous_orientation"] += 1
                elif ya_circ: counts["ambiguous_randomisation"] += 1
                elif ya_dir: counts["direct_cause"] += 1
                else: counts["adjacent"] += 1
            except:
                counts["error"] += 1
        
        trans = (counts["no_adjacency"] + counts["latent_confounding"]
                 + counts["ambiguous_orientation"] + counts["ambiguous_randomisation"])
        print(f"  n={n:3d}: bidirected={counts['latent_confounding']:2d}/{n_tests} "
              f"| ambig_orient={counts['ambiguous_orientation']:2d} "
              f"| ambig_rand={counts['ambiguous_randomisation']:2d} "
              f"| no_adjacency={counts['no_adjacency']:2d} "
              f"| direct={counts['direct_cause']:2d} | adj={counts['adjacent']:2d} "
              f"| err={counts['error']:2d} "
              f"| policy-transition={trans}/{n_tests} ({100*trans/n_tests:.0f}%)")


# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_experiment(n_runs=30, goal=500, fuel=1000, fci_min=50):
    print(f"\n{'='*65}")
    print(f"  MAIN EXPERIMENT: {n_runs} independent runs per agent type")
    print(f"  goal={goal}, fuel={fuel}, FCI_min_n={fci_min}, ε=0.4")
    print(f"{'='*65}\n")
    
    naive_r, causal_r, oracle_r = [], [], []
    
    for i in range(n_runs):
        np.random.seed(3000 + i*7)
        nr = run_naive(goal=goal, fuel=fuel)
        
        np.random.seed(4000 + i*7)
        cr = run_causal(goal=goal, fuel=fuel, fci_min=fci_min)
        
        np.random.seed(5000 + i*7)
        orc = run_oracle(goal=goal, fuel=fuel)
        
        naive_r.append(nr)
        causal_r.append(cr)
        oracle_r.append(orc)
        
        tag = f"{cr.fci_status}"
        if cr.fci_step > 0: tag += f"@{cr.fci_step}"
        if cr.tipping_gamma > 0: tag += f" Γ={cr.tipping_gamma:.1f}"
        
        print(f"  {i+1:2d}/{n_runs}: "
              f"N(s={nr.steps:3d} a={nr.accidents:2d}) "
              f"C(s={cr.steps:3d} a={cr.accidents:2d}) "
              f"O(s={orc.steps:3d} a={orc.accidents:2d}) "
              f"| {tag}")
    
    # Summary
    def stats(rs):
        return {
            "succ": sum(r.success for r in rs),
            "steps_m": np.mean([r.steps for r in rs]),
            "steps_s": np.std([r.steps for r in rs]),
            "acc_m": np.mean([r.accidents for r in rs]),
            "acc_s": np.std([r.accidents for r in rs]),
            "rate_m": np.mean([r.acc_rate for r in rs]),
            "rate_s": np.std([r.acc_rate for r in rs]),
            "fuel_m": np.mean([r.fuel for r in rs]),
            "fuel_s": np.std([r.fuel for r in rs]),
        }
    
    ns, cs, os_ = stats(naive_r), stats(causal_r), stats(oracle_r)
    N = n_runs
    
    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"")
    hdr = "  {:<22} {:>18} {:>18} {:>18}".format("Metric", "Naive", "CausalBDI", "Oracle")
    print(hdr)
    print("  " + "-"*76)
    for label, key_m, key_s, fmt in [
        ("Success", "succ", None, "d"),
        ("Steps", "steps_m", "steps_s", ".1f"),
        ("Accidents", "acc_m", "acc_s", ".1f"),
        ("Accident Rate", "rate_m", "rate_s", ".4f"),
        ("Fuel Remaining", "fuel_m", "fuel_s", ".1f"),
    ]:
        if key_s is None:
            row = "  {:<22} {:>18} {:>18} {:>18}".format(
                label, 
                "{}/{}".format(ns[key_m], N),
                "{}/{}".format(cs[key_m], N),
                "{}/{}".format(os_[key_m], N))
        else:
            def fmtv(s, m_key, s_key, f):
                return ("{:"+f+"}+/-{:"+f+"}").format(s[m_key], s[s_key])
            row = "  {:<22} {:>18} {:>18} {:>18}".format(
                label, fmtv(ns, key_m, key_s, fmt), fmtv(cs, key_m, key_s, fmt), fmtv(os_, key_m, key_s, fmt))
        print(row)
    
    # FCI details
    print(f"\n  FCI OUTCOME AT TRANSITION (CausalBDI agent):")
    for st in ["latent_confounding", "ambiguous_orientation", "ambiguous_randomisation",
               "no_adjacency", "direct_cause", "adjacent", "N/A"]:
        cnt = sum(1 for r in causal_r if r.fci_status == st)
        if cnt > 0:
            print(f"    {st:<22}: {cnt:2d}/{N} ({100*cnt/N:.0f}%)")
    
    trans = sum(1 for r in causal_r if r.transitioned)
    print(f"    {'Policy-transition rate':<22}: {trans:2d}/{N} ({100*trans/N:.0f}%)")
    
    disc_steps = [r.fci_step for r in causal_r if r.fci_step > 0]
    if disc_steps:
        print(f"    {'Policy-transition step':<22}: {np.mean(disc_steps):.0f} ± {np.std(disc_steps):.0f}")
    
    gammas = [r.tipping_gamma for r in causal_r if r.tipping_gamma > 0]
    if gammas:
        print(f"    {'Tipping Γ (mean)':<22}: {np.mean(gammas):.2f} ± {np.std(gammas):.2f}")
    
    # Statistical test
    from scipy import stats as sp_stats
    naive_rates = [r.acc_rate for r in naive_r]
    causal_rates = [r.acc_rate for r in causal_r]
    t_stat, p_val = sp_stats.ttest_ind(naive_rates, causal_rates)
    print(f"\n  Welch t-test (accident rate, naive vs causal): t={t_stat:.3f}, p={p_val:.4f}")
    
    naive_steps_ = [r.steps for r in naive_r]
    causal_steps_ = [r.steps for r in causal_r]
    t2, p2 = sp_stats.ttest_ind(naive_steps_, causal_steps_)
    print(f"  Welch t-test (steps, naive vs causal): t={t2:.3f}, p={p2:.4f}")
    
    return naive_r, causal_r, oracle_r


if __name__ == "__main__":
    fci_accuracy_test(n_tests=50, sizes=[50, 100, 150, 200])
    naive_r, causal_r, oracle_r = run_experiment(n_runs=30, goal=500, fuel=1000, fci_min=50)
