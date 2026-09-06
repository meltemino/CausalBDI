# CausalBDI

Reference implementation accompanying the paper:

> **From Implicit to Explicit Causal Models in BDI Agents:
> A Proof-of-Concept with Causal Discovery and Adaptive Policy.**
> Meltem Gullusac, Baris Tekin Tezel, Moharram Challenger.
> *CLaRAMAS 2026 (1st Int. Workshop on Causal Learning and Reasoning in
> Agents and Multiagent Systems).*

CausalBDI augments a Jason/AgentSpeak BDI agent with an explicit
Structural Causal Model (SCM) maintained by an external causal inference
server. The agent progresses through a four-phase epistemic lifecycle:

1. **Naive exploration** — epsilon-greedy random actions to collect
   interventional data.
2. **Causal discovery** — Fast Causal Inference (FCI) on randomized
   observations, to test whether the observed proxy is a direct cause of
   the outcome or is confounded with it by a latent variable.
3. **Epistemic transition** — beliefs shift from correlational to causal
   when discovery fails to support the naive direct-cause reading.
4. **Causal policy** — Thompson Sampling with proxy-stratified
   posteriors, calibrated by Rosenbaum sensitivity analysis.

## Discovery statuses

`scm.run_discovery(Status)` returns one of the following, following the
classification in Sect. 5.2 of the paper. Marks are the PAG endpoint marks of
the Yellow–Accident edge in causal-learn's encoding (-1 tail, 1 arrowhead,
2 circle, 0 no edge):

| Marks | Status | Reading |
|---|---|---|
| (1,1) | `latent_confounding` | bidirected edge; latent common cause supported by the PAG alone |
| (2,1) / (2,2) | `ambiguous_orientation` | circle endpoint; `Y -> A` and `Y <-> A` are not distinguishable with {Y, U, A} |
| (0,0) | `no_adjacency` | no *detected* adjacency; **not** evidence of a latent common cause |
| (-1,1) | `direct_cause` | directed edge; naive model supported, no transition |
| circle + `Y–U` edge | `ambiguous_randomisation` | randomisation diagnostic failed |

All statuses except `direct_cause` trigger the conservative policy transition.
This is a decision under uncertainty, not a claim of structural identification.

## Repository layout

```
CausalBDI/
├── scm_server.py              # Python HTTP server: FCI, estimation, TS
├── causalbdi_experiment.py    # Standalone simulation (Section 7 evaluation)
├── requirements.txt           # Python dependencies
├── build.gradle               # Gradle build for the JaCaMo project
├── claramas.jcm               # JaCaMo project descriptor
├── src/
│   ├── asl/
│   │   ├── a1.asl             # Cognitive (BDI) agent
│   │   └── env.asl            # Environment agent (data-generating DAG)
│   └── java/scm/              # Custom Jason internal actions
│       ├── ScmClient.java
│       ├── record_observation.java
│       ├── run_discovery.java
│       ├── test_intervention.java
│       ├── run_sensitivity.java
│       ├── get_optimal_action.java
│       ├── phase_ready.java
│       ├── set_epoch.java
│       └── reset_data.java
└── README.md
```

## Requirements

### Python (server + standalone evaluation)

- Python 3.9+
- See `requirements.txt`:
  - `causal-learn` (for FCI)
  - `numpy`
  - `scikit-learn` (optional)

Install with:

```bash
pip install -r requirements.txt
```

### JaCaMo / Jason (agent-based demo)

- Java 11+
- Gradle 7+
- Internet access for first build (downloads JaCaMo 1.3.0 via Maven)

## Running the standalone evaluation

The standalone script reproduces the quantitative results in Section 7 of
the paper (Table 2) without needing the JaCaMo runtime:

```bash
python causalbdi_experiment.py
```

This runs 30 independent simulation runs for each of the three agent
types (Naive, CausalBDI, Oracle) with the parameters reported in the
paper:

| Parameter         | Value |
|-------------------|-------|
| Goal distance     | 500   |
| Initial fuel      | 1000  |
| ε (epsilon-greedy)| 0.4   |
| FCI min n_random  | 50    |
| Discovery interval| 25    |
| Runs per agent    | 30    |

## Running the JaCaMo agent demo

Start the SCM server first:

```bash
python scm_server.py
# SCM Server v2 (Causal BDI) — http://127.0.0.1:8008
```

In a second terminal, build and launch the BDI agents:

```bash
./gradlew run
```

The cognitive agent (`a1.asl`) communicates with the environment agent
(`env.asl`) and queries the SCM server for causal discovery, sensitivity
analysis, and Thompson Sampling action selection.

## Causal model

The data-generating DAG (Figure 2 in the paper):

```
            Cleaning  (latent root cause)
            /    \
       Slippery   Yellow   (Slippery latent, Yellow observed)
       (latent)
           \
            Accident  <-  Action (treatment)
```

The agent observes only `Yellow`, `Action`, and `Accident`; `Slippery`
and `Cleaning` are unobserved. `Yellow` serves as a proxy for the latent
confounding pathway.

Conditional probabilities (`scm_server.py` mirrors these in Python; the
Jason `env.asl` agent samples them at runtime):

| Quantity                        | Value |
|---------------------------------|-------|
| P(Cleaning = 1)                 | 0.30  |
| P(Slippery = 1 \| Cleaning = 1) | 0.90  |
| P(Slippery = 1 \| Cleaning = 0) | 0.05  |
| P(Yellow = 1 \| Cleaning = 1)   | 0.80  |
| P(Yellow = 1 \| Cleaning = 0)   | 0.10  |
| P(Accident \| S=1, fast)        | 0.70  |
| P(Accident \| S=1, slow)        | 0.10  |
| P(Accident \| S=0, fast)        | 0.01  |
| P(Accident \| S=0, slow)        | 0.005 |

## SCM server endpoints

The server exposes the following HTTP/JSON endpoints on port 8008:

| Endpoint                | Method | Purpose                                     |
|-------------------------|--------|---------------------------------------------|
| `/health`               | GET    | Liveness check; reports library availability|
| `/record_observation`   | POST   | Append (Y, U, A, is_random) to history      |
| `/discover`             | POST   | Run FCI on randomized data only             |
| `/ace`                  | POST   | Proxy-stratified causal effect estimation   |
| `/sensitivity`          | POST   | Rosenbaum sensitivity bounds                |
| `/phase_ready`          | POST   | Check overlap and randomization sufficiency |
| `/optimize_policy`      | POST   | Thompson Sampling action selection          |
| `/set_epoch`            | POST   | Switch naive (0) / causal (1) epoch         |
| `/reset`                | POST   | Clear all observation history               |

## Reproducing Table 2

```bash
python causalbdi_experiment.py
```

Expected output (with default parameters, ±1 standard deviation across
30 independent runs):

| Metric          | Naive          | CausalBDI       | Oracle          |
|-----------------|----------------|-----------------|-----------------|
| Success rate    | 30/30          | 30/30           | 30/30           |
| Steps to goal   | 233 ± 7        | 349 ± 58        | 214 ± 6         |
| Accident rate   | 0.091 ± 0.016  | 0.068 ± 0.016   | 0.037 ± 0.012   |
| Fuel remaining  | 682 ± 23       | 559 ± 64        | 754 ± 15        |
| Policy-transition rate | —       | 30/30 (100 %)   | —               |
| Policy-transition step | —       | 139 ± 18        | —               |

Welch's t-test for accident rate (Naive vs CausalBDI): t = 5.31, p < 0.001.

The transition rate reports how often the FCI-guided decision rule activated
the causal policy, not how often a latent common cause was structurally
identified. At the transition point the script also reports the distribution
of PAG outcomes; with the default parameters this is `ambiguous_orientation`
in 16 runs and `no_adjacency` in 14 runs.

## Repository

This code is hosted at: <https://github.com/meltemino/CausalBDI>

## Citation

The paper has been accepted at the **1st International Workshop on
Causal Learning and Reasoning in Agents and Multiagent Systems
(CLaRAMAS 2026)** and is forthcoming in the Springer Communications in
Computer and Information Science (CCIS) series.

If you use this code, please cite the paper:

```bibtex
@inproceedings{gullusac2026causalbdi,
  title     = {From Implicit to Explicit Causal Models in {BDI} Agents:
               A Proof-of-Concept with Causal Discovery and Adaptive Policy},
  author    = {Gullusac, Meltem and Tezel, Baris Tekin and Challenger, Moharram},
  booktitle = {Proceedings of the 1st International Workshop on Causal
               Learning and Reasoning in Agents and Multiagent Systems
               (CLaRAMAS 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

> **Note:** This entry will be updated with the final bibliographic
> details (volume, pages, publisher, DOI) once the proceedings are
> published. Check this README for the latest version, or contact the
> authors directly.

## License

This project is released under the MIT License — see `LICENSE` for
details.

## Contact

For questions about the implementation, please open an issue on GitHub
or contact the authors:

- Meltem Gullusac — `meltem.gullusac@deu.edu.tr`
- Baris Tekin Tezel — `baris.tezel@deu.edu.tr`
- Moharram Challenger — `moharram.challenger@uantwerpen.be`
