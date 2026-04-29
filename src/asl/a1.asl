/* a1.asl — Causally-Sound BDI Agent
 *
 * Key design decisions:
 * 1. Epsilon-greedy exploration during NAIVE phase -> randomized actions for valid FCI
 * 2. Per-observation is_random flag tracking -> server-side filtering for discovery
 * 3. Epoch signal for time-varying estimates (naive=0, causal=1)
 * 4. Sample-size threshold for discovery (n_random >= 50)
 * 5. Post-discovery: sensitivity analysis informs belief confidence
 * 6. Thompson Sampling in causal phase (delegated to server)
 */

/* === Beliefs === */
// Sensor data (initial values; environment overwrites these via KQML)
pos(0).
fuel(1000). yellow(0). accident(0). done(0).

// Epistemic state
belief_structure(naive).            // naive -> causal_confounding
belief_latent_danger(0.0).          // P(LatentDanger | Yellow)
belief_confidence(low).             // low / medium / high (from sensitivity analysis)
step_count(0).
last_action(none).
last_action_random(false).          // was the last action randomized?

// Exploration parameters
epsilon(0.4).                       // 40% random actions during naive phase
discovery_interval(25).             // attempt discovery every 25 steps
min_discovery_steps(50).            // FCI requires at least 50 random observations

/* === Start === */
!start.
+!start <-
    scm.reset_data;
    scm.set_epoch(0);
    .print("=== AGENT ACTIVE ===");
    .print("[INIT] Mental model: NAIVE (correlational)");
    .print("[INIT] Epsilon-greedy active (eps=0.4). Collecting randomized data...").

/* === Perception trigger === */
+yellow(Y) : done(0) <-
    // 1) Record the data from the previous action
    ?last_action(Action);
    ?accident(Acc);
    ?last_action_random(WasRandom);

    if (Action \== none) {
        scm.record_observation(Y, Action, Acc, WasRandom);
        !check_discovery;
    };
    // 2) Latent state inference
    !infer_latent_state(Y);

    // 3) Action selection
    !choose_action(Y).

/* === INFERENCE (epistemic update) === */

// Naive: yellow light = danger (correlation trap — agent does not know this yet)
+!infer_latent_state(Y) : belief_structure(naive) <-
    if (Y == 1) { -+belief_latent_danger(0.9);
    }
    else        { -+belief_latent_danger(0.1); }.

// Causal: Yellow is just a proxy;
// posterior is more cautious due to the latent confounder.
// Confidence level fine-tunes the estimate.
+!infer_latent_state(Y) : belief_structure(causal_confounding) <-
    ?belief_confidence(Conf);
    if (Y == 1) {
        // Yellow=1: cleaning likely -> slippery floor likely but not certain
        if (Conf == high)   { Prob = 0.55;
        }   // sharper estimate with sufficient data
        else { if (Conf == medium) { Prob = 0.60;
        }
        else                { Prob = 0.65;
        } }  // wider uncertainty with limited data
    } else {
        // Yellow=0: low danger but not zero
        if (Conf == high)   { Prob = 0.05;
        }
        else { if (Conf == medium) { Prob = 0.08;
        }
        else                { Prob = 0.12;
        } }
    };

    -+belief_latent_danger(Prob);
    .print("[MIND] P(Latent|Y=", Y, ")=", Prob, " [confidence=", Conf, "]").

/* === ACTION SELECTION === */

+!choose_action(Ynow) : done(0) <-
    ?belief_latent_danger(P_Danger);
    ?fuel(F);
    // Bypass when fuel is critical
    if (F < 10) {
        .print("[DECIDE] CRITICAL FUEL! Forcing FAST.");
        -+last_action_random(false);
        !act(fast);
    } else {
        ?belief_structure(S);
        if (S == causal_confounding) {
            // === CAUSAL PHASE: Thompson Sampling (server-side) ===
            scm.phase_ready(Ready);
            if (Ready == true) { Explore = false; } else { Explore = true; };

            scm.get_optimal_action(Ynow, Explore, Best);
            .print("[DECIDE] (Thompson) Y=", Ynow, " Ready=", Ready, " Best=", Best);

            -+last_action_random(false);
            // TS choices are NOT marked as random for FCI purposes
            !act(Best);
        } else {
            // === NAIVE PHASE: Epsilon-Greedy Exploration ===
            ?epsilon(Eps);
            .random(Roll);

            if (Roll < Eps) {
                // RANDOM ACTION (critical for FCI input)
                .random(CoinFlip);
                if (CoinFlip < 0.5) { RandAction = fast; } else { RandAction = slow; };
                .print("[DECIDE] RANDOM action: ", RandAction, " (eps-greedy, roll=", Roll, ")");
                -+last_action_random(true);
                !act(RandAction);
            } else {
                // Naive decision: based on correlational belief
                Threshold = 0.5;
                if (P_Danger > Threshold) {
                    .print("[DECIDE] Naive: High danger (", P_Danger, "). SLOW.");
                    -+last_action_random(false);
                    !act(slow);
                } else {
                    .print("[DECIDE] Naive: Low danger (", P_Danger, "). FAST.");
                    -+last_action_random(false);
                    !act(fast);
                }
            }
        }
    }.

/* === Execute action === */
+!act(A) <-
    -+last_action(A);
    .send(env, achieve, do(A)).

/* === META-COGNITION: Causal Discovery === */

+!check_discovery : step_count(N) & belief_structure(naive) <-
    NewN = N + 1;
    -+step_count(NewN);

    ?discovery_interval(Interval);

    if ((NewN mod Interval) == 0) {
        .print("=== [RESEARCH] Causal discovery attempt (step=", NewN, ") ===");
        scm.run_discovery(Status);
        .print("[RESEARCH] status_code = ", Status);

        if (Status == proxy) {
            .print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            .print("!!! DISCOVERY: Yellow = PROXY            !!!");
            .print("!!! Latent confounder detected           !!!");
            .print("!!! Updating mental model...             !!!");
            .print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            // Epistemic transition: naive -> causal
            -+belief_structure(causal_confounding);
            scm.set_epoch(1);
            // Determine confidence via sensitivity analysis
            !assess_confidence;
        };
        if (Status == ambiguous_latent) {
            .print("[RESEARCH] Ambiguous latent structure detected.");
            .print("[RESEARCH] Circle mark suggests possible latent confounder.");
            .print("[RESEARCH] Transitioning to causal phase with caution.");
            -+belief_structure(causal_confounding);
            scm.set_epoch(1);
            !assess_confidence;
        };
        if (Status == direct_cause) {
            .print("[RESEARCH] FCI: Yellow -> Accident (direct cause).");
            .print("[RESEARCH] Naive model remains valid.");
        };
        if (Status == insufficient) {
            .print("[RESEARCH] Insufficient random data. Exploration continues.");
        };
        if (Status == adjacent) {
            .print("[RESEARCH] Yellow-Accident adjacency found. Naive model still valid.");
        }
    }.

// Skip discovery while already in causal phase (fallback)
+!check_discovery.

/* === Sensitivity-based confidence assessment === */
+!assess_confidence <-
    scm.run_sensitivity(TippingGamma);
    if (TippingGamma > 3.0) {
        -+belief_confidence(high);
        .print("[CONFIDENCE] High: Gamma*=", TippingGamma, " (strong robustness)");
    } else {
        if (TippingGamma > 1.5) {
            -+belief_confidence(medium);
            .print("[CONFIDENCE] Medium: Gamma*=", TippingGamma, " (moderate robustness)");
        } else {
            -+belief_confidence(low);
            .print("[CONFIDENCE] Low: Gamma*=", TippingGamma, " (weak robustness, caution!)");
        }
    }.
