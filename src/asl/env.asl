/* env.asl — Latent Variable Environment with Race Condition Fix
 *
 * DAG (Data Generating Process):
 * Cleaning -> Slippery (latent confounder)
 * Cleaning -> Yellow   (observed proxy)
 * Slippery x Action -> Accident
 */

// --- Initial state (matches the paper's evaluation setup) ---
pos(0).
goal(500).
fuel(1000).
yellow(0).
accident(0).
done(0).
won(0).
env_step(0).

// Latent variables (the agent cannot observe these directly)
cleaning(0).
slippery(0).

// === Start the simulation ===
+!start <-
    .print("====================================");
    .print("[ENV] Simulation started");
    .print("[ENV] DAG: Cleaning -> {Yellow, Slippery}");
    .print("[ENV]      Slippery x Action -> Accident");
    .print("[ENV] Goal: 500 units, Fuel: 1000");
    .print("====================================");
    !game_loop.

// === Game loop ===
+!game_loop : done(0) <-
    .wait(500);
    !generate_context;
    !emit_percepts;
    !game_loop.

+!game_loop : done(1) <-
    ?won(W);
    ?pos(P);
    ?fuel(F);
    ?env_step(S);
    .print("====================================");
    .print("[ENV] Simulation completed.");
    if (W == 1) {
        .print("[ENV] RESULT: SUCCESS (pos=", P, " fuel=", F, " steps=", S, ")");
    } else {
        .print("[ENV] RESULT: FAILURE (pos=", P, " fuel=", F, " steps=", S, ")");
    };
    .print("====================================").

// === Generative Model (Causal DAG) ===
+!generate_context <-
    ?env_step(S);
    -+env_step(S + 1);

    // 1. Cleaning (root cause - exogenous)
    // P(Cleaning = 1) = 0.30
    .random(R1);
    if (R1 < 0.30) { -+cleaning(1); } else { -+cleaning(0); };

    // 2. Slippery (latent confounder)
    // P(Slippery=1 | Cleaning=1) = 0.90
    // P(Slippery=1 | Cleaning=0) = 0.05
    ?cleaning(C);
    .random(R2);
    if (C == 1) {
        if (R2 < 0.90) { -+slippery(1); } else { -+slippery(0); };
    } else {
        if (R2 < 0.05) { -+slippery(1); } else { -+slippery(0); };
    };

    // 3. Yellow Light (observed proxy for Cleaning)
    // P(Yellow=1 | Cleaning=1) = 0.80
    // P(Yellow=1 | Cleaning=0) = 0.10
    .random(R3);
    if (C == 1) {
        if (R3 < 0.80) { -+yellow(1); } else { -+yellow(0); };
    } else {
        if (R3 < 0.10) { -+yellow(1); } else { -+yellow(0); };
    }.

// === Action outcomes (structural equation for Accident) ===
// P(Accident | Slippery, Action)
+!do(Action) : fuel(F) & pos(P) & goal(G) & slippery(S) <-
    .random(Ra);
    if (S == 1 & Action == fast & Ra < 0.70) {
        !apply(Action, accident);
    } else {
        if (S == 1 & Action == slow & Ra < 0.10) {
            !apply(Action, accident);
        } else {
            if (S == 0 & Action == fast & Ra < 0.01) {
                !apply(Action, accident);
            } else {
                if (S == 0 & Action == slow & Ra < 0.005) {
                    !apply(Action, accident);
                } else {
                    !apply(Action, safe);
                }
            }
        }
    }.

// === Apply accident outcome ===
+!apply(Action, accident) : fuel(F) & pos(P) & goal(G) <-
    FuelCost = 5;
    NewP = P;
    NewF = F - FuelCost;

    .print("[ENV] *** ACCIDENT! *** Action=", Action, " FuelCost=", FuelCost);
    -+accident(1);
    -+pos(NewP);
    -+fuel(NewF);
    if (NewF <= 0) {
        -+done(1); -+won(0);
        .print("[ENV] OUT OF FUEL. Failure.");
    } else {
        if (NewP >= G) {
            -+done(1);
            -+won(1);
            .print("[ENV] GOAL REACHED. Success.");
        }
    }.

// === Apply safe-driving outcome ===
+!apply(Action, safe) : fuel(F) & pos(P) & goal(G) <-
    FuelCost = 1;
    if (Action == fast) { Dist = 3; } else { Dist = 1; };

    -+accident(0);
    NewP = P + Dist;
    NewF = F - FuelCost;

    -+pos(NewP);
    -+fuel(NewF);
    if (NewF <= 0) {
        -+done(1); -+won(0);
        .print("[ENV] OUT OF FUEL. Failure.");
    } else {
        if (NewP >= G) {
            -+done(1);
            -+won(1);
            .print("[ENV] GOAL REACHED. Success.");
        }
    }.

// === Percept emission ===
+!emit_percepts : yellow(Y) & accident(A) & pos(P) & fuel(F) & done(D) <-
    .send(a1, untell, yellow(_));
    .send(a1, untell, accident(_));
    .send(a1, untell, pos(_));
    .send(a1, untell, fuel(_));
    .send(a1, untell, done(_));
    .wait(50);
    .send(a1, tell, yellow(Y));
    .send(a1, tell, accident(A));
    .send(a1, tell, pos(P));
    .send(a1, tell, fuel(F));
    .send(a1, tell, done(D)).
