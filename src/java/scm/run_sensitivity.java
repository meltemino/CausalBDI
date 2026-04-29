package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.run_sensitivity(TippingGamma)
 *
 * Calls /sensitivity endpoint and returns the tipping point Γ*.
 * Γ* indicates how much unmeasured confounding would be needed to
 * overturn the causal conclusion.
 *
 * Returns:
 *   TippingGamma > 3.0  → high confidence (robust to strong confounding)
 *   TippingGamma > 1.5  → medium confidence
 *   TippingGamma ≤ 1.5  → low confidence (fragile result)
 *   TippingGamma = -1.0 → insufficient data or error
 */
public class run_sensitivity extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            String res = ScmClient.postJson("/sensitivity", "{}");

            // Extract tipping_gamma
            double tippingGamma = ScmClient.getNumber(res, "tipping_gamma");

            // If tipping_gamma is 0.0 (not found / null in JSON), it means
            // all tested Gamma values were robust → report high value
            boolean stable = ScmClient.getBoolean(res, "stable");
            if (!stable) {
                tippingGamma = -1.0;
            } else if (tippingGamma == 0.0) {
                // null in JSON → no tipping point found → very robust
                // Check if the key exists at all
                if (!res.contains("\"tipping_gamma\"") || res.contains("\"tipping_gamma\":null")
                    || res.contains("\"tipping_gamma\": null")) {
                    tippingGamma = 99.0;  // sentinel: extremely robust
                }
            }

            String interpretation = ScmClient.getString(res, "interpretation");
            if (interpretation != null && !interpretation.isEmpty()) {
                ts.getLogger().info("[SENSITIVITY] " + interpretation);
            }

            ts.getLogger().info("[SENSITIVITY] Tipping Gamma = " + tippingGamma);

            return un.unifies(args[0], new NumberTermImpl(tippingGamma));

        } catch (Exception e) {
            ts.getLogger().warning("run_sensitivity soft-fail: " + e.getMessage());
            return un.unifies(args[0], new NumberTermImpl(-1.0));
        }
    }
}