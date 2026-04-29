package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.set_epoch(EpochNumber)
 * Signals the SCM server that the agent has transitioned to a new epistemic phase.
 *   epoch=0 → naive (correlational beliefs)
 *   epoch=1 → causal (post-discovery, proxy-aware)
 *
 * The server uses this to separate observations by epoch for
 * time-varying estimation (avoids mixing naive-policy data with causal-policy data).
 */
public class set_epoch extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            int epoch = (int) ((NumberTerm) args[0]).solve();
            String payload = "{\"epoch\":" + epoch + "}";
            ScmClient.postJson("/set_epoch", payload);
            ts.getLogger().info("[scm.set_epoch] Epoch set to " + epoch);
            return true;
        } catch (Exception e) {
            ts.getLogger().warning("set_epoch soft-fail: " + e.getMessage());
            return true;
        }
    }
}