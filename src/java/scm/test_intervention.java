package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.test_intervention(RiskFast, RiskSlow)
 * Calls /ace and returns risk_fast, risk_slow.
 * If unstable -> returns -1,-1.
 */
public class test_intervention extends DefaultInternalAction {

    private boolean parseBoolean(String json, String key, boolean defaultVal) {
        String k = "\"" + key + "\"";
        int i = json.indexOf(k);
        if (i < 0) return defaultVal;
        int c = json.indexOf(':', i);
        if (c < 0) return defaultVal;

        String tail = json.substring(c + 1).trim();
        if (tail.startsWith("true")) return true;
        if (tail.startsWith("false")) return false;
        return defaultVal;
    }

    private boolean hasKey(String json, String key) {
        return json.contains("\"" + key + "\"");
    }

    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        double riskFast = -1.0;
        double riskSlow = -1.0;

        try {
            String res = ScmClient.postJson("/ace", "{}");

            if (hasKey(res, "risk_fast") || hasKey(res, "risk_slow")) {
                riskFast = ScmClient.getNumber(res, "risk_fast");
                riskSlow = ScmClient.getNumber(res, "risk_slow");
            } else {
                riskFast = ScmClient.getNumber(res, "ace_fast");
                riskSlow = ScmClient.getNumber(res, "ace_slow");
            }

            boolean stable = parseBoolean(res, "stable", true);
            if (!stable) {
                double nFast = ScmClient.getNumber(res, "n_fast");
                double nSlow = ScmClient.getNumber(res, "n_slow");
                ts.getLogger().info("[scm.test_intervention] UNSTABLE (n_fast=" + nFast + ", n_slow=" + nSlow + "). Returning -1,-1.");
                riskFast = -1.0;
                riskSlow = -1.0;
            }

        } catch (Exception e) {
            ts.getLogger().warning("test_intervention soft-fail: " + e.getMessage());
            riskFast = -1.0;
            riskSlow = -1.0;
        }

        return un.unifies(args[0], new NumberTermImpl(riskFast)) &&
               un.unifies(args[1], new NumberTermImpl(riskSlow));
    }
}
