package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.run_discovery(StatusCode)
 * Calls /discover and returns status_code as Atom (proxy/adjacent/insufficient/...)
 */
public class run_discovery extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            if (args.length < 1) {
                ts.getLogger().warning("scm.run_discovery/1 called with no arguments.");
                return false;
            }

            String res = ScmClient.postJson("/discover", "{}");
            String statusCode = ScmClient.getString(res, "status_code");
            if (statusCode == null || statusCode.isEmpty()) statusCode = "unknown";
            statusCode = statusCode.toLowerCase();

            String statusText = ScmClient.getString(res, "status_text");

            ts.getLogger().info("--- Causal Discovery Result ---");
            if (statusText != null && !statusText.isEmpty()) ts.getLogger().info(statusText);
            else ts.getLogger().info(res);

            return un.unifies(args[0], new Atom(statusCode));

        } catch (Exception e) {
            ts.getLogger().severe("Error in run_discovery: " + e.getMessage());
            return false;
        }
    }
}
