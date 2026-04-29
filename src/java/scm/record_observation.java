package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.record_observation(Y, ActionAtom, Accident, IsRandom)
 *
 * v2: Added 4th argument IsRandom (true/false atom).
 * The server uses this flag to separate randomized data (valid for FCI)
 * from policy-driven data (potentially confounded by agent's policy).
 *
 * Sends: {"yellow":Y, "action":"fast|slow", "accident":Acc, "is_random":bool}
 */
public class record_observation extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            int y = (int) ((NumberTerm) args[0]).solve();

            // Parse action (Atom or String)
            String action;
            if (args[1] instanceof Atom) {
                action = ((Atom) args[1]).getFunctor();
            } else {
                action = args[1].toString();
                if (action.startsWith("\"") && action.endsWith("\"") && action.length() >= 2) {
                    action = action.substring(1, action.length() - 1);
                }
            }

            int a = (int) ((NumberTerm) args[2]).solve();

            // v2: Parse is_random flag (4th argument, optional)
            boolean isRandom = false;
            if (args.length >= 4) {
                String randomStr = args[3].toString().toLowerCase().trim();
                isRandom = randomStr.equals("true") || randomStr.equals("1");
            }

            String payload = String.format(
                "{\"yellow\":%d,\"action\":\"%s\",\"accident\":%d,\"is_random\":%s}",
                y, action, a, isRandom ? "true" : "false"
            );

            ScmClient.postJson("/record_observation", payload);
            return true;

        } catch (Exception e) {
            ts.getLogger().warning("record_observation soft-fail: " + e.getMessage());
            return true;
        }
    }
}