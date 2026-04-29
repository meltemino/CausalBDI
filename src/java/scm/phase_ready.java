package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.phase_ready(Ready)
 * Calls /phase_ready and unifies Ready with true/false.
 */
public class phase_ready extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            String res = ScmClient.postJson("/phase_ready", "{}");
            boolean ready = ScmClient.getBoolean(res, "ready");
            return un.unifies(args[0], new Atom(ready ? "true" : "false"));
        } catch (Exception e) {
            ts.getLogger().warning("phase_ready soft-fail: " + e.getMessage());
            return un.unifies(args[0], new Atom("false"));
        }
    }
}
