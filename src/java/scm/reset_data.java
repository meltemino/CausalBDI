package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.reset_data()
 * Calls /reset.
 */
public class reset_data extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            ScmClient.postJson("/reset", "{}");
            return true;
        } catch (Exception e) {
            ts.getLogger().warning("reset_data soft-fail: " + e.getMessage());
            return true;
        }
    }
}
