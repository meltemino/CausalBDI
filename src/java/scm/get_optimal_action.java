package scm;

import jason.asSemantics.*;
import jason.asSyntax.*;

/**
 * scm.get_optimal_action(Yellow, [Explore], BestAction)
 * Calls /optimize_policy.
 */
public class get_optimal_action extends DefaultInternalAction {
    @Override
    public Object execute(TransitionSystem ts, Unifier un, Term[] args) throws Exception {
        try {
            int y = (int) ((NumberTerm) args[0]).solve();

            boolean explore = true;
            int bestIndex;

            if (args.length == 2) {
                bestIndex = 1;
            } else if (args.length == 3) {
                bestIndex = 2;
                explore = args[1].toString().equals("true");
            } else {
                ts.getLogger().severe("get_optimal_action: expected 2 or 3 args, got " + args.length);
                return false;
            }

            String payload = "{\"yellow\":" + y + ",\"explore\":" + (explore ? "true" : "false") + "}";
            String res = ScmClient.postJson("/optimize_policy", payload);

            String best = ScmClient.getString(res, "best_action");
            if (best == null || best.isEmpty()) best = "slow";

            return un.unifies(args[bestIndex], new Atom(best));

        } catch (Exception e) {
            ts.getLogger().severe("Error in get_optimal_action: " + e.getMessage());
            return false;
        }
    }
}
