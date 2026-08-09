from __future__ import annotations

import json

from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend


class CryptoFactorExperiment2Feedback(Experiment2Feedback):
    """LLM explains results; deterministic four-metric policy owns the decision."""

    def generate_feedback(self, exp, trace: Trace) -> HypothesisFeedback:
        accepted = list(getattr(exp, "accepted_factor_names", []))
        rejected = getattr(exp, "rejected_factor_details", [])
        baseline_metrics = getattr(exp, "crypto_baseline_metrics", {})
        current_metrics = getattr(exp, "crypto_current_metrics", {})
        runtime_errors = getattr(exp, "crypto_runtime_errors", [])
        deterministic_decision = bool(accepted)
        system_prompt = (
            self.scen.get_scenario_all_desc()
            + "\n\nExplain the gate metrics and baseline/current Qlib results, then propose the next hypothesis. "
            "You may not change acceptance: only the deterministic strict PFS/RRE/Best_IC/Best_IR gates plus "
            "execution and correlation de-duplication decide which factors enter the library."
        )
        user_prompt = json.dumps(
            {
                "hypothesis": str(exp.hypothesis),
                "candidate_metrics": getattr(exp, "crypto_candidate_metrics", []).to_dict("records"),
                "accepted": accepted,
                "rejected": rejected,
                "qlib_baseline": baseline_metrics,
                "qlib_current": current_metrics,
                "non_gating_runtime_errors": runtime_errors,
            },
            ensure_ascii=False,
            default=str,
        )
        response_data = {}
        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=dict[str, str],
            )
            response_data = json.loads(response)
        except Exception as exc:
            logger.warning(f"Crypto feedback LLM failed; deterministic decision is preserved: {exc}")
        reason = response_data.get(
            "Reasoning",
            f"Accepted factors: {accepted or 'none'}. Rejected details: {rejected}. "
            f"Qlib errors are non-gating: {runtime_errors}",
        )
        return HypothesisFeedback(
            observations=response_data.get("Observations", json.dumps(current_metrics, default=str)),
            hypothesis_evaluation=response_data.get("Feedback for Hypothesis", reason),
            new_hypothesis=response_data.get(
                "New Hypothesis", "Use the rejection reasons to propose a distinct no-lookahead factor."
            ),
            reason=reason,
            decision=deterministic_decision,
            acceptable=deterministic_decision,
        )
