from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.scenarios.qlib.crypto_factor.runner import CryptoFactorRunner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def test_multiple_candidates_only_accepted_factor_reaches_next_round():
    accepted = FactorTask("Accepted", "passes", "x")
    rejected = FactorTask("Rejected", "fails", "y")
    experiment = QlibFactorExperiment([accepted, rejected])
    experiment.sub_workspace_list = [object(), object()]
    experiment.rejected_factor_details = [{"factor": "Rejected", "reasons": ["PFS boundary"]}]

    CryptoFactorRunner._filter_experiment(experiment, {"Accepted"})

    assert [task.factor_name for task in experiment.sub_tasks] == ["Accepted"]
    assert len(experiment.sub_workspace_list) == 1
    assert experiment.rejected_factor_details[0]["factor"] == "Rejected"
