from pathlib import Path

import pandas as pd

from rdagent.scenarios.qlib.crypto_factor.config import CryptoFactorPropSetting
from rdagent.scenarios.qlib.crypto_factor.evaluation_adapter import EvaluationAdapter
from rdagent.scenarios.qlib.crypto_factor.qlib_analysis import build_backtest_config
from rdagent.scenarios.qlib.crypto_factor.store import AcceptedFactorStore


def test_candidate_metrics_gate_registration_and_qlib_config(tmp_path):
    data_path = tmp_path / "candles.pkl"
    pd.to_pickle([pd.DataFrame({"unused": [1]})], data_path)
    factors = tmp_path / "factors"
    factors.mkdir()
    (factors / "__init__.py").write_text("")
    source = factors / "Candidate.py"
    source.write_text("def signal(df, param, factor_name): return df\n")
    evaluation = tmp_path / "evaluation.py"
    evaluation.write_text(
        "import pandas as pd\n"
        "def load_candle_data(path): return path, [pd.DataFrame()]\n"
        "def discover_factors(root, configured=None): return configured\n"
        "def _split_factors(names, root): return names, []\n"
        "def _evaluate_factor_batches(*args):\n"
        "    quality = {'Candidate': {'pfs': 0.81, 'rre': 0.31}}\n"
        "    stats = pd.DataFrame([{'factor':'Candidate','ic_mean':0.04,'icir':0.21,"
        "'ic_tstat':2.0,'ic_win_rate':0.6}])\n"
        "    return ['Candidate'], quality, {'1H':[stats]}, {}\n"
        "def build_comparison_table(cols, quality, multi, periods):\n"
        "    return pd.DataFrame([{'Factor':'Candidate','PFS':0.81,'RRE':0.31,'Best_IC':0.04,'Best_IR':0.21}])\n"
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    settings = CryptoFactorPropSetting(
        data_path=data_path,
        evaluation_path=evaluation,
        baseline_dirs=[baseline],
        cache_dir=tmp_path / "cache",
        hold_periods=("1H",),
    )

    comparison, results = EvaluationAdapter(settings).evaluate(factors, ["Candidate"])
    assert comparison.iloc[0]["Factor"] == "Candidate"
    assert results[0].passed

    accepted = AcceptedFactorStore(settings.cache_dir)
    accepted.accept("Candidate", source, results[0].to_dict())
    assert accepted.names == ["Candidate"]
    assert accepted.source_dir("Candidate").joinpath("Candidate.py").exists()
    assert build_backtest_config(settings)["strategy"]["kwargs"]["topk"] == 50
