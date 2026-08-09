from pathlib import Path

import pandas as pd

from rdagent.scenarios.qlib.crypto_factor.registry import BaselineRegistry


def _make_baselines(root: Path, names: list[str], *, padded_csv: bool = False) -> None:
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "_helper.py").write_text("VALUE = 1\n")
    for name in names:
        (root / f"{name}.py").write_text(
            "from ._helper import VALUE\n"
            "def signal(df, param, factor_name):\n"
            "    df[factor_name] = df['close'] * VALUE\n"
            "    return df\n"
        )
    factor_column = "Factor     " if padded_csv else "Factor"
    pd.DataFrame(
        {
            factor_column: [f"{name}   " if padded_csv else name for name in names],
            " Best_IC" if padded_csv else "Best_IC": [0.03 if name == "Alpha022" else 0.04 for name in names],
        }
    ).to_csv(root / "effective_factors_evaluation.csv", index=False)


def test_loads_29_plus_33_trusted_baselines_and_relative_imports(tmp_path):
    first_names = [f"Legacy{i:02d}" for i in range(29)]
    second_names = [f"Alpha{i:03d}" for i in range(33) if i != 22] + ["Alpha022"]
    first, second = tmp_path / "first", tmp_path / "second"
    _make_baselines(first, first_names, padded_csv=True)
    _make_baselines(second, second_names)

    registry = BaselineRegistry.load([first, second])

    assert len(registry.factors) == 62
    alpha022 = registry.by_name("Alpha022")
    assert alpha022.trusted is True
    assert alpha022.metadata["Best_IC"] == 0.03
