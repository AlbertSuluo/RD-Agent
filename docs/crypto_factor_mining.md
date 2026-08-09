# Hourly cryptocurrency factor mining

`fin_factor_crypto` is independent of the original `fin_factor` CSI300 workflow. It evaluates newly generated
hourly cryptocurrency factors on the full candle universe and admits them only when all four finite metrics are
strictly above the configured thresholds:

- `PFS > 0.8`
- `RRE > 0.3`
- `abs(Best_IC) > 0.03`
- `abs(Best_IR) > 0.2`

Equality fails. Trusted baseline directories are registered without applying these new-factor gates, so a baseline
such as `Alpha022` with `Best_IC == 0.03` remains available. Executability and `0.99` correlation de-duplication
are retained. Qlib results inform feedback but never override a deterministic admission decision.

## Run

```bash
rdagent fin_factor_crypto \
  --data-path /path/to/candles_1h_train.pkl \
  --evaluation-path /path/to/evaluation.py \
  --baseline-dir /path/to/effective_factors_test \
  --baseline-dir /path/to/effective_factors_test_haiyang_191 \
  --loop-n 10
```

Each option also has a `QLIB_CRYPTO_FACTOR_` environment setting. `BASELINE_DIRS` follows Pydantic Settings list
syntax (for example, a JSON array). Threshold and cache paths can be overridden through CLI options.

## Data and storage policy

The original pickle is exposed to factor workspaces through a symbolic link. Gate evaluation uses all instruments,
keeps OHLCV perturbations within one trial, and deletes full-universe candidate values after the gate finishes.
Only the fixed Top100 universe is cached for trusted/accepted factors. Cache keys include the pickle metadata,
factor source and helper sources, and selected universe. Model input is emitted as year-partitioned Parquet and
loaded through Qlib `StaticDataLoader`; the Qlib binary provider contains only Top100 test-period close/volume data.

Top100 is fixed by average `quote_volume` over 2021-01-01 through 2023-12-31. Model segments are 2021-2023 train,
2024 validation, and 2025-01-01 through 2025-07-06 test. The backtest uses `60min`, Top50 with five replacements,
close execution, 4 bp on both entry and exit, no benchmark, zero minimum fee, no price limit, and no trade unit.
Crypto risk metrics use 8,760 hours per year.

## Artifacts and resume

The cache root contains candidate metric CSVs, rejection reasons, accepted source files, Top100 values, Qlib
predictions/configuration/portfolio reports, and JSON summaries. RD-Agent's normal session checkpoints remain the
resume mechanism; pass their path with `--path`.
