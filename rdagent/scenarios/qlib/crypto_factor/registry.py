from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BaselineFactor:
    name: str
    source_path: Path
    source_dir: Path
    metadata: dict[str, object]
    trusted: bool = True


def _package_name(root: Path) -> str:
    return f"_rdagent_crypto_baseline_{hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]}"


def _ensure_package(root: Path) -> str:
    """Register an arbitrary baseline directory as a package.

    This preserves both ``from .operators ...`` and sibling helper imports.
    """

    root = root.resolve()
    name = _package_name(root)
    if name in sys.modules:
        return name
    init_path = root / "__init__.py"
    if init_path.exists():
        spec = importlib.util.spec_from_file_location(name, init_path, submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create baseline package for {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        module.__spec__ = spec
        sys.modules[name] = module
    return name


def import_factor_module(source_dir: Path, factor_name: str):
    if not factor_name.isidentifier():
        raise ValueError(f"invalid factor module name: {factor_name!r}")
    source_dir = Path(source_dir).expanduser().resolve()
    if not (source_dir / f"{factor_name}.py").is_file():
        raise FileNotFoundError(source_dir / f"{factor_name}.py")
    importlib.invalidate_caches()
    return importlib.import_module(f"{_ensure_package(source_dir)}.{factor_name}")


def _metadata_by_factor(root: Path) -> dict[str, dict[str, object]]:
    csv_path = root / "effective_factors_evaluation.csv"
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    factor_column = next((c for c in frame.columns if c.lower() == "factor"), None)
    if factor_column is None:
        raise ValueError(f"metadata CSV has no Factor column: {csv_path}")
    records = {}
    for _, row in frame.iterrows():
        record = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.to_dict().items()}
        records[str(record[factor_column]).strip()] = record
    return records


class BaselineRegistry:
    """Trusted baseline source registry; thresholds never filter these factors."""

    def __init__(self, factors: list[BaselineFactor]):
        names = [factor.name for factor in factors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate baseline factors across directories: {duplicates}")
        self.factors = factors

    @classmethod
    def load(cls, directories: list[Path], *, validate_imports: bool = True) -> "BaselineRegistry":
        factors: list[BaselineFactor] = []
        for raw_dir in directories:
            root = Path(raw_dir).expanduser().resolve()
            if not root.is_dir():
                raise NotADirectoryError(root)
            metadata = _metadata_by_factor(root)
            sources = [
                path
                for path in sorted(root.glob("*.py"))
                if path.name != "__init__.py" and not path.stem.startswith("_")
            ]
            for source in sources:
                if validate_imports:
                    module = import_factor_module(root, source.stem)
                    if not (hasattr(module, "signal") or hasattr(module, "signal_panel")):
                        raise AttributeError(f"baseline {source.stem} has no signal/signal_panel interface")
                factors.append(
                    BaselineFactor(
                        name=source.stem,
                        source_path=source,
                        source_dir=root,
                        metadata=metadata.get(source.stem, {}),
                    )
                )
        return cls(factors)

    @property
    def names(self) -> list[str]:
        return [factor.name for factor in self.factors]

    def by_name(self, name: str) -> BaselineFactor:
        for factor in self.factors:
            if factor.name == name:
                return factor
        raise KeyError(name)

    def prompt_summary(self) -> str:
        return ", ".join(self.names)
