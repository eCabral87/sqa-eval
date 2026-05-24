from __future__ import annotations

from pathlib import Path


def scan_audio(
    directory: str | Path, pattern: str = "*.wav", recursive: bool = False
) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def match_references(
    test_files: list[Path], ref_dir: str | Path, prefix: str = "REF_"
) -> dict[Path, Path | None]:
    ref_dir = Path(ref_dir)
    result: dict[Path, Path | None] = {}
    ref_files_lower: dict[str, Path] = {}
    if ref_dir.exists():
        for ref_file in ref_dir.iterdir():
            if ref_file.is_file():
                ref_files_lower[ref_file.name.lower()] = ref_file

    for test_path in test_files:
        ref_name = f"{prefix}{test_path.name}"
        ref_lower = ref_name.lower()
        result[test_path] = ref_files_lower.get(ref_lower)
    return result


def resolve_experiment(base_dir: str | Path, systems: list[str]) -> dict[str, list[Path]]:
    base_dir = Path(base_dir)
    result: dict[str, list[Path]] = {}
    for system in systems:
        sys_dir = base_dir / system
        if not sys_dir.exists() or not sys_dir.is_dir():
            raise ValueError(f"System directory does not exist: {sys_dir}")
        result[system] = sorted(sys_dir.glob("*.wav"))
    return result


def match_experiment_refs(
    system_files: dict[str, list[Path]],
    ref_dir: str | Path,
    prefix: str = "REF_",
) -> dict[str, dict[Path, Path | None]]:
    result: dict[str, dict[Path, Path | None]] = {}
    for system, files in system_files.items():
        result[system] = match_references(files, ref_dir, prefix)
    return result
