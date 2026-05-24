import pytest

from sqa_eval.io import match_experiment_refs, match_references, resolve_experiment, scan_audio


class TestScanAudio:
    def test_scan_audio_finds_wav(self, tmp_path):
        (tmp_path / "a.wav").touch()
        (tmp_path / "b.wav").touch()
        files = scan_audio(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".wav" for f in files)

    def test_scan_audio_ignores_non_wav(self, tmp_path):
        (tmp_path / "a.wav").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "c.mp4").touch()
        files = scan_audio(tmp_path)
        assert len(files) == 1

    def test_scan_audio_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.wav").touch()
        (sub / "b.wav").touch()
        files = scan_audio(tmp_path, recursive=True)
        assert len(files) == 2

    def test_scan_audio_empty_dir(self, tmp_path):
        assert scan_audio(tmp_path) == []


class TestMatchReferences:
    def test_match_references_found(self, tmp_path):
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        test = tmp_path / "file1.wav"
        test.touch()
        (ref_dir / "REF_file1.wav").touch()

        mapping = match_references([test], ref_dir)
        assert mapping[test] is not None

    def test_match_references_not_found(self, tmp_path):
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        test = tmp_path / "file1.wav"
        test.touch()

        mapping = match_references([test], ref_dir)
        assert mapping[test] is None

    def test_match_references_case_insensitive(self, tmp_path):
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        test = tmp_path / "file1.wav"
        test.touch()
        (ref_dir / "REF_FILE1.wav").touch()

        mapping = match_references([test], ref_dir)
        assert mapping[test] is not None

    def test_match_references_custom_prefix(self, tmp_path):
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        test = tmp_path / "file1.wav"
        test.touch()
        (ref_dir / "CLEAN_file1.wav").touch()

        mapping = match_references([test], ref_dir, prefix="CLEAN_")
        assert mapping[test] is not None


class TestResolveExperiment:
    def test_resolve_experiment(self, experiment_layout):
        result = resolve_experiment(experiment_layout, ["sys_a", "sys_b"])
        assert len(result["sys_a"]) == 2
        assert len(result["sys_b"]) == 2

    def test_resolve_experiment_missing_dir(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            resolve_experiment(tmp_path, ["no_such_system"])


class TestMatchExperimentRefs:
    def test_match_experiment_refs(self, experiment_layout):
        system_files = resolve_experiment(experiment_layout, ["sys_a", "sys_b"])
        ref_mapping = match_experiment_refs(system_files, experiment_layout / "ref")
        for sys_name, mapping in ref_mapping.items():
            for test_path, ref_path in mapping.items():
                assert ref_path is not None
