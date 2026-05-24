import struct
import wave

import pytest


@pytest.fixture
def sample_audio(tmp_path):
    path = tmp_path / "sample01.wav"
    _write_dummy_wav(path)
    return path


@pytest.fixture
def sample_ref_audio(tmp_path):
    path = tmp_path / "REF_sample01.wav"
    _write_dummy_wav(path)
    return path


@pytest.fixture
def experiment_layout(tmp_path):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    _write_dummy_wav(ref_dir / "REF_file1.wav")
    _write_dummy_wav(ref_dir / "REF_file2.wav")

    sys_a = tmp_path / "sys_a"
    sys_a.mkdir()
    _write_dummy_wav(sys_a / "file1.wav")
    _write_dummy_wav(sys_a / "file2.wav")

    sys_b = tmp_path / "sys_b"
    sys_b.mkdir()
    _write_dummy_wav(sys_b / "file1.wav")
    _write_dummy_wav(sys_b / "file2.wav")

    return tmp_path


@pytest.fixture
def mock_raw_scores_5():
    return {
        "mos": 3.2,
        "dnsmos_ovrl": 2.8,
        "scoreq": 3.5,
        "utmos": 3.0,
        "nisqa_mos": 3.1,
    }


@pytest.fixture
def mock_raw_scores_22(mock_raw_scores_5):
    extra = {
        "distill_mos": 3.3,
        "sigmos_ovrl": 3.4,
        "sigmos_col": 3.2,
        "sigmos_disc": 3.5,
        "sigmos_loud": 3.1,
        "sigmos_reverb": 3.6,
        "sigmos_sig": 3.3,
        "sigmos_noise": 3.0,
        "spksim": 0.7,
        "sbert": 0.85,
        "mcd": 5.2,
        "sdr": 12.5,
        "pesq": 2.8,
        "pesqc2": 2.9,
        "lsd": 3.1,
        "estoi": 0.82,
        "lps": 0.15,
    }
    return {**mock_raw_scores_5, **extra}


def _write_dummy_wav(path, duration_sec=1.0, sample_rate=16000, n_channels=1):
    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for _ in range(n_samples):
            f.writeframes(struct.pack("<h", 0))
