import pytest 
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from monitor.model import PROFILE_ENV, active_profile, load_profile

def test_default_profile_is_gpt2_small(monkeypatch) -> None:
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert active_profile().name == "gpt2_small"

def environment_variable_selects_the_profile(monkeypatch) -> None: 
    monkeypatch.setenv(PROFILE_ENV, "gemma2_2b")
    profile = active_profile() 
    assert profile.model_name == "google/gemma-2-2b"
    assert profile.dtype == "bfloat16"
    assert profile.process_weights is False

def test_unknown_profile_is_refused(monkeypatch) -> None: 
    monkeypatch.setenv(PROFILE_ENV, "gpt5")
    with pytest.raises(ValueError, match="gpt2_small"):
        active_profile() 
