import pytest 
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from monitor.model import PROFILE_ENV, active_profile, load_profile

def test_default_profile_is_gpt2_small(monkeypatch) -> None:
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert active_profile().name == "gpt2_small"

