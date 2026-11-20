import pytest 
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from monitor.model import PROFILE_ENV, active_profile, load_profiles

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

def test_bad_dtype_and_unknown_keys_are_refused(tmp_path) -> None:
    path = tmp_path / "profiles.yaml"
    fields = {
        "label": "x",
        "model_name": "m",
        "dtype": "float16",
        "process_weights": True,
        "sae_release": "r",
        "sae_id": "i",
        "layer": 1,
        "neuronpedia_id": "m/1",
    }
    path.write_text("p:\n" + "".join(f"  {k}: {v}\n" for k, v in fields.items()))
    with pytest.raises(ValueError, match="dtype"):
        load_profiles(path)
    path.write_text(path.read_text() + "  extra: 1\n")
    with pytest.raises(TypeError):
        load_profiles(path)

@pytest.mark.parametrize("profile", load_profiles().values(), ids=lambda p: p.name)
def test_profile_matches_the_installed_sae_lens_registry(profile) -> None:
    release = get_pretrained_saes_directory()[profile.sae_release]
    assert profile.sae_id in release.saes_map
    assert release.neuronpedia_id[profile.sae_id] == profile.neuronpedia_id

