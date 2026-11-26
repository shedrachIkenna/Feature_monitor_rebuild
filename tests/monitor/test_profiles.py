import pytest
import torch
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from monitor import model
from monitor.model import PROFILE_ENV, active_profile, load_profiles


@pytest.fixture
def fresh_load_cache():
    """Clear cached model before and after a test"""
    model.load_profile.cache_clear()
    yield
    model.load_profile.cache_clear()


def test_default_profile_is_gpt2_small(monkeypatch) -> None:
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert active_profile().name == "gpt2_small"


def test_environment_variable_selects_the_profile(monkeypatch) -> None:
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


def test_file_stems_keep_profiles_apart() -> None:
    profiles = load_profiles()
    assert profiles["gpt2_small"].file_stem == "gpt2_8"  # baseline model name
    assert profiles["gemma2_2b"].file_stem == "gemma-2-2b_16"


def test_loader_passes_each_profile_settings(monkeypatch, fresh_load_cache) -> None:
    calls = []

    class FakeBridge:
        @staticmethod
        def boot_transformers(name, device, dtype, hf_config_overrides):
            calls.append(("model", name, device, dtype, hf_config_overrides))
            return FakeBridge()

        def enable_compatibility_mode(self, no_processing):
            calls.append(("no_processing", no_processing))

    class FakeSAE:
        @staticmethod
        def from_pretrained(release, sae_id, device):
            calls.append(("sae", release, sae_id, device))
            return FakeSAE()

    monkeypatch.setattr(model, "TransformerBridge", FakeBridge)
    monkeypatch.setattr(model, "SAE", FakeSAE)

    monkeypatch.setenv(PROFILE_ENV, "gemma2_2b")
    model.load_model_and_sae("mps")
    model.load_model_and_sae("mps")  # second call to test if cache works
    monkeypatch.setenv(PROFILE_ENV, "gpt2_small")
    model.load_model_and_sae()
    monkeypatch.setattr(model, "truncation", lambda profile: {"layers": profile.layer + 1})
    model.load_profile(active_profile(), truncate=True)

    gpt2 = ("sae", "gpt2-small-res-jb", "blocks.8.hook_resid_pre", "cpu")
    assert calls == [
        ("model", "google/gemma-2-2b", "mps", torch.bfloat16, None),
        ("no_processing", True),
        ("sae", "gemma-scope-2b-pt-res-canonical", "layer_16/width_16k/canonical", "mps"),
        ("model", "gpt2", "cpu", torch.float32, None),
        ("no_processing", False),
        gpt2,
        ("model", "gpt2", "cpu", torch.float32, {"layers": 9}),
        ("no_processing", False),
        gpt2,
    ]


def test_truncation_user_each_configs_own_name_for_the_layer_count(monkeypatch) -> None:
    class FakeConfig:
        def __init__(self, attribute_map):
            self.attribute_map = attribute_map

    configs = {
        "gpt2": FakeConfig({"num_hidden_layers": "n_layers"}),
        "google/gemma-2-2b": FakeConfig({}),
    }
    monkeypatch.setattr(model.AutoConfig, "from_pretrained", lambda name: configs[name])
    profiles = load_profiles()
    assert model.truncation(profiles["gpt2_small"]) == {"n_layers": 9}
    assert model.truncation(profiles["gemma2_2b"]) == {"num_hidden_layers": 17}
