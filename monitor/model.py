# Load the active profile's language model and its residual-stream SAEs 

# The profiles is described in profiles.yaml 
# Default profile language model is GPT-2 

import os 
from dataclasses import dataclass 
from pathlib import Path 

import torch 
import yaml 

PROFILE_PATH = Path(__file__).resolve().with_name("profiles.yaml")
PROFILE_ENV = "FEATURE_MONITOR_PROFILE"
DEFAULT_PROFILE = "gpt2_small"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

@dataclass(frozen=True)
class Profile:
    """One model, the SAE that reads it and where that SAE's features live"""
    name: str 
    label: str 
    model_name: str 
    dtype: str 
    process_weights: str 
    sae_release: str 
    sae_id: str 
    layer: int 
    neuronpedia_id: str 

    @property
    def file_stem(self) -> str: 
        """Prefix for the profile's baseline, threshold and cache files. Example gpt2_8."""
        return f"{self.model_name.split("/")[-1]}_{self.layer}"


def load_profiles(path: Path = PROFILE_PATH) -> dict[str, Profile]:
    """Read every profile. Unknown or missing keys raises TypeError; bad dtype raises ValueError"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = {name: Profile(name=name, **fields) for name, fields in raw.items()}
    for profile in profiles.values():
        if profile.dtype not in DTYPES:
            raise ValueError(f"Profile {profile.name}: dtype must be one of {sorted(DTYPES)}")
    return profiles

def active_profile() -> Profile: 
    """Return the profile named by FEATURE_MONITOR_PROFILE, default is gpt2_small if unset"""
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    profiles = load_profile()
    if name not in profiles:
        raise ValueError(f"unknown {PROFILE_ENV}={name!r}; choose one of {sorted(profiles)}")
    return profiles[name]