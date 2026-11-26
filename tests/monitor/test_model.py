import pytest
import torch

from monitor.model import active_profile, load_model_and_sae


# Check that the loaded SAE can accept activations from the loaded model
@pytest.mark.requires_model
def test_sae_input_size_matches_model_width() -> None:
    model, sae = load_model_and_sae()  # loads the model and the SAE
    assert model.cfg.device == "cpu"  # Checks that the model is running on cpu
    # Check that the SAE input size (d_in) equals the model hidden dim (d_model)
    # This verifes that the SAE is sized to receive the model's activations
    assert sae.cfg.d_in == model.cfg.d_model


"""
The test below checks that the newer TransformerBridge model produces the same 
activations as the older HookedTransformer loader for GPT-2 Small. 
We do this by comparing the activations at the hooked point the SAE uses 
"""


@pytest.mark.requires_model
@pytest.mark.filterwarnings("ignore:HookedTransformer.*deprecated:DeprecationWarning")
def test_bridge_matches_deprecated_loader() -> None:
    """This test should be removed once Transforemer lens 4.0 removes HookedTransformer"""
    if active_profile().name != "gpt2_small":
        pytest.skip("compares against HookedTransformer's GPT-2 Small only")
    from transformer_lens import HookedTransformer

    model, sae = load_model_and_sae()
    reference = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    hook = sae.cfg.metadata.hook_name

    for prompt in ["How do I reset my password?", "What causes migraine"]:
        # For each of the above prompts, pass each prompt into the model and collect the activations at a selected hook
        _, bridge_cache = model.run_with_cache(prompt, names_filter=[hook])
        _, reference_cache = reference.run_with_cache(prompt, names_filter=[hook])
        # check that the activations from both models are numerically close enough
        # Test fails if the activations differ too much
        torch.testing.assert_close(bridge_cache[hook], reference_cache[hook])
