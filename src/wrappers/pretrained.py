from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from wrappers.composer import ComposerWrapper


def fix_and_register_ties(model, fwd_pattern="fwd", rev_pattern="rev"):
    """
    Ties weights in memory and generates the required _tied_weights_keys.
    To be used whenever model param count is ~2x or the warning of uninitialized
    paramters is recieved
    Future revision wont need after _tie_weights() is fixed.
    """
    tied_paths = []

    # Start with existing tied keys (like word embeddings) if present
    if hasattr(model, "_tied_weights_keys") and model._tied_weights_keys is not None:
        tied_paths.extend(model._tied_weights_keys)

    # Dictionary of all modules for quick lookup
    num_modules = dict(model.named_modules())

    for name, module in num_modules.items():
        if fwd_pattern in name and "proj" in name:
            rev_name = name.replace(fwd_pattern, rev_pattern)

            if rev_name in num_modules:
                fwd_mod = module
                rev_mod = num_modules[rev_name]

                # Check if it's a layer with weights (Linear, Conv, etc.)
                if hasattr(fwd_mod, "weight"):
                    # 1. Physical memory tie
                    rev_mod.weight = fwd_mod.weight
                    # 2. Add to HF tracking list
                    tied_paths.append(f"{rev_name}.weight")

                if hasattr(fwd_mod, "bias") and fwd_mod.bias is not None:
                    rev_mod.bias = fwd_mod.bias
                    tied_paths.append(f"{rev_name}.bias")
                # Logging for debugging
                # print(f"Bound: {rev_name} -> {name}")

    # Deduplicate and assign to the magic HF attribute
    model._tied_weights_keys = list(set(tied_paths))
    return model


def build_model(
    pretrained_name_or_path: str, from_scratch: bool, mlm: bool = False, **kwargs
) -> ComposerWrapper:
    model_config = AutoConfig.from_pretrained(
        pretrained_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name_or_path, trust_remote_code=True
    )
    tokenizer.characters = "atcg"
    if from_scratch:
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_name_or_path, trust_remote_code=True
        )
    model = fix_and_register_ties(model)
    return ComposerWrapper(model, tokenizer, mlm=mlm)
