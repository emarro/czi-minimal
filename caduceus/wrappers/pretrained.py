from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from wrappers.composer import ComposerWrapper


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
    return ComposerWrapper(model, tokenizer, mlm=mlm)
