"""
Register hydra helper resolvers
"""

import re
from omegaconf import OmegaConf


def register_resolvers():
    """
    Register resolver clean that replaces characters that might make parsing pasing difficult with another string
    You can call clean in hydra like ${clean:${str_to_parse}}
    """
    if not OmegaConf.has_resolver("clean"):
        OmegaConf.register_new_resolver(
            name="clean",
            resolver=lambda s, replace="": re.sub(r"[\'\",\\s]+", replace, s).replace(
                " ", "-"
            )
            if s is not None
            else s,
        )


register_resolvers()
