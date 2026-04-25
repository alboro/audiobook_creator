# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

STEP_NAME = "remove_reference_numbers"


class RemoveReferenceNumbersNormalizer(BaseNormalizer):
    """Remove bracketed reference numbers such as [3] or [12.1].

    Example: "See [3] and [12.1] for details." → "See  and  for details."
    """

    STEP_NAME = STEP_NAME
    STEP_VERSION = 1

    def validate_config(self):
        pass  # no config required

    def normalize(self, text: str, chapter_title: str = "") -> str:
        return re.sub(r"\[\d+(\.\d+)?\]", "", text)

