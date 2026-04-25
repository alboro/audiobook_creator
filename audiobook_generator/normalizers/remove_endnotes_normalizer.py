# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

STEP_NAME = "remove_endnotes"


class RemoveEndnotesNormalizer(BaseNormalizer):
    """Remove inline endnote numbers (digits directly after punctuation or word chars).

    Example: "See Smith¹ for details.2" → "See Smith for details."
    """

    STEP_NAME = STEP_NAME
    STEP_VERSION = 1

    def validate_config(self):
        pass  # no config required

    def normalize(self, text: str, chapter_title: str = "") -> str:
        # Remove digits that appear directly after word/punctuation characters
        return re.sub(r'(?<=[\w.,!?;:"\')])\d+', "", text)

