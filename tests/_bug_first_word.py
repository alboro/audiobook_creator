"""Debug script to reproduce first_word bug."""
import sys
sys.path.insert(0, '.')

from audiobook_generator.core.audio_checkers.base_audio_chunk_checker import normalize_for_compare
from audiobook_generator.core.audio_checkers.whisper_similarity_checker import _build_pre_compare_normalizer
from audiobook_generator.core.audio_checkers.first_word_checker import FirstWordChecker
from types import SimpleNamespace
from pathlib import Path

orig = 'Были спасены от её благочестивой разрушительности после её возвращения в католицизм.'
trans = 'да. Были спасены от ее благочестивой разрушительности после ее возвращения в католицизм.'

print("=== normalize_for_compare (no pre_compare) ===")
on = normalize_for_compare(orig)
tn = normalize_for_compare(trans)
print(f"orig_words[:3]: {on.split()[:3]}")
print(f"trans_words[:3]: {tn.split()[:3]}")
print(f"first mismatch: {on.split()[0]!r} != {tn.split()[0]!r} => {on.split()[0] != tn.split()[0]}")

pre = _build_pre_compare_normalizer('ru')
if pre:
    print()
    print("=== pre_compare applied ===")
    orig2 = pre(orig)
    trans2 = pre(trans)
    print(f"pre(orig)[:80]:  {orig2[:80]!r}")
    print(f"pre(trans)[:80]: {trans2[:80]!r}")
    on2 = normalize_for_compare(orig2)
    tn2 = normalize_for_compare(trans2)
    print(f"orig_words[:3]:  {on2.split()[:3]}")
    print(f"trans_words[:3]: {tn2.split()[:3]}")
    print(f"first mismatch: {on2.split()[0]!r} != {tn2.split()[0]!r} => {on2.split()[0] != tn2.split()[0]}")
else:
    print("pre_compare not available")

print()
print("=== FirstWordChecker.check() ===")
cfg = SimpleNamespace(
    language='ru-RU',
    audio_check_threshold=0.94,
)
checker = FirstWordChecker(cfg)
result = checker.check(Path('/tmp/fake.wav'), orig, trans, None)
print(f"disputed: {result.disputed}")
print(f"pre_compare available: {checker._pre_compare is not None}")

