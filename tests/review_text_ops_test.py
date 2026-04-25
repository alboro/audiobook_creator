# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

from audiobook_generator.ui.review_text_ops import (
    apply_review_edit,
    collapse_adjacent_duplicate_paragraphs,
)


OLD_PARAGRAPH = (
    "Одно это соображение покажет, что доктрина искупления основана на чисто денежном представлении, "
    "соответствующем представлению о долге, который мог бы уплатить другой человек. "
    "А поскольку это денежное представление, в свою очередь, соответствует системе вторичных искуплений, "
    "получаемых посредством денег, отдаваемых церкви за прощение,. то вероятнее всего одни и те же люди "
    "выдумали и ту и другую из этих теорий. И что в действительности никакого искупления не существует; "
    "что это басня. И что человек находится в том же относительном положении к своему Созда́телю, в каком "
    "он находился всегда, с тех пор как существует человек. И что думать так - величайшее для него утешение."
)

NEW_PARAGRAPH = (
    "Одно это соображение покажет, что доктрина искупления основана на чисто денежном представлении, "
    "соответствующем представлению о долге, который мог бы уплатить другой человек. "
    "А поскольку это денежное представление, в свою очередь, соответствует системе вторичных искуплений, "
    "получаемых посредством денег, отдаваемых церкви за прощение. То вероятнее всего одни и те же люди "
    "выдумали и ту и другую из этих теорий. И что в действительности никакого искупления не существует; "
    "что это басня. И что человек находится в том же относительном положении к своему Созда́телю, в каком "
    "он находился всегда, с тех пор как существует человек. И что думать так - величайшее для него утешение."
)

PREV_PARAGRAPH = (
    "Ибо внутреннее свидетельство состоит в том, что теория или доктрина искупления имеет своим основанием "
    "представление о денежной справедливости. А не о нравственной справедливости."
)

NEXT_PARAGRAPH = (
    "Пусть он верит в это, и он будет жить более последовательно и нравственно, чем при всякой иной системе."
)


def test_collapse_adjacent_duplicate_paragraphs_removes_exact_neighbor_duplicate():
    text = (
        PREV_PARAGRAPH
        + "\n\n"
        + NEW_PARAGRAPH
        + "\n\n"
        + NEW_PARAGRAPH
        + "\n\n"
        + NEXT_PARAGRAPH
    )

    result = collapse_adjacent_duplicate_paragraphs(text)

    assert result.count("Одно это соображение покажет") == 1
    assert PREV_PARAGRAPH in result
    assert NEXT_PARAGRAPH in result



def test_apply_review_edit_collapses_accidental_duplicate_in_new_text():
    full_text = PREV_PARAGRAPH + "\n\n" + OLD_PARAGRAPH + "\n\n" + NEXT_PARAGRAPH
    accidentally_duplicated_new_text = NEW_PARAGRAPH + "\n\n" + NEW_PARAGRAPH

    result = apply_review_edit(full_text, OLD_PARAGRAPH, accidentally_duplicated_new_text)

    assert result.count("Одно это соображение покажет") == 1
    assert OLD_PARAGRAPH not in result
    assert NEW_PARAGRAPH in result



def test_apply_review_edit_regular_single_replace_still_works():
    full_text = PREV_PARAGRAPH + "\n\n" + OLD_PARAGRAPH + "\n\n" + NEXT_PARAGRAPH

    result = apply_review_edit(full_text, OLD_PARAGRAPH, NEW_PARAGRAPH)

    assert result.count("Одно это соображение покажет") == 1
    assert OLD_PARAGRAPH not in result
    assert NEW_PARAGRAPH in result

