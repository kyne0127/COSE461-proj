from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from extraction.role_parser import parse_roles


# ---------------------------------------------------------------------------
# TestBasicVerbPreposition — 동사 + 전치사 모두 존재하는 정상 케이스
# ---------------------------------------------------------------------------

class TestBasicVerbPreposition:
    def test_place_on(self):
        r = parse_roles(["cup", "box"], "place cup on box")
        assert r == {"target": "cup", "destination": "box"}

    def test_put_into(self):
        r = parse_roles(["mug", "tray"], "put mug into tray")
        assert r == {"target": "mug", "destination": "tray"}

    def test_move_to(self):
        r = parse_roles(["block", "bin"], "move block to bin")
        assert r == {"target": "block", "destination": "bin"}

    def test_pick_in(self):
        r = parse_roles(["apple", "bowl"], "pick apple in bowl")
        assert r == {"target": "apple", "destination": "bowl"}

    def test_grab_onto(self):
        r = parse_roles(["bottle", "shelf"], "grab bottle onto shelf")
        assert r == {"target": "bottle", "destination": "shelf"}

    def test_take_inside(self):
        r = parse_roles(["marker", "box"], "take marker inside box")
        assert r == {"target": "marker", "destination": "box"}

    def test_drop_on(self):
        r = parse_roles(["ball", "table"], "drop ball on table")
        assert r == {"target": "ball", "destination": "table"}

    def test_insert_into(self):
        r = parse_roles(["pin", "hole"], "insert pin into hole")
        assert r == {"target": "pin", "destination": "hole"}


# ---------------------------------------------------------------------------
# TestArticleSkipping — the / a / an 관사 무시
# ---------------------------------------------------------------------------

class TestArticleSkipping:
    def test_the_before_target(self):
        r = parse_roles(["cup", "box"], "place the cup on the box")
        assert r == {"target": "cup", "destination": "box"}

    def test_a_before_target(self):
        r = parse_roles(["cup", "box"], "place a cup on a box")
        assert r == {"target": "cup", "destination": "box"}

    def test_an_before_target(self):
        r = parse_roles(["apple", "bowl"], "put an apple in a bowl")
        assert r == {"target": "apple", "destination": "bowl"}

    def test_multiple_articles(self):
        r = parse_roles(["mug", "tray"], "move the mug onto the tray")
        assert r == {"target": "mug", "destination": "tray"}


# ---------------------------------------------------------------------------
# TestMultiWordObjects — 다중 단어 오브젝트 레이블
# ---------------------------------------------------------------------------

class TestMultiWordObjects:
    def test_two_word_target(self):
        r = parse_roles(["blue cup", "box"], "place the blue cup on the box")
        assert r == {"target": "blue cup", "destination": "box"}

    def test_two_word_destination(self):
        r = parse_roles(["cup", "wooden tray"], "place the cup on the wooden tray")
        assert r == {"target": "cup", "destination": "wooden tray"}

    def test_both_multi_word(self):
        r = parse_roles(["blue cup", "wooden tray"], "place the blue cup on the wooden tray")
        assert r == {"target": "blue cup", "destination": "wooden tray"}

    def test_three_word_object(self):
        r = parse_roles(["small red block", "tray"], "pick the small red block and place it on the tray")
        assert r == {"target": "small red block", "destination": "tray"}

    def test_longer_label_preferred_over_substring(self):
        # "blue cup" should win over "cup" when both match
        r = parse_roles(["cup", "blue cup", "box"], "place the blue cup on the box")
        assert r["target"] == "blue cup"


# ---------------------------------------------------------------------------
# TestAllVerbs — 각 동사 개별 검증
# ---------------------------------------------------------------------------

class TestAllVerbs:
    @pytest.mark.parametrize("verb", ["place", "put", "move", "pick", "grab", "take", "drop", "insert"])
    def test_each_verb_identifies_target(self, verb):
        task = f"{verb} cup on box"
        r = parse_roles(["cup", "box"], task)
        assert r["target"] == "cup"


# ---------------------------------------------------------------------------
# TestAllPrepositions — 각 전치사 개별 검증
# ---------------------------------------------------------------------------

class TestAllPrepositions:
    @pytest.mark.parametrize("prep", ["in", "into", "on", "onto", "to", "inside"])
    def test_each_preposition_identifies_destination(self, prep):
        task = f"place cup {prep} box"
        r = parse_roles(["cup", "box"], task)
        assert r["destination"] == "box"


# ---------------------------------------------------------------------------
# TestFallbacks — 동사/전치사 없을 때 fallback 로직
# ---------------------------------------------------------------------------

class TestFallbacks:
    def test_no_verb_falls_back_to_text_search_for_target(self):
        # "goes" is not a known verb → target from text search
        r = parse_roles(["cup", "box"], "the cup goes into the box")
        assert r["target"] == "cup"
        assert r["destination"] == "box"

    def test_no_preposition_falls_back_to_remaining_object(self):
        # no known preposition → destination = remaining object
        r = parse_roles(["cup", "box"], "grab the cup")
        assert r["target"] == "cup"
        assert r["destination"] == "box"

    def test_neither_verb_nor_preposition_uses_index_fallback(self):
        # no verb, no preposition → object_list[0] = target, next = destination
        r = parse_roles(["cup", "box"], "cup and box task")
        assert r["target"] in ("cup", "box")
        assert r["destination"] in ("cup", "box")
        assert r["target"] != r["destination"]

    def test_single_object_destination_equals_target(self):
        # only one object → destination falls back to target
        r = parse_roles(["cup"], "grab the cup")
        assert r["target"] == "cup"
        assert r["destination"] == "cup"

    def test_verb_not_matched_but_object_in_text(self):
        # verb doesn't appear but object names are in text
        r = parse_roles(["marker", "bottle"], "move marker to bottle")
        assert r["target"] == "marker"
        assert r["destination"] == "bottle"


# ---------------------------------------------------------------------------
# TestCaseAndPunctuation — 대소문자 / 구두점
# ---------------------------------------------------------------------------

class TestCaseAndPunctuation:
    def test_uppercase_task(self):
        r = parse_roles(["cup", "box"], "Place the Cup on the Box.")
        assert r == {"target": "cup", "destination": "box"}

    def test_mixed_case_object_list(self):
        r = parse_roles(["Blue Cup", "Wooden Tray"], "place the blue cup on the wooden tray")
        assert r == {"target": "Blue Cup", "destination": "Wooden Tray"}

    def test_punctuation_in_task(self):
        r = parse_roles(["cup", "box"], "place the cup on the box!")
        assert r == {"target": "cup", "destination": "box"}


# ---------------------------------------------------------------------------
# TestOutputSchema — 반환 타입 구조 검증
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_returns_dict_with_target_and_destination(self):
        r = parse_roles(["cup", "box"], "place cup on box")
        assert isinstance(r, dict)
        assert set(r.keys()) == {"target", "destination"}

    def test_values_are_strings(self):
        r = parse_roles(["cup", "box"], "place cup on box")
        assert isinstance(r["target"], str)
        assert isinstance(r["destination"], str)

    def test_values_come_from_object_list(self):
        objs = ["cup", "box"]
        r = parse_roles(objs, "place cup on box")
        assert r["target"] in objs
        assert r["destination"] in objs


# ---------------------------------------------------------------------------
# TestEdgeCases — 경계 조건
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_object_list_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_roles([], "place cup on box")

    def test_object_not_in_task_description(self):
        # object exists in list but not in task text → fallback assigns anyway
        r = parse_roles(["pen", "notebook"], "do something useful")
        assert r["target"] in ("pen", "notebook")
        assert r["destination"] in ("pen", "notebook")

    def test_target_and_destination_differ_when_two_objects(self):
        r = parse_roles(["cup", "box"], "place cup on box")
        assert r["target"] != r["destination"]

    def test_real_task_marker_sprite(self):
        # task used in sandbox/remote_inference_example.py
        r = parse_roles(["marker", "sprite bottle"], "move the marker next to the sprite bottle")
        assert r["target"] == "marker"
        assert r["destination"] == "sprite bottle"

    def test_real_task_block_tray(self):
        # task from proposal.md motivating example
        r = parse_roles(["red block", "tray"], "Place the red block in the tray.")
        assert r["target"] == "red block"
        assert r["destination"] == "tray"
