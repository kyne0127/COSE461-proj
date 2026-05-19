from __future__ import annotations

import re
from typing import Iterable


_VERBS = (
    "place",
    "put",
    "move",
    "pick",
    "grab",
    "take",
    "drop",
    "insert",
)
_DEST_PREPOSITIONS = ("in", "into", "on", "onto", "to", "inside")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    return normalized.split() if normalized else []


def _find_object_after_terms(
    object_list: Iterable[str],
    task_description: str,
    terms: Iterable[str],
    skip_terms: Iterable[str] = (),
) -> str | None:
    task_tokens = _tokenize(task_description)
    skip = set(skip_terms)
    objects = sorted((_normalize(obj), obj) for obj in object_list)
    # (start_pos, match_type, -match_len, original_obj)
    # match_type: 0=exact, 1=suffix — minimise so exact beats suffix
    best: tuple[int, int, int, str] | None = None

    for idx, token in enumerate(task_tokens):
        if token not in terms:
            continue
        start = idx + 1
        while start < len(task_tokens) and task_tokens[start] in skip:
            start += 1

        for normalized_obj, original_obj in objects:
            obj_tokens = normalized_obj.split()
            if not obj_tokens:
                continue
            end = start + len(obj_tokens)
            if task_tokens[start:end] == obj_tokens:
                candidate = (start, 0, -len(obj_tokens), original_obj)
                if best is None or candidate < best:
                    best = candidate
                continue
            # Suffix fallback: 'yellow marker' → try suffix ['marker'] at start
            for sfx in range(1, len(obj_tokens)):
                suffix = obj_tokens[sfx:]
                end_s = start + len(suffix)
                if task_tokens[start:end_s] == suffix:
                    candidate = (start, 1, -len(suffix), original_obj)
                    if best is None or candidate < best:
                        best = candidate
                    break

    return best[3] if best else None


def _find_object_in_text(object_list: Iterable[str], text: str) -> str | None:
    normalized_text = f" {_normalize(text)} "
    # (match_type, match_len, obj): match_type 0=exact, 1=suffix — minimise type,
    # maximise match_len so longest exact beats longest suffix.
    matches: list[tuple[int, int, str]] = []
    for obj in object_list:
        normalized_obj = _normalize(obj)
        if not normalized_obj:
            continue
        if f" {normalized_obj} " in normalized_text:
            matches.append((0, len(normalized_obj), obj))
            continue
        # Suffix fallback: 'yellow marker' → try 'marker' when model returns a more
        # specific label than the task wording (e.g. VLM adds a colour qualifier).
        tokens = normalized_obj.split()
        for suffix_start in range(1, len(tokens)):
            suffix = " ".join(tokens[suffix_start:])
            if suffix and f" {suffix} " in normalized_text:
                matches.append((1, len(suffix), obj))
                break
    if not matches:
        return None
    # Prefer exact (type=0), then longer match length
    return max(matches, key=lambda t: (-t[0], t[1]))[2]


def parse_roles(object_list: list[str], task_description: str) -> dict[str, str]:
    """Classify task objects into target and destination using simple language cues."""
    if not object_list:
        raise ValueError("object_list must contain at least one object")

    target = _find_object_after_terms(
        object_list,
        task_description,
        _VERBS,
        skip_terms=("the", "a", "an"),
    )
    destination = _find_object_after_terms(
        object_list,
        task_description,
        _DEST_PREPOSITIONS,
        skip_terms=("the", "a", "an"),
    )

    if target is None:
        target = _find_object_in_text(object_list, task_description)

    if destination is None:
        remaining = [obj for obj in object_list if obj != target]
        destination = _find_object_in_text(remaining, task_description)

    if target is None and object_list:
        target = object_list[0]
    if destination is None:
        remaining = [obj for obj in object_list if obj != target]
        destination = remaining[0] if remaining else target

    return {"target": target, "destination": destination}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse target/destination roles.")
    parser.add_argument("task_description")
    parser.add_argument("object_list", nargs="+")
    args = parser.parse_args()

    print(
        json.dumps(
            parse_roles(args.object_list, args.task_description),
            ensure_ascii=False,
            indent=2,
        )
    )
