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
    best: tuple[int, int, str] | None = None

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
                candidate = (start, -len(obj_tokens), original_obj)
                if best is None or candidate < best:
                    best = candidate

    return best[2] if best else None


def _find_object_in_text(object_list: Iterable[str], text: str) -> str | None:
    normalized_text = f" {_normalize(text)} "
    matches = []
    for obj in object_list:
        normalized_obj = _normalize(obj)
        if normalized_obj and f" {normalized_obj} " in normalized_text:
            matches.append((len(normalized_obj), obj))
    if not matches:
        return None
    return max(matches)[1]


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
