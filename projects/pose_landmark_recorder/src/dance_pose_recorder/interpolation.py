"""Short-gap interpolation utilities for pose landmark time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class GapSegment:
    start_frame: int
    end_frame: int
    length: int
    prev_frame: int | None
    next_frame: int | None


def contiguous_ranges(frames: Iterable[int]) -> list[GapSegment]:
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return []

    ranges: list[GapSegment] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append(GapSegment(start, previous, previous - start + 1, None, None))
        start = previous = frame
    ranges.append(GapSegment(start, previous, previous - start + 1, None, None))
    return ranges


def find_short_gaps(
    frames: list[int],
    stable_mask: list[bool],
    max_gap: int,
    candidate_mask: list[bool] | None = None,
) -> list[GapSegment]:
    if len(frames) != len(stable_mask):
        raise ValueError("frames and stable_mask must have the same length")
    if candidate_mask is not None and len(frames) != len(candidate_mask):
        raise ValueError("frames and candidate_mask must have the same length")
    if max_gap < 1:
        return []

    if candidate_mask is None:
        candidate_mask = [not stable for stable in stable_mask]

    segments: list[GapSegment] = []
    index = 0
    while index < len(frames):
        if stable_mask[index]:
            index += 1
            continue

        start_index = index
        while index < len(frames) and not stable_mask[index]:
            index += 1
        end_index = index - 1
        prev_index = start_index - 1
        next_index = index
        length = end_index - start_index + 1

        if (
            length <= max_gap
            and prev_index >= 0
            and next_index < len(frames)
            and stable_mask[prev_index]
            and stable_mask[next_index]
            and all(candidate_mask[start_index : end_index + 1])
        ):
            segments.append(
                GapSegment(
                    start_frame=int(frames[start_index]),
                    end_frame=int(frames[end_index]),
                    length=length,
                    prev_frame=int(frames[prev_index]),
                    next_frame=int(frames[next_index]),
                )
            )

    return segments


def interpolate_group_linear(
    group: pd.DataFrame,
    fields: list[str],
    stable_column: str,
    max_gap: int,
    candidate_column: str | None = None,
) -> tuple[pd.DataFrame, list[GapSegment]]:
    """Fill short unstable runs bounded by stable rows using linear interpolation."""

    result = group.sort_values("frame").copy()
    frames = [int(frame) for frame in result["frame"].tolist()]
    stable_mask = [bool(value) for value in result[stable_column].tolist()]
    candidate_mask = None
    if candidate_column is not None:
        candidate_mask = [bool(value) for value in result[candidate_column].tolist()]
    segments = find_short_gaps(frames, stable_mask, max_gap, candidate_mask=candidate_mask)
    if not segments:
        return result, []

    frame_to_index = {int(frame): index for index, frame in enumerate(result["frame"].tolist())}
    for segment in segments:
        if segment.prev_frame is None or segment.next_frame is None:
            continue
        prev_index = frame_to_index[segment.prev_frame]
        next_index = frame_to_index[segment.next_frame]
        frame_span = segment.next_frame - segment.prev_frame
        if frame_span <= 0:
            continue

        for frame in range(segment.start_frame, segment.end_frame + 1):
            row_index = frame_to_index[frame]
            ratio = (frame - segment.prev_frame) / frame_span
            for field in fields:
                prev_value = result.iloc[prev_index][field]
                next_value = result.iloc[next_index][field]
                if pd.isna(prev_value) or pd.isna(next_value):
                    continue
                result.iat[row_index, result.columns.get_loc(field)] = (
                    float(prev_value) + (float(next_value) - float(prev_value)) * ratio
                )

    return result, segments
