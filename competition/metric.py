"""
Competition metric and submission helpers for the AE4353 contrail competition.

This one file has two jobs, and every change must keep both working:

1. The Kaggle metric. Paste this file, unchanged, into a Kaggle notebook and publish it as the
   competition's metric. Kaggle's rules for that notebook:
   - it calls a function named `score`, whose first three arguments are `solution`, `submission`
     and `row_id_column_name`; every argument must be type-annotated;
   - `score` must return a single, finite, non-null float;
   - it passes every column of both files except `Usage`, after aligning the rows on the row id;
   - only a `ParticipantVisibleError` is shown to participants, every other error to the host only.
   Nothing here runs on import except definitions.

2. The student helpers. The dataset ships this file, and the starter notebook imports `rle_encode`,
   `rle_decode`, `global_dice` and `write_submission` from it, so students score their validation
   split with exactly the code the leaderboard runs.

Run `python metric.py` for the doctests and self-tests.
"""
import numpy as np
import pandas as pd

MASK_SHAPE = (256, 256)
ID_COLUMN = "Id"
RLE_COLUMN = "PredictionString"


class ParticipantVisibleError(Exception):
    # Kaggle shows this error's message to participants. Every other error is shown to the
    # competition host only, so nothing about the solution can leak through it.
    pass


def rle_encode(mask):
    """
    Encode a 2D binary mask as a run-length encoded submission string.

    Pixels are numbered from 1, top to bottom and then left to right, so pixel
    1 is (0, 0), pixel 2 is (1, 0) and pixel 257 is (0, 1). That is column-major
    order, not the row-major order a plain .flatten() gives you.

    Args:
        mask (np.ndarray): 2D array, anything non-zero counts as contrail.

    Returns:
        str: space-delimited `start length` pairs, or "-" for an empty mask.

    >>> mask = np.zeros((4, 4), dtype=bool)
    >>> mask[0:3, 0] = mask[1, 2] = True
    >>> rle_encode(mask)
    '1 3 10 1'
    """
    flat = np.asarray(mask).flatten(order="F").astype(bool)

    # sentinels at both ends so a run touching an edge is still a transition
    padded = np.concatenate([[False], flat, [False]])
    transitions = np.flatnonzero(padded[1:] != padded[:-1]) + 1

    if transitions.size == 0:
        return "-"

    starts, ends = transitions[0::2], transitions[1::2]
    return " ".join(f"{s} {l}" for s, l in zip(starts, ends - starts))


def _is_empty(encoded):
    """True for '-', an empty string, or a missing value (pandas reads an empty cell as NaN)."""
    if encoded is None or (isinstance(encoded, float) and np.isnan(encoded)):
        return True
    return str(encoded).strip() in ("", "-")


def _parse_runs(encoded, shape=MASK_SHAPE):
    """
    Check a non-empty RLE string and return its 1-indexed (starts, lengths).

    Raises:
        ParticipantVisibleError: if the string is not valid for a mask of this shape.
    """
    try:
        values = np.array([int(token) for token in str(encoded).split()], dtype=np.int64)
    except ValueError:
        raise ParticipantVisibleError(
            "the run-length string must be space-separated integers, or '-' for an empty mask"
        ) from None
    if len(values) % 2:
        raise ParticipantVisibleError(
            "the run-length string has an odd number of values; runs are `start length` pairs"
        )

    starts, lengths = values[0::2], values[1::2]
    n_pixels = shape[0] * shape[1]
    if (starts < 1).any() or (lengths < 1).any() or (starts + lengths - 1 > n_pixels).any():
        raise ParticipantVisibleError(
            f"the run-length string has a run outside the mask: pixels are numbered 1 to "
            f"{n_pixels}, and every length must be at least 1"
        )
    return starts, lengths


def rle_decode(encoded, shape=MASK_SHAPE):
    """
    Decode a run-length encoded string back into a 2D boolean mask.

    Args:
        encoded (str): an RLE string, or "-" for an empty mask.
        shape (tuple): output shape, defaults to (256, 256).

    Returns:
        np.ndarray: boolean array of the given shape.

    Raises:
        ParticipantVisibleError: if the string is not a valid encoding for this shape.

    >>> rle_decode("1 3 10 1", shape=(4, 4)).astype(int)
    array([[1, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    if not _is_empty(encoded):
        starts, lengths = _parse_runs(encoded, shape)
        for start, length in zip(starts - 1, lengths):  # the string is 1-indexed, numpy is not
            mask[start:start + length] = True
    return mask.reshape(shape, order="F")


def global_dice(pred, true):
    """
    Global Dice coefficient, 2|X and Y| / (|X| + |Y|).

    Every pixel of every record is pooled into one intersection and one total,
    so this is not the mean of per-record Dice scores. Most records here contain
    no contrail at all, and a per-record mean would reward predicting nothing.

    This is the competition metric. Use it on your own validation split, and in
    particular use it to pick the threshold at which you binarise your model's
    output.

    Args:
        pred (np.ndarray): predicted masks, boolean or 0/1.
        true (np.ndarray): ground truth masks, same shape as pred.

    Returns:
        float: the global Dice coefficient, or 1.0 if both are empty.
    """
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)

    intersection = np.count_nonzero(pred & true)
    total = np.count_nonzero(pred) + np.count_nonzero(true)

    return 1.0 if total == 0 else float(2.0 * intersection / total)


def write_submission(record_ids, masks, path="submission.csv"):
    """
    Write a submission file with the header the competition expects.

    Args:
        record_ids: one Id per mask, in any order.
        masks (np.ndarray): masks of shape (N, H, W), boolean or 0/1.
        path: output path or file-like object, defaults to "submission.csv".
    """
    pd.DataFrame({
        ID_COLUMN: list(record_ids),
        RLE_COLUMN: [rle_encode(m) for m in masks],
    }).to_csv(path, index=False)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Global Dice coefficient for run-length encoded contrail masks.

    Both files hold one row per record: the row id and a `PredictionString`, a run-length encoded
    256x256 mask. Runs are space-delimited `start length` pairs, pixels are numbered from 1 in
    column-major order (top to bottom, then left to right), and an empty mask is written as '-'.

    Every pixel of every scored record is pooled into a single intersection and a single total, and
    the score is 2 * |intersection| / (|predicted| + |true|). It is not the mean of per-record Dice
    scores: most records contain no contrail, and a per-record mean would reward predicting nothing.
    Predicting nothing scores exactly 0. If neither side has a single positive pixel the score is 1.

    Rows are matched on the row id, so the result does not depend on row order, and extra
    submission rows are ignored. Ids are compared as strings. Every submission row is checked
    before any scoring, so a malformed row is reported the same way whichever rows are scored.

    >>> solution = pd.DataFrame({"Id": ["a1", "b2"], "PredictionString": ["1 3 10 2", "-"]})
    >>> submission = pd.DataFrame({"Id": ["b2", "a1"], "PredictionString": ["-", "1 3 10 2"]})
    >>> score(solution.copy(), submission.copy(), "Id")
    1.0

    Three of the five true pixels, and nothing else: 2 * 3 / (3 + 5).

    >>> submission["PredictionString"] = ["-", "1 3"]
    >>> score(solution.copy(), submission.copy(), "Id")
    0.75

    >>> submission["PredictionString"] = ["-", "-"]
    >>> score(solution.copy(), submission.copy(), "Id")
    0.0

    >>> submission["PredictionString"] = ["-", "1 3 10"]
    >>> score(solution.copy(), submission.copy(), "Id")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ParticipantVisibleError: Id a1: the run-length string has an odd number of values
    """
    for column in (row_id_column_name, RLE_COLUMN):
        if column not in submission.columns:
            raise ParticipantVisibleError(f"the submission needs a '{column}' column")

    # Compare ids as strings on both sides. A record id is 12 hex characters, so
    # roughly one in 200 is all digits -- pandas reads a column of those as
    # integers, and "000000000001" stops matching itself across two files.
    solution_ids = solution[row_id_column_name].astype(str)
    submission_ids = submission[row_id_column_name].astype(str)

    duplicated = submission_ids[submission_ids.duplicated()]
    if len(duplicated):
        raise ParticipantVisibleError(
            f"the submission has {len(duplicated)} duplicated Id(s), for example "
            f"{sorted(set(duplicated))[:3]}"
        )

    for record_id, encoded in zip(submission_ids, submission[RLE_COLUMN]):
        if not _is_empty(encoded):
            try:
                _parse_runs(encoded)
            except ParticipantVisibleError as error:
                raise ParticipantVisibleError(f"Id {record_id}: {error}") from None

    predictions = dict(zip(submission_ids, submission[RLE_COLUMN]))
    missing = set(solution_ids) - set(predictions)
    if missing:
        # no example ids: they would reveal which records are being scored
        raise ParticipantVisibleError(
            f"submission is missing {len(missing)} Id(s); it needs one row per test record"
        )

    intersection = 0
    n_pred = 0
    n_true = 0
    for record_id, truth in zip(solution_ids, solution[RLE_COLUMN]):
        pred = rle_decode(predictions[record_id])
        try:
            gt = rle_decode(truth)
        except ParticipantVisibleError as error:
            # a malformed solution is the host's problem, never the participant's
            raise ValueError(f"solution row {record_id}: {error}") from None
        intersection += np.count_nonzero(pred & gt)
        n_pred += np.count_nonzero(pred)
        n_true += np.count_nonzero(gt)

    if n_pred + n_true == 0:
        return 1.0

    return float(2.0 * intersection / (n_pred + n_true))


if __name__ == "__main__":
    # Also runs when this file is pasted into a notebook, where __name__ is "__main__" too,
    # so it must stay free of file I/O and of anything slow.
    import doctest
    import io

    failures, tests = doctest.testmod()
    assert failures == 0, f"{failures} of {tests} doctests failed"
    print(f"doctests ok: {tests}")

    rng = np.random.default_rng(4353)

    # an asymmetric mask catches a transposed encoder, which a square mask hides
    asymmetric = np.zeros(MASK_SHAPE, dtype=bool)
    asymmetric[2, 100:150] = True

    for name, mask in [
        ("empty", np.zeros(MASK_SHAPE, dtype=bool)),
        ("full", np.ones(MASK_SHAPE, dtype=bool)),
        ("sparse", rng.random(MASK_SHAPE) < 0.005),
        ("checkerboard", np.indices(MASK_SHAPE).sum(axis=0) % 2 == 0),
        ("asymmetric", asymmetric),
    ]:
        assert np.array_equal(rle_decode(rle_encode(mask)), mask), name
    print("round-trip ok: empty, full, sparse, checkerboard, asymmetric")

    # write_submission -> CSV -> score, end to end, the way Kaggle sees it: the
    # solution without its Usage column, and an all-digit id to trip dtype inference
    masks = np.stack([rng.random(MASK_SHAPE) < 0.004 for _ in range(8)])
    ids = [f"{0xa0f31 + i * 0x9e377:012x}" for i in range(len(masks) - 1)] + ["000000000001"]
    buffer = io.StringIO()
    write_submission(ids, masks, buffer)
    submission = pd.read_csv(io.StringIO(buffer.getvalue()))
    assert list(submission.columns) == [ID_COLUMN, RLE_COLUMN], list(submission.columns)

    truth = pd.DataFrame({ID_COLUMN: ids, RLE_COLUMN: [rle_encode(m) for m in masks]})
    assert score(truth.copy(), submission.copy(), ID_COLUMN) == 1.0
    assert score(truth.copy(), submission.assign(**{RLE_COLUMN: "-"}), ID_COLUMN) == 0.0
    assert score(truth.assign(Usage="Public"), submission.copy(), ID_COLUMN) == 1.0

    for bad in ["1 2 3", "0 5", "65536 2", "1 0", "a b", "1.5 2"]:
        try:
            score(truth.copy(), submission.assign(**{RLE_COLUMN: bad}), ID_COLUMN)
        except ParticipantVisibleError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    print("score() ok: perfect 1.0, empty 0.0, Usage ignored, malformed strings rejected")
