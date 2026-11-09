"""Time splits on returns-table rows (3.0 spec §7, §8, §9): CSCV blocks, walk-forward
windows, purged folds with an embargo. Pure index arithmetic."""
import numpy as np
import pytest

from plutobt.splits import (
    MIN_BARS,
    SplitError,
    cscv_blocks,
    purged_folds,
    walk_forward_windows,
)


def test_min_bars():
    assert MIN_BARS == 30


def test_cscv_blocks_cover_rows_after_the_drop():
    dropped, blocks = cscv_blocks(1619, 16)
    assert dropped == 1619 % 16
    assert len(blocks) == 16 and all(len(b) == 1619 // 16 for b in blocks)
    assert blocks[0].start == dropped and blocks[-1].stop == 1619
    assert all(a.stop == b.start for a, b in zip(blocks, blocks[1:], strict=False))


def test_walk_forward_rolling_and_anchored():
    rolling = walk_forward_windows(600, 5, anchored=False)
    assert len(rolling) == 5
    assert rolling[0] == (range(0, 100), range(100, 200))
    assert rolling[4] == (range(400, 500), range(500, 600))
    anchored = walk_forward_windows(600, 5, anchored=True)
    assert anchored[2] == (range(0, 300), range(300, 400))
    assert [t for _, t in anchored] == [t for _, t in rolling]


@pytest.mark.parametrize("folds", range(2, 9))
@pytest.mark.parametrize("embargo", [0, 1, 7, 40])
def test_purged_embargo_rows_never_train(folds, embargo):
    n = 60 * folds + 13            # blocks of 60+ rows, so even a 40-bar embargo leaves training
    out = purged_folds(n, folds, embargo)
    assert len(out) == folds
    assert sorted(r for _, test in out for r in test) == list(range(n))
    for train, test in out:
        t = set(train.tolist())
        assert not t & set(test)
        assert not t & set(range(test.stop, test.stop + embargo))
        # everything else trains
        assert t == set(range(n)) - set(test) - set(range(test.stop, test.stop + embargo))
        assert train.dtype.kind == "i" and np.all(np.diff(train) > 0)


def test_remainders_497_rows():
    dropped, blocks = cscv_blocks(497, 16)
    assert dropped == 1 and blocks[0] == range(1, 32)
    windows = walk_forward_windows(497, 5, anchored=False)
    assert windows[-1][1] == range(5 * 82, 497)          # the extra 5 rows join the last
    assert all(len(t) == 82 for _, t in windows[:-1])
    folds = purged_folds(497, 6, 0)
    assert folds[-1][1] == range(5 * 82, 497)


def test_refusals_name_the_largest_supported():
    with pytest.raises(SplitError, match=r"at most 12"):
        cscv_blocks(400, 16)
    with pytest.raises(SplitError, match="even"):
        cscv_blocks(1600, 15)
    with pytest.raises(SplitError, match=r"at most 4"):
        walk_forward_windows(150, 5, anchored=False)         # chunks of 25 rows
    with pytest.raises(SplitError, match=r"at most 5"):
        purged_folds(170, 6, 0)                               # blocks of 28 rows
    with pytest.raises(SplitError, match="no training rows"):
        purged_folds(60, 2, 30)                               # second fold: embargo
    with pytest.raises(SplitError):
        purged_folds(600, 6, -1)


def test_too_short_for_any_split_says_so():
    with pytest.raises(SplitError, match="too short"):
        cscv_blocks(50, 2)
