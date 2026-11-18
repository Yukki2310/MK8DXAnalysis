from mk8dx.scoring import TOTAL_POINTS, points_for_rank, validate_scores


def test_total_is_82():
    assert TOTAL_POINTS == 82


def test_valid_scores_pass():
    scores = {rank: points_for_rank(rank) for rank in range(1, 13)}
    result = validate_scores(scores)
    assert result.ok
    assert result.total == 82


def test_wrong_total_flagged():
    scores = {rank: points_for_rank(rank) for rank in range(1, 13)}
    scores[12] = 2  # 1 -> 2 に誤読したケース
    result = validate_scores(scores)
    assert not result.ok
    assert result.total == 83


def test_missing_rank_flagged():
    scores = {rank: points_for_rank(rank) for rank in range(1, 12)}
    result = validate_scores(scores)
    assert not result.ok
