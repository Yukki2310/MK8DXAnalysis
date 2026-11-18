from mk8dx.state import MatchState, Phase


def test_happy_path_and_cooldown():
    st = MatchState("2v2", cooldown_sec=60.0)
    assert st.phase is Phase.WAITING_RACE
    st.on_result_candidate()
    assert st.phase is Phase.RESULT_DETECTING
    st.on_screen_stable()
    assert st.phase is Phase.PROCESSING
    st.on_processing_done(valid=True, now=1000.0)
    assert st.phase is Phase.COOLDOWN
    assert st.current_race == 1
    # クールダウン中（60秒未満）は解除されない
    st.tick(screen_gone=True, now=1030.0)
    assert st.phase is Phase.COOLDOWN
    # 画面が消えて60秒経過で次レース待ちへ
    st.tick(screen_gone=True, now=1061.0)
    assert st.phase is Phase.WAITING_RACE


def test_invalid_goes_to_review_and_approve():
    st = MatchState("2v2")
    st.on_result_candidate()
    st.on_screen_stable()
    st.on_processing_done(valid=False, now=0.0)
    assert st.phase is Phase.AWAITING_REVIEW
    assert st.current_race == 0
    st.approve_pending(now=10.0)
    assert st.current_race == 1
    assert st.phase is Phase.COOLDOWN


def test_candidate_disappears():
    st = MatchState("2v2")
    st.on_result_candidate()
    st.on_screen_gone()
    assert st.phase is Phase.WAITING_RACE
