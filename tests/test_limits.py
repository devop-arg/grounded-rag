"""Rate limiter: per-process salt when the configured one is empty, stale-key sweep."""

from __future__ import annotations

import hashlib

from app.limits import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_configured_salt_is_used_verbatim() -> None:
    limiter = RateLimiter(1, 60, salt="pepper")
    assert limiter.key("1.2.3.4") == hashlib.sha256(b"pepper1.2.3.4").hexdigest()


def test_empty_salt_becomes_a_random_per_process_salt() -> None:
    first = RateLimiter(1, 60, salt="")
    second = RateLimiter(1, 60, salt="")
    assert first.key("1.2.3.4") == first.key("1.2.3.4")  # stable inside the instance
    assert first.key("1.2.3.4") != second.key("1.2.3.4")  # not a fixed constant
    assert first.key("1.2.3.4") != hashlib.sha256(b"1.2.3.4").hexdigest()  # not brute-forceable


def test_salt_is_not_in_repr_or_key() -> None:
    limiter = RateLimiter(1, 60, salt="")
    assert "1.2.3.4" not in limiter.key("1.2.3.4")
    assert "RateLimiter(limit=1, window_s=60.0)" == repr(limiter)


def test_stale_keys_are_swept_and_live_ones_kept() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1, 60, salt="s", clock=clock, sweep_every=3)
    for index in range(5):
        assert limiter.allow(f"10.0.0.{index}") is True
    assert len(limiter._hits) == 5

    clock.advance(120)  # every window is now empty
    assert limiter.allow("10.1.1.1") is True  # 6th call triggers the sweep
    assert len(limiter._hits) == 1
    assert limiter.count("10.1.1.1") == 1


def test_sweep_keeps_an_ip_that_is_still_inside_its_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, 60, salt="s", clock=clock, sweep_every=2)
    assert limiter.allow("10.0.0.1") is True  # call 1
    clock.advance(30)
    assert limiter.allow("10.0.0.2") is True  # call 2 -> sweep (nothing stale yet)
    assert len(limiter._hits) == 2

    clock.advance(40)  # t=1070: 10.0.0.1 (t=1000) is outside its window
    assert limiter.allow("10.0.0.3") is True  # call 3
    assert limiter.allow("10.0.0.1") is True  # call 4 -> sweep
    assert limiter.count("10.0.0.2") == 1  # t=1030 > cutoff 1010: kept
    assert limiter.count("10.0.0.1") == 1
    assert len(limiter._hits) == 3


def test_sweep_does_not_change_the_limit_behaviour() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, 60, salt="s", clock=clock, sweep_every=1)
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is False
    clock.advance(61)
    assert limiter.allow("10.0.0.1") is True
    assert limiter.count("10.0.0.1") == 1
