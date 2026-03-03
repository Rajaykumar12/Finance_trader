"""Stage 4 verification — C++ Gamma Engine tests.

Tests Black-Scholes pricing against known values and validates the
full GEX computation pipeline.
"""

import numpy as np
import pytest

import gamma_engine


# ── Black-Scholes unit tests ──────────────────────────────────────────

class TestBlackScholes:
    """Verify BS pricing against well-known analytical values."""

    def test_atm_call(self):
        """At-the-money call should be roughly 0.4 * S * sigma * sqrt(T)."""
        # S=100, K=100, T=1yr, r=5%, sigma=20%
        res = gamma_engine.black_scholes(100, 100, 1.0, 0.05, 0.20, True)
        # Known BS price ≈ 11.91 for S=K=100, T=1, r=5%, σ=20%
        assert res.price == pytest.approx(11.91, abs=0.1), f"Price={res.price}"
        assert 0.5 < res.delta < 0.7, f"ATM call delta should be ~0.55-0.65, got {res.delta}"
        assert res.gamma > 0, "Gamma should be positive"
        print(f"\n✅ ATM Call: price={res.price:.4f}, delta={res.delta:.4f}, gamma={res.gamma:.6f}")

    def test_deep_itm_call(self):
        """Deep in-the-money call should be close to intrinsic value."""
        res = gamma_engine.black_scholes(200, 100, 1.0, 0.05, 0.20, True)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert res.price == pytest.approx(intrinsic, abs=1.0)
        assert res.delta > 0.95, "Deep ITM call delta → 1"

    def test_deep_otm_put(self):
        """Deep out-of-money put should be nearly worthless."""
        res = gamma_engine.black_scholes(200, 100, 1.0, 0.05, 0.20, False)
        assert res.price < 0.01
        assert res.delta > -0.05, "Deep OTM put delta → 0"

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) (put-call parity)."""
        S, K, T, r, sigma = 150, 140, 0.5, 0.04, 0.25
        call = gamma_engine.black_scholes(S, K, T, r, sigma, True)
        put = gamma_engine.black_scholes(S, K, T, r, sigma, False)
        parity_diff = call.price - put.price - (S - K * np.exp(-r * T))
        assert abs(parity_diff) < 0.01, f"Put-call parity violated: diff={parity_diff}"
        print(f"✅ Put-Call Parity: C={call.price:.4f}, P={put.price:.4f}, diff={parity_diff:.6f}")

    def test_gamma_same_for_call_and_put(self):
        """Gamma should be identical for calls and puts at the same strike."""
        call = gamma_engine.black_scholes(100, 100, 0.25, 0.05, 0.30, True)
        put = gamma_engine.black_scholes(100, 100, 0.25, 0.05, 0.30, False)
        assert call.gamma == pytest.approx(put.gamma, abs=1e-10)

    def test_edge_cases(self):
        """Zero/negative inputs should not crash."""
        res = gamma_engine.black_scholes(0, 100, 1.0, 0.05, 0.20, True)
        assert res.price == 0.0
        res = gamma_engine.black_scholes(100, 100, 0, 0.05, 0.20, True)
        assert res.price == 0.0


# ── GEX computation tests ────────────────────────────────────────────

class TestGammaExposure:
    """Verify the vectorised GEX calculation."""

    def test_basic_gex_computation(self):
        """Compute GEX for a small synthetic chain."""
        strikes = np.array([90, 95, 100, 105, 110], dtype=np.float64)
        ivs = np.array([0.30, 0.25, 0.20, 0.25, 0.30], dtype=np.float64)
        ois = np.array([1000, 2000, 5000, 3000, 1500], dtype=np.float64)
        is_calls = np.array([1, 1, 1, 0, 0], dtype=np.int32)

        result = gamma_engine.compute_gamma_exposure(
            strikes, ivs, ois, is_calls,
            spot=100.0, rate=0.05, time_to_expiry=30/365,
        )

        assert result.total_gex != 0, "Total GEX should not be zero"
        assert len(result.strikes) == 5
        assert len(result.gex_per_strike) == 5
        assert len(result.types) == 5
        assert result.types[0] == "call"
        assert result.types[3] == "put"
        print(f"\n✅ GEX: total={result.total_gex:.0f}, flip={result.flip_point:.2f}")
        for i in range(5):
            print(f"   K={result.strikes[i]:.0f}  GEX={result.gex_per_strike[i]:.0f}  {result.types[i]}")

    def test_calls_have_positive_gex(self):
        """Pure call chain → total GEX should be positive (dealers long gamma)."""
        strikes = np.array([95, 100, 105], dtype=np.float64)
        ivs = np.array([0.25, 0.20, 0.25], dtype=np.float64)
        ois = np.array([1000, 1000, 1000], dtype=np.float64)
        is_calls = np.array([1, 1, 1], dtype=np.int32)

        result = gamma_engine.compute_gamma_exposure(
            strikes, ivs, ois, is_calls,
            spot=100.0, rate=0.05, time_to_expiry=30/365,
        )
        assert result.total_gex > 0, f"Pure calls should give positive GEX, got {result.total_gex}"

    def test_puts_have_negative_gex(self):
        """Pure put chain → total GEX should be negative."""
        strikes = np.array([95, 100, 105], dtype=np.float64)
        ivs = np.array([0.25, 0.20, 0.25], dtype=np.float64)
        ois = np.array([1000, 1000, 1000], dtype=np.float64)
        is_calls = np.array([0, 0, 0], dtype=np.int32)

        result = gamma_engine.compute_gamma_exposure(
            strikes, ivs, ois, is_calls,
            spot=100.0, rate=0.05, time_to_expiry=30/365,
        )
        assert result.total_gex < 0, f"Pure puts should give negative GEX, got {result.total_gex}"

    def test_mismatched_arrays_raise(self):
        """Mismatched array lengths should raise."""
        with pytest.raises(RuntimeError):
            gamma_engine.compute_gamma_exposure(
                np.array([100.0]),
                np.array([0.2, 0.3]),  # wrong length
                np.array([1000.0]),
                np.array([1], dtype=np.int32),
                100.0, 0.05, 0.08,
            )
