/*
 * Gamma Exposure (GEX) Engine — high-performance Black-Scholes + GEX
 *
 * Computes dealer gamma exposure across an options chain by:
 *   1. Pricing each option via Black-Scholes to get Gamma
 *   2. Multiplying gamma × open_interest × 100 × spot²  to get $ GEX per strike
 *   3. Flipping sign for puts (dealers are short when retail is long)
 *   4. Finding the "GEX flip point" where aggregate gamma changes sign
 *
 * Exposed to Python via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ─── Normal CDF (Abramowitz & Stegun approximation) ─────────────────

static double norm_cdf(double x) {
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::fabs(x);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * std::exp(-x*x/2.0);
    return 0.5 * (1.0 + sign * y);
}

static double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// ─── Black-Scholes Greeks ───────────────────────────────────────────

struct BSResult {
    double price;
    double delta;
    double gamma;
};

/*
 * S     = spot price
 * K     = strike
 * T     = time to expiry (years)
 * r     = risk-free rate
 * sigma = implied volatility
 * is_call = true for calls, false for puts
 */
static BSResult black_scholes(double S, double K, double T, double r,
                               double sigma, bool is_call) {
    if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
        return {0.0, 0.0, 0.0};
    }

    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    BSResult res;
    if (is_call) {
        res.price = S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        res.delta = norm_cdf(d1);
    } else {
        res.price = K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        res.delta = norm_cdf(d1) - 1.0;
    }

    // Gamma is same for calls and puts
    res.gamma = norm_pdf(d1) / (S * sigma * std::sqrt(T));
    return res;
}

// ─── Vectorised GEX computation ─────────────────────────────────────

struct GEXResult {
    double total_gex;
    double flip_point;
    std::vector<double> strikes;
    std::vector<double> gex_per_strike;
    std::vector<std::string> types;
};

/*
 * Compute Gamma Exposure for an entire options chain.
 *
 * Parameters (all same-length vectors):
 *   strikes         — strike prices
 *   implied_vols    — implied volatility per contract
 *   open_interests  — open interest per contract
 *   is_calls        — 1 for call, 0 for put
 *   spot            — underlying spot price  (scalar)
 *   rate            — risk-free rate          (scalar, e.g. 0.05)
 *   time_to_expiry  — years to expiry         (scalar)
 */
static GEXResult compute_gamma_exposure(
        py::array_t<double> strikes,
        py::array_t<double> implied_vols,
        py::array_t<double> open_interests,
        py::array_t<int>    is_calls,
        double spot,
        double rate,
        double time_to_expiry)
{
    auto K   = strikes.unchecked<1>();
    auto iv  = implied_vols.unchecked<1>();
    auto oi  = open_interests.unchecked<1>();
    auto flg = is_calls.unchecked<1>();

    ssize_t n = K.shape(0);
    if (iv.shape(0) != n || oi.shape(0) != n || flg.shape(0) != n) {
        throw std::runtime_error("All input arrays must have the same length");
    }

    GEXResult result;
    result.total_gex = 0.0;
    result.flip_point = -1.0;
    result.strikes.resize(n);
    result.gex_per_strike.resize(n);
    result.types.resize(n);

    // Running sum to detect sign flip
    double running_gex = 0.0;
    double prev_running = 0.0;
    double prev_strike = 0.0;

    for (ssize_t i = 0; i < n; ++i) {
        bool is_call = (flg(i) == 1);
        BSResult bs = black_scholes(spot, K(i), time_to_expiry, rate,
                                     iv(i), is_call);

        // Dealer GEX per strike:
        //   gamma × OI × 100 (shares per contract) × spot²
        // For puts, dealers are typically short, so flip sign.
        double contract_gex = bs.gamma * oi(i) * 100.0 * spot * spot;
        if (!is_call) {
            contract_gex = -contract_gex;
        }

        result.strikes[i] = K(i);
        result.gex_per_strike[i] = contract_gex;
        result.types[i] = is_call ? "call" : "put";
        result.total_gex += contract_gex;

        // Detect flip point
        prev_running = running_gex;
        running_gex += contract_gex;
        if (i > 0 && prev_running * running_gex < 0) {
            // Linear interpolation between prev and current strike
            double w = std::fabs(prev_running) /
                       (std::fabs(prev_running) + std::fabs(running_gex));
            result.flip_point = prev_strike + w * (K(i) - prev_strike);
        }
        prev_strike = K(i);
    }

    return result;
}

// ─── Pybind11 module ────────────────────────────────────────────────

PYBIND11_MODULE(gamma_engine, m) {
    m.doc() = "High-performance Gamma Exposure engine (Black-Scholes + GEX)";

    py::class_<BSResult>(m, "BSResult")
        .def_readonly("price", &BSResult::price)
        .def_readonly("delta", &BSResult::delta)
        .def_readonly("gamma", &BSResult::gamma);

    py::class_<GEXResult>(m, "GEXResult")
        .def_readonly("total_gex",        &GEXResult::total_gex)
        .def_readonly("flip_point",       &GEXResult::flip_point)
        .def_readonly("strikes",          &GEXResult::strikes)
        .def_readonly("gex_per_strike",   &GEXResult::gex_per_strike)
        .def_readonly("types",            &GEXResult::types);

    m.def("black_scholes", &black_scholes,
          py::arg("spot"), py::arg("strike"), py::arg("time_to_expiry"),
          py::arg("rate"), py::arg("sigma"), py::arg("is_call"),
          "Compute Black-Scholes price, delta, and gamma.");

    m.def("compute_gamma_exposure", &compute_gamma_exposure,
          py::arg("strikes"), py::arg("implied_vols"),
          py::arg("open_interests"), py::arg("is_calls"),
          py::arg("spot"), py::arg("rate"), py::arg("time_to_expiry"),
          "Compute Gamma Exposure across an options chain.");
}
