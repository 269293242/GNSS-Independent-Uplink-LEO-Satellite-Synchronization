#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 参数设置
N = 9
u0, u1 = 1, 2
ga, gb = 4, 0
gc, gd = 2, 4
H = 7

L_VALUES = [5, 7]

SNR_RANGE = np.arange(-10, 11, 1)
NUM_TRIALS = 1000
ADD_PHASE_NOISE = False


def generate_zc(N, u):
    k = np.arange(N)
    return np.exp(-1j * np.pi * u * k * (k + 1) / N)


def lpp_map(N, g, b):
    return (g * np.arange(N) + b) % N


def generate_spi_zc(N, u0, u1, ga, gb, gc, gd):
    x0 = generate_zc(N, u0)
    x1 = generate_zc(N, u1)
    y = x0[lpp_map(N, ga, gb)] + x1[lpp_map(N, gc, gd)]
    return y, x0, x1


def apply_channel(sig, delay, f_norm, snr_db, add_phase_noise=False):
    """
    delay: integer timing offset
    f_norm: normalized CFO
    """
    N = len(sig)
    k = np.arange(N)

    out = np.roll(sig, delay)
    out = out * np.exp(1j * 2 * np.pi * f_norm * k / N)

    if add_phase_noise:
        out *= np.exp(1j * np.random.normal(0, 0.05, N))

    snr_linear = 10 ** (snr_db / 10)
    noise_pwr = 1.0 / snr_linear
    noise_std = np.sqrt(noise_pwr / 2)
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std

    return out + noise


def correlate(received, reference):
    N = len(received)
    rho = np.zeros(N, dtype=complex)
    for d in range(N):
        rho[d] = np.sum(received * np.conj(np.roll(reference, d)))
    return rho


def method_18_detector(received, ref0, ref1, u0, u1, ga, gc, N, H):
    rho0 = correlate(received, ref0)
    rho1 = correlate(received, ref1)
    d0 = int(np.argmax(np.abs(rho0)))
    d1 = int(np.argmax(np.abs(rho1)))

    try:
        a = (u0 * ga**2 - u1 * gc**2) % N

        if np.gcd(a, N) != 1:
            return d0, 0

        inv_a = pow(a, -1, N)

        t_hat = (inv_a * (u0 * ga**2 * d0 - u1 * gc**2 * d1)) % N
        f_temp = (u0 * ga**2 * (d0 - t_hat)) % N
        f_int_hat = (f_temp + (H - 1) // 2) % H

        return t_hat, f_int_hat

    except (ValueError, ZeroDivisionError):
        return d0, 0


def method_30_detector(received, reference, N, L):
    best_d, best_f = 0, 0
    best_metric = -np.inf
    k = np.arange(N)

    if L == 7:
        fc_candidates = [-3, -2, -1, 0, 1, 2, 3]
    elif L == 5:
        fc_candidates = [-2.8, -1.4, 0, 1.4, 2.8]
    else:
        raise ValueError("Unsupported L value.")

    for fc in fc_candidates:
        comp_signal = received * np.exp(-1j * 2 * np.pi * fc * k / N)
        rho = correlate(comp_signal, reference)
        metric = np.abs(rho) ** 2

        max_idx = int(np.argmax(metric))
        max_value = metric[max_idx]

        if max_value > best_metric:
            best_metric = max_value
            best_d = max_idx
            best_f = fc

    return best_d, best_f


def method_31_detector(received, reference, N):
    best_d = 0
    best_metric = -np.inf

    mid_point = (N - 1) // 2

    for d in range(N):
        ref_shifted = np.roll(reference, d)

        rho1 = np.sum(received[:mid_point + 1] * np.conj(ref_shifted[:mid_point + 1]))
        rho2 = np.sum(received[mid_point + 1:] * np.conj(ref_shifted[mid_point + 1:]))

        metric = np.abs(rho1) + np.abs(rho2)

        if metric > best_metric:
            best_metric = metric
            best_d = d

    return best_d, 0


def simulate_figure2():
    spi_seq, x0_spi, x1_spi = generate_spi_zc(N, u0, u1, ga, gb, gc, gd)
    ref0 = x0_spi[lpp_map(N, ga, gb)]
    ref1 = x1_spi[lpp_map(N, gc, gd)]

    results = {
        'Method_18': [],
        'Method_30_L5': [],
        'Method_30_L7': [],
        'Method_31': []
    }

    print("Start simulation for Fig. 2...")

    for snr_db in SNR_RANGE:
        success_count = {
            'Method_18': 0,
            'Method_30_L5': 0,
            'Method_30_L7': 0,
            'Method_31': 0
        }

        for _ in range(NUM_TRIALS):
            t = np.random.randint(0, N)
            f_int = np.random.randint(0, H)
            f_frac = np.random.uniform(-0.5, 0.5)
            f_norm = f_int - (H - 1) // 2 + f_frac

            r = apply_channel(
                spi_seq, t, f_norm, snr_db, add_phase_noise=ADD_PHASE_NOISE
            )

            t_hat_18, _ = method_18_detector(r, ref0, ref1, u0, u1, ga, gc, N, H)
            if t_hat_18 == t:
                success_count['Method_18'] += 1

            t_hat_30_5, _ = method_30_detector(r, spi_seq, N, 5)
            if t_hat_30_5 == t:
                success_count['Method_30_L5'] += 1

            t_hat_30_7, _ = method_30_detector(r, spi_seq, N, 7)
            if t_hat_30_7 == t:
                success_count['Method_30_L7'] += 1

            t_hat_31, _ = method_31_detector(r, spi_seq, N)
            if t_hat_31 == t:
                success_count['Method_31'] += 1

        for method in results:
            results[method].append(success_count[method] / NUM_TRIALS)

        if snr_db % 2 == 0:
            print(f"SNR {snr_db:>3} dB : ", end="")
            for method in results:
                prob = success_count[method] / NUM_TRIALS
                print(f"{method}={prob:.3f}  ", end="")
            print()

    return results


def plot_figure2(results):
    plt.figure(figsize=(10, 7))

    plt.plot(
        SNR_RANGE, results['Method_30_L7'],
        'k*-', label='Method (30), L=7', lw=2, markersize=8
    )
    plt.plot(
        SNR_RANGE, results['Method_18'],
        'bo-', label='Method (18)', lw=2, markersize=6
    )
    plt.plot(
        SNR_RANGE, results['Method_30_L5'],
        'r^--', label='Method (30), L=5', lw=2, markersize=6
    )
    plt.plot(
        SNR_RANGE, results['Method_31'],
        'gs:', label='Method (31)', lw=2, markersize=6
    )

    plt.xlabel('SNR [dB]')
    plt.ylabel('P(τ̂ = τ)')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    plt.xlim(-10, 10)
    plt.title('Probability of timing detection as function of SNR for different\n'
              'detection methods of SPI-ZC sequence')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Fig. 2 Simulation: Comparison of Detection Methods")
    print("=" * 60)

    results = simulate_figure2()

    print("\nPlotting Fig. 2...")
    plot_figure2(results)

    print("\n--- Detection probability table ---")
    df_data = {'SNR(dB)': SNR_RANGE}
    for method_name, method_results in results.items():
        df_data[method_name] = method_results

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False, float_format='%.3f'))
