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

DELTA_U_VALUES = [1, 4, 7]
U_PAIRS = [(1, 0), (4, 0), (7, 0)]

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


def generate_double_zc(N, u0, u1):
    return generate_zc(N, u0) + generate_zc(N, u1)


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


def double_zc_detector(received, x0, x1, N, H):
    best_d, best_f = 0, 0
    best_metric = -np.inf
    k = np.arange(N)

    for f_int in range(H):
        f_norm = f_int - (H - 1) // 2
        comp = received * np.exp(-1j * 2 * np.pi * f_norm * k / N)
        rho0 = correlate(comp, x0)
        rho1 = correlate(comp, x1)
        metric = np.abs(rho0) ** 2 + np.abs(rho1) ** 2
        idx = int(np.argmax(metric))

        if metric[idx] > best_metric:
            best_metric = metric[idx]
            best_d, best_f = idx, f_int

    return best_d, best_f


def simulate_figure1():
    spi_seq, x0_spi, x1_spi = generate_spi_zc(N, u0, u1, ga, gb, gc, gd)
    ref0 = x0_spi[lpp_map(N, ga, gb)]
    ref1 = x1_spi[lpp_map(N, gc, gd)]

    det_prob_spi = []
    det_prob_double = {delta_u: [] for delta_u in DELTA_U_VALUES}

    print("Start simulation for Fig. 1...")
    print(f"SPI-ZC: u0={u0}, u1={u1}, Δu={u0-u1}")
    print(f"Double-ZC cases: Δu = {DELTA_U_VALUES}")

    for snr_db in SNR_RANGE:
        ok_spi = 0
        ok_double = {delta_u: 0 for delta_u in DELTA_U_VALUES}

        for _ in range(NUM_TRIALS):
            t = np.random.randint(0, N)
            f_int = np.random.randint(0, H)
            f_frac = np.random.uniform(-0.5, 0.5)
            f_norm = f_int - (H - 1) // 2 + f_frac

            r_spi = apply_channel(
                spi_seq, t, f_norm, snr_db, add_phase_noise=ADD_PHASE_NOISE
            )
            t_hat, _ = method_18_detector(r_spi, ref0, ref1, u0, u1, ga, gc, N, H)
            if t_hat == t:
                ok_spi += 1

            for i, delta_u in enumerate(DELTA_U_VALUES):
                u0_test, u1_test = U_PAIRS[i]
                double_seq = generate_double_zc(N, u0_test, u1_test)

                r_dz = apply_channel(
                    double_seq, t, f_norm, snr_db, add_phase_noise=ADD_PHASE_NOISE
                )
                t_hat2, _ = double_zc_detector(
                    r_dz,
                    generate_zc(N, u0_test),
                    generate_zc(N, u1_test),
                    N,
                    H
                )
                if t_hat2 == t:
                    ok_double[delta_u] += 1

        det_prob_spi.append(ok_spi / NUM_TRIALS)
        for delta_u in DELTA_U_VALUES:
            det_prob_double[delta_u].append(ok_double[delta_u] / NUM_TRIALS)

        if snr_db % 2 == 0:
            print(f"SNR {snr_db:>3} dB : SPI-ZC={ok_spi/NUM_TRIALS:.3f}  ", end="")
            for delta_u in DELTA_U_VALUES:
                print(f"Δu={delta_u}={ok_double[delta_u]/NUM_TRIALS:.3f}  ", end="")
            print()

    return det_prob_spi, det_prob_double


def plot_figure1(spi_prob, double_prob_dict):
    plt.figure(figsize=(10, 6))

    plt.plot(SNR_RANGE, spi_prob, 'bo-', label='SPI-ZC', lw=2, markersize=6)

    styles = ['rs--', 'g^:', 'md-.']
    for i, delta_u in enumerate(DELTA_U_VALUES):
        plt.plot(
            SNR_RANGE,
            double_prob_dict[delta_u],
            styles[i],
            label=f'Seq. From [14], Δu={delta_u}',
            lw=2,
            markersize=6
        )

    plt.xlabel('SNR [dB]')
    plt.ylabel('P(τ̂ = τ, f̂ = f)')
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.title('Probability of detection as function of SNR for N = 9')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Fig. 1 Simulation: SPI-ZC vs Double-ZC")
    print("=" * 60)

    spi_prob, double_prob_dict = simulate_figure1()

    print("\nPlotting Fig. 1...")
    plot_figure1(spi_prob, double_prob_dict)

    print("\n--- Detection probability table ---")
    df_data = {'SNR(dB)': SNR_RANGE, 'SPI-ZC': spi_prob}
    for delta_u in DELTA_U_VALUES:
        df_data[f'Double-ZC(Δu={delta_u})'] = double_prob_dict[delta_u]

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False, float_format='%.3f'))
