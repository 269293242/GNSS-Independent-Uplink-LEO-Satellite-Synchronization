#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  图1仿真：SPI-ZC vs Double-ZC性能对比
#  复现论文："GNSS-Independent Uplink LEO Satellite Synchronization"
#  作者：王书恒 (Shuheng Wang), 哈尔滨工业大学
#  日期：2025年7月7日
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- 参数区 -----------------------------------------------------------
N   = 9                   # 序列长度   (论文 Tab. I)
u0, u1 = 1, 2             # ZC 根索引 (SPI-ZC用)
ga, gb = 4, 0             # SPI-ZC 第 1 组 LPP 参数
gc, gd = 2, 4             # SPI-ZC 第 2 组 LPP 参数
H   = 7                   # 可见整数 CFO 个数 (−3…+3)，式 (11)

# 图1中的不同Δu = u0 - u1 值
DELTA_U_VALUES = [1, 4, 7]
U_PAIRS = [(1, 0), (4, 0), (7, 0)]  # (u0, u1) 对应 Δu = 1, 4, 7

SNR_RANGE   = np.arange(-10, 11, 1)      # dB
NUM_TRIALS  = 1000                       # 试验次数
ADD_PHASE_NOISE = False                  # 论文未考虑相位噪声
# -----------------------------------------------------------------------------

# ---------- 基础函数 ----------------------------------------------------------
def generate_zc(N, u):
    k = np.arange(N)
    return np.exp(-1j * np.pi * u * k * (k + 1) / N)

def lpp_map(N, g, b):
    return (g * np.arange(N) + b) % N

def generate_spi_zc(N, u0, u1, ga, gb, gc, gd):
    x0 = generate_zc(N, u0)
    x1 = generate_zc(N, u1)
    y  = x0[lpp_map(N, ga, gb)] + x1[lpp_map(N, gc, gd)]
    return y, x0, x1

def generate_double_zc(N, u0, u1):
    return generate_zc(N, u0) + generate_zc(N, u1)

def apply_channel(sig, delay, f_norm, snr_db, add_phase_noise=False):
    """
    sig      : baseband sequence (length N)
    delay    : integer sample delay 0…N-1
    f_norm   : normalized CFO  (cycles / sequence-length)
    """
    N = len(sig)
    k = np.arange(N)
    
    # 1. 时延
    out = np.roll(sig, delay)
    
    # 2. 频偏
    out = out * np.exp(1j * 2*np.pi * f_norm * k / N)

    # 3. 相位噪声（可选）
    if add_phase_noise:
        out *= np.exp(1j * np.random.normal(0, 0.05, N))

    # 4. AWGN噪声
    sig_pwr = 1.0
    snr_linear = 10**(snr_db / 10)
    noise_pwr = sig_pwr / snr_linear
    noise_std = np.sqrt(noise_pwr / 2)
    noise = (np.random.randn(N) + 1j*np.random.randn(N)) * noise_std
    
    return out + noise

def correlate(received, reference):
    N = len(received)
    rho = np.zeros(N, dtype=complex)
    for d in range(N):
        rho[d] = np.sum(received * np.conj(np.roll(reference, d)))
    return rho

# ---------- Method (18): SPI-ZC 检测器 ----------------------------------------
def method_18_detector(received, ref0, ref1, u0, u1, ga, gc, N, H):
    """Method (18): SPI-ZC检测器（闭式解）"""
    rho0 = correlate(received, ref0)
    rho1 = correlate(received, ref1)
    d0 = int(np.argmax(np.abs(rho0)))
    d1 = int(np.argmax(np.abs(rho1)))

    try:
        # 论文公式18中的系数a
        a = (u0*ga**2 - u1*gc**2) % N
        
        # 确保a与N互质，才能计算模逆
        if np.gcd(a, N) != 1:
            return d0, 0
        
        inv_a = pow(a, -1, N)
        
        # 公式16: 时延估计
        t_hat = (inv_a * (u0*ga**2*d0 - u1*gc**2*d1)) % N
        
        # 公式17: 整数频偏估计
        f_temp = (u0*ga**2 * (d0 - t_hat)) % N
        f_int_hat = (f_temp + (H-1)//2) % H
        
        return t_hat, f_int_hat
        
    except (ValueError, ZeroDivisionError):
        return d0, 0

# ---------- Double-ZC 检测器 --------------------------------------------------
def double_zc_detector(received, x0, x1, N, H):
    """Double-ZC检测器 - 穷举搜索时延和频偏"""
    best_d, best_f = 0, 0
    best_metric = -np.inf
    k = np.arange(N)

    # 遍历所有可能的整数频偏
    for f_int in range(H):
        f_norm = f_int - (H-1)//2          # −3…+3
        comp = received * np.exp(-1j * 2*np.pi * f_norm * k / N)
        rho0 = correlate(comp, x0)
        rho1 = correlate(comp, x1)
        metric = (np.abs(rho0)**2 + np.abs(rho1)**2)
        idx = int(np.argmax(metric))
        if metric[idx] > best_metric:
            best_metric = metric[idx]
            best_d, best_f = idx, f_int
    return best_d, best_f

# ---------- Monte-Carlo 仿真 --------------------------------------------------
def simulate_figure1():
    """复现论文图1的仿真"""
    # 生成SPI-ZC参考序列
    spi_seq, x0_spi, x1_spi = generate_spi_zc(N, u0, u1, ga, gb, gc, gd)
    ref0 = x0_spi[lpp_map(N, ga, gb)]
    ref1 = x1_spi[lpp_map(N, gc, gd)]

    # 存储结果
    det_prob_spi = []
    det_prob_double = {}
    
    # 为每个Δu值初始化
    for delta_u in DELTA_U_VALUES:
        det_prob_double[delta_u] = []

    print("开始仿真图1...")
    print(f"SPI-ZC: u0={u0}, u1={u1}, Δu={u0-u1}")
    print(f"Double-ZC测试: Δu = {DELTA_U_VALUES}")

    for snr_db in SNR_RANGE:
        ok_spi = 0
        ok_double = {delta_u: 0 for delta_u in DELTA_U_VALUES}

        for _ in range(NUM_TRIALS):
            # 随机真实参数
            t = np.random.randint(0, N)
            f_int = np.random.randint(0, H)
            f_frac = np.random.uniform(-0.5, 0.5)
            f_norm = f_int - (H-1)//2 + f_frac

            # ---- SPI-ZC测试
            r_spi = apply_channel(spi_seq, t, f_norm, snr_db,
                                  add_phase_noise=ADD_PHASE_NOISE)
            t_hat, f_hat = method_18_detector(
                r_spi, ref0, ref1, u0, u1, ga, gc, N, H)
            if t_hat == t:
                ok_spi += 1

            # ---- Double-ZC测试 (不同Δu值)
            for i, delta_u in enumerate(DELTA_U_VALUES):
                u0_test, u1_test = U_PAIRS[i]
                double_seq = generate_double_zc(N, u0_test, u1_test)
                
                r_dz = apply_channel(double_seq, t, f_norm, snr_db,
                                     add_phase_noise=ADD_PHASE_NOISE)
                t_hat2, f_hat2 = double_zc_detector(r_dz, generate_zc(N, u0_test),
                                                    generate_zc(N, u1_test), N, H)
                if t_hat2 == t:
                    ok_double[delta_u] += 1

        # 计算检测概率
        det_prob_spi.append(ok_spi / NUM_TRIALS)
        for delta_u in DELTA_U_VALUES:
            det_prob_double[delta_u].append(ok_double[delta_u] / NUM_TRIALS)
        
        if snr_db % 2 == 0:
            print(f"SNR {snr_db:>3} dB : SPI-ZC={ok_spi/NUM_TRIALS:.3f}  ", end="")
            for delta_u in DELTA_U_VALUES:
                print(f"Δu={delta_u}={ok_double[delta_u]/NUM_TRIALS:.3f}  ", end="")
            print()

    return det_prob_spi, det_prob_double

# ---------- 绘图函数 ----------------------------------------------------------
def plot_figure1(spi_prob, double_prob_dict):
    """绘制图1：SPI-ZC vs 不同Δu值的Double-ZC对比"""
    plt.figure(figsize=(10,6))
    
    # 绘制SPI-ZC曲线
    plt.plot(SNR_RANGE, spi_prob, 'bo-', label='SPI-ZC', lw=2, markersize=6)
    
    # 绘制不同Δu值的Double-ZC曲线
    colors = ['rs--', 'g^:', 'md-.']
    for i, delta_u in enumerate(DELTA_U_VALUES):
        plt.plot(SNR_RANGE, double_prob_dict[delta_u], colors[i], 
                label=f'Seq. From [14], Δu={delta_u}', lw=2, markersize=6)
    
    plt.xlabel('SNR [dB]')
    plt.ylabel('P(τ̂ = τ, f̂ = f)')
    plt.grid(alpha=0.3)
    plt.ylim(-0.02, 1.02)
    plt.title('Probability of detection as function of SNR for N = 9')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- 主程序 ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("📊 图1仿真：SPI-ZC vs Double-ZC性能对比")
    print("=" * 60)
    
    # 运行图1仿真
    print("\n🔄 运行图1仿真...")
    spi_prob, double_prob_dict = simulate_figure1()
    
    print("\n📊 绘制图1...")
    plot_figure1(spi_prob, double_prob_dict)
    
    # 图1结果表
    print("\n--- 图1检测概率表 ---")
    df_data = {'SNR(dB)': SNR_RANGE, 'SPI-ZC': spi_prob}
    for delta_u in DELTA_U_VALUES:
        df_data[f'Double-ZC(Δu={delta_u})'] = double_prob_dict[delta_u]
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False, float_format='%.3f'))
    
    print("\n--- 方法说明 ---")
    print("SPI-ZC: 论文提出的超级互质ZC序列")
    print("Double-ZC: 传统的双ZC序列叠加方法")
    print("Δu = u0 - u1: ZC序列根索引差值")
    
    print("\n✅ 图1仿真完成！") 