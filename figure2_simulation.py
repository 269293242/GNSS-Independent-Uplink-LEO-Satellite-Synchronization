#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  图2仿真：不同检测方法的性能对比
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

# 图2中的检测方法参数
L_VALUES = [5, 7]         # Method (30) 的L值

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

# ---------- Method (30): 频偏补偿检测器 ----------------------------------------
def method_30_detector(received, reference, N, L):
    """Method (30): 使用频偏补偿的检测器"""
    best_d, best_f = 0, 0
    best_metric = -np.inf
    k = np.arange(N)
    
    # 根据论文定义正确的频偏候选值
    if L == 7:
        # L=7: fc/∆f ∈ {0,±1,±2,±3}
        fc_candidates = [-3, -2, -1, 0, 1, 2, 3]
    elif L == 5:
        # L=5: fc/∆f ∈ {0,±1.4,±2.8}
        fc_candidates = [-2.8, -1.4, 0, 1.4, 2.8]
    
    # 在频偏候选值中搜索
    for fc in fc_candidates:
        # 频偏补偿 (公式30)
        comp_signal = received * np.exp(-1j * 2*np.pi * fc * k / N)
        
        # 计算相关
        rho = correlate(comp_signal, reference)
        metric = np.abs(rho)**2
        
        # 找到最大值
        max_idx = int(np.argmax(metric))
        max_value = metric[max_idx]
        
        if max_value > best_metric:
            best_metric = max_value
            best_d = max_idx
            best_f = fc
    
    return best_d, best_f

# ---------- Method (31): 分段相关检测器 ----------------------------------------
def method_31_detector(received, reference, N):
    """Method (31): 分段相关检测器（按论文公式31精确实现）"""
    best_d = 0
    best_metric = -np.inf
    
    # 分割点：N=9时，(N-1)/2=4, (N+1)/2=5
    mid_point = (N-1) // 2  # 前半段结束点
    
    for d in range(N):
        ref_shifted = np.roll(reference, d)
        
        # 第一部分：k = 0 to (N-1)/2
        rho1 = np.sum(received[:mid_point+1] * np.conj(ref_shifted[:mid_point+1]))
        
        # 第二部分：k = (N+1)/2 to N-1
        rho2 = np.sum(received[mid_point+1:] * np.conj(ref_shifted[mid_point+1:]))
        
        # 论文公式(31): |ρ(d)| = |∑rho1| + |∑rho2|
        # 注意：这里是分别计算模值再相加，不是相加后计算模值
        metric = np.abs(rho1) + np.abs(rho2)
        
        if metric > best_metric:
            best_metric = metric
            best_d = d
    
    return best_d, 0  # 该方法不估计频偏

# ---------- Monte-Carlo 仿真 --------------------------------------------------
def simulate_figure2():
    """复现论文图2的仿真"""
    # 生成SPI-ZC参考序列
    spi_seq, x0_spi, x1_spi = generate_spi_zc(N, u0, u1, ga, gb, gc, gd)
    ref0 = x0_spi[lpp_map(N, ga, gb)]
    ref1 = x1_spi[lpp_map(N, gc, gd)]

    # 存储结果
    results = {}
    results['Method_18'] = []  # Method (18)
    results['Method_30_L5'] = []  # Method (30) with L=5
    results['Method_30_L7'] = []  # Method (30) with L=7
    results['Method_31'] = []  # Method (31)

    print("开始仿真图2...")
    print("检测方法:")
    print("  Method (18): SPI-ZC检测器（闭式解）")
    print("  Method (30): 频偏补偿检测器 (L=5, L=7)")
    print("  Method (31): 分段相关检测器")

    for snr_db in SNR_RANGE:
        success_count = {
            'Method_18': 0,
            'Method_30_L5': 0,
            'Method_30_L7': 0,
            'Method_31': 0
        }

        for _ in range(NUM_TRIALS):
            # 随机真实参数
            t = np.random.randint(0, N)
            f_int = np.random.randint(0, H)
            f_frac = np.random.uniform(-0.5, 0.5)
            f_norm = f_int - (H-1)//2 + f_frac

            # 应用信道
            r = apply_channel(spi_seq, t, f_norm, snr_db,
                             add_phase_noise=ADD_PHASE_NOISE)

            # ---- Method (18): SPI-ZC检测器
            t_hat_18, f_hat_18 = method_18_detector(r, ref0, ref1, u0, u1, ga, gc, N, H)
            if t_hat_18 == t:
                success_count['Method_18'] += 1

            # ---- Method (30): 频偏补偿检测器 (L=5)
            t_hat_30_5, f_hat_30_5 = method_30_detector(r, spi_seq, N, 5)
            if t_hat_30_5 == t:
                success_count['Method_30_L5'] += 1

            # ---- Method (30): 频偏补偿检测器 (L=7)
            t_hat_30_7, f_hat_30_7 = method_30_detector(r, spi_seq, N, 7)
            if t_hat_30_7 == t:
                success_count['Method_30_L7'] += 1

            # ---- Method (31): 分段相关检测器
            t_hat_31, _ = method_31_detector(r, spi_seq, N)
            if t_hat_31 == t:
                success_count['Method_31'] += 1

        # 计算检测概率
        for method in results.keys():
            results[method].append(success_count[method] / NUM_TRIALS)
        
        if snr_db % 2 == 0:
            print(f"SNR {snr_db:>3} dB : ", end="")
            for method in results.keys():
                prob = success_count[method] / NUM_TRIALS
                print(f"{method}={prob:.3f}  ", end="")
            print()

    return results

# ---------- 绘图函数 ----------------------------------------------------------
def plot_figure2(results):
    """绘制图2：四种不同检测方法的性能对比"""
    plt.figure(figsize=(10,7))
    
    # 定义颜色和标记样式（按照论文图2）
    plt.plot(SNR_RANGE, results['Method_30_L7'], 'k*-', 
             label='Method (30), L=7', lw=2, markersize=8)
    plt.plot(SNR_RANGE, results['Method_18'], 'bo-', 
             label='Method (18)', lw=2, markersize=6)
    plt.plot(SNR_RANGE, results['Method_30_L5'], 'r^--', 
             label='Method (30), L=5', lw=2, markersize=6)
    plt.plot(SNR_RANGE, results['Method_31'], 'gs:', 
             label='Method (31)', lw=2, markersize=6)
    
    plt.xlabel('SNR [dB]')
    plt.ylabel('P(τ̂ = τ)')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    plt.xlim(-10, 10)
    plt.title('Probability of timing detection as function of SNR for different\ndetection methods of SPI-ZC sequence')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# ---------- 主程序 ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("📊 图2仿真：不同检测方法的性能对比")
    print("=" * 60)
    
    # 运行图2仿真
    print("\n🔄 运行图2仿真...")
    results = simulate_figure2()
    
    print("\n📊 绘制图2...")
    plot_figure2(results)
    
    # 图2结果表
    print("\n--- 图2检测概率表 ---")
    df_data = {'SNR(dB)': SNR_RANGE}
    for method_name, method_results in results.items():
        df_data[method_name] = method_results
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False, float_format='%.3f'))
     
    print("\n--- 方法说明 ---")
    print("Method (18): SPI-ZC检测器（闭式解）- 论文提出的主要方法")
    print("Method (30): 频偏补偿检测器 - 传统搜索方法")
    print("Method (31): 分段相关检测器 - 分段处理方法")
    print("L=5/7: 频偏候选值的数量")
    
    print("\n✅ 图2仿真完成！") 