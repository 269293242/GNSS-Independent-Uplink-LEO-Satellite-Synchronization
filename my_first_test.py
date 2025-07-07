# SPI-ZC Based Uplink Synchronization Simulation
# 基于文章: "GNSS-Independent Uplink LEO Satellite Synchronization"
# 最终版本 - 重现文章结果

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体和matplotlib参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]

# === 1. ZC序列生成 ===
def generate_zc(N, u):
    """生成Zadoff-Chu序列"""
    k = np.arange(N)
    return np.exp(-1j * np.pi * u * k * (k + 1) / N)

# === 2. 大素数排列 ===
def lpp_interleaver(N, g, b):
    """大素数排列交织器"""
    return np.mod(g * np.arange(N) + b, N).astype(int)

# === 3. SPI-ZC序列生成 ===
def generate_spi_zc(N, u0, u1, g0, b0, g1, b1):
    """生成SPI-ZC序列: y[k] = x_u0[g^(0)[k]] + x_u1[g^(1)[k]]"""
    x0 = generate_zc(N, u0)
    x1 = generate_zc(N, u1)
    
    g0_idx = lpp_interleaver(N, g0, b0)
    g1_idx = lpp_interleaver(N, g1, b1)
    
    # SPI-ZC序列：两个交织后的ZC序列相加
    y = x0[g0_idx] + x1[g1_idx]
    
    return y, x0, x1, g0_idx, g1_idx

# === 4. 现实信道模型 ===
def apply_realistic_channel(signal, delay, freq_offset, snr_db, add_phase_noise=True):
    """应用现实信道效应"""
    N = len(signal)
    k = np.arange(N)
    
    # 1. 应用时延
    delayed_signal = np.roll(signal, delay)
    
    # 2. 应用频偏
    if freq_offset != 0:
        freq_phase = np.exp(1j * 2 * np.pi * freq_offset * k / N)
        delayed_signal = delayed_signal * freq_phase
    
    # 3. 添加相位噪声（可选）
    if add_phase_noise:
        phase_noise_std = 0.05  # 小幅度相位噪声
        phase_noise = phase_noise_std * np.random.randn(N)
        phase_noise_vector = np.exp(1j * phase_noise)
        delayed_signal = delayed_signal * phase_noise_vector
    
    # 4. 添加AWGN噪声
    if snr_db is not None:
        signal_power = np.mean(np.abs(delayed_signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    delayed_signal = delayed_signal + noise
    
    return delayed_signal

# === 5. 相关函数 ===
def correlate_signals(received, reference):
    """计算相关函数"""
    N = len(received)
    correlation = np.zeros(N, dtype=complex)
    
    for delay in range(N):
        r_shifted = np.roll(received, -delay)
        correlation[delay] = np.sum(r_shifted * np.conj(reference))
    
    return correlation

# === 6. SPI-ZC检测器（根据文章算法）===
def spi_zc_detector(received_signal, x0, x1, g0_idx, g1_idx, freq_search_range=None):
    """SPI-ZC检测器 - 实现文章中的联合检测方法"""
    N = len(received_signal)
    k = np.arange(N)
    
    if freq_search_range is None:
        freq_search_range = np.arange(-6, 7)  # 根据文章调整搜索范围
    
    best_correlation = 0
    best_delay = 0
    best_freq_offset = 0
    
    for freq_offset in freq_search_range:
        # 频偏补偿
        freq_compensation = np.exp(-1j * 2 * np.pi * freq_offset * k / N)
        compensated_signal = received_signal * freq_compensation
        
        # 方法1：分别计算两个分量的相关（根据文章公式8-9）
        rho0 = correlate_signals(compensated_signal, x0[g0_idx])
        rho1 = correlate_signals(compensated_signal, x1[g1_idx])
        
        # 联合检测度量
        combined_metric = np.abs(rho0)**2 + np.abs(rho1)**2
        
        # 找到最大值
        max_corr = np.max(combined_metric)
        max_delay = np.argmax(combined_metric)
        
        if max_corr > best_correlation:
            best_correlation = max_corr
            best_delay = max_delay
            best_freq_offset = freq_offset
    
    return best_delay, best_freq_offset, best_correlation

# === 7. 主仿真函数 ===
def main_simulation():
    """主仿真 - 重现文章结果"""
    print("=== SPI-ZC上行同步仿真 - 重现文章结果 ===")
    
    # 仿真参数（根据文章Figure 1和Figure 2）
    N = 127  # 序列长度（质数）
    u0, u1 = 1, 2  # ZC序列根索引
    g0, b0 = 4, 0  # 第一个LPP参数
    g1, b1 = 2, 4  # 第二个LPP参数
    
    # 信道参数
    true_delay = 15
    true_freq_offset = 2  # 标准化频偏
    
    # SNR范围（根据文章图表）
    snr_range = np.arange(-10, 11, 1)  # -10 to 10 dB
    num_trials = 500  # 蒙特卡洛试验次数
    
    print(f"序列长度: {N}")
    print(f"真实时延: {true_delay} 样本")
    print(f"真实频偏: {true_freq_offset}")
    print(f"SNR范围: {snr_range[0]} 到 {snr_range[-1]} dB")
    print(f"试验次数: {num_trials}")
    
    # 生成SPI-ZC序列
    spi_zc_seq, x0, x1, g0_idx, g1_idx = generate_spi_zc(N, u0, u1, g0, b0, g1, b1)
    
    # 计算序列性能指标
    autocorr = correlate_signals(spi_zc_seq, spi_zc_seq)
    autocorr_power = np.abs(autocorr)**2
    peak_value = np.max(autocorr_power)
    peak_idx = np.argmax(autocorr_power)
    sidelobe_values = np.delete(autocorr_power, peak_idx)
    max_sidelobe = np.max(sidelobe_values)
    psr = 10 * np.log10(peak_value / max_sidelobe)
    
    print(f"\n序列性能指标:")
    print(f"峰值旁瓣比 (PSR): {psr:.2f} dB")
    print(f"自相关峰值位置: {peak_idx}")
    
    # 仿真不同的信道条件
    detection_prob_ideal = []  # 理想信道（仅时延+噪声）
    detection_prob_freq_offset = []  # 有频偏的信道
    detection_prob_realistic = []  # 现实信道（频偏+相位噪声+噪声）
    
    print(f"\n开始蒙特卡洛仿真...")
    
    for snr_db in snr_range:
        correct_ideal = 0
        correct_freq_offset = 0
        correct_realistic = 0
        
        for trial in range(num_trials):
            # 1. 理想信道（仅时延+噪声）
            received_ideal = apply_realistic_channel(
                spi_zc_seq, true_delay, 0, snr_db, add_phase_noise=False
            )
            detected_delay_ideal, _, _ = spi_zc_detector(
                received_ideal, x0, x1, g0_idx, g1_idx, freq_search_range=[0]
            )
            if abs(detected_delay_ideal - true_delay) <= 1:
                correct_ideal += 1
            
            # 2. 有频偏的信道
            received_freq = apply_realistic_channel(
                spi_zc_seq, true_delay, true_freq_offset, snr_db, add_phase_noise=False
            )
            detected_delay_freq, _, _ = spi_zc_detector(
                received_freq, x0, x1, g0_idx, g1_idx
            )
            if abs(detected_delay_freq - true_delay) <= 1:
                correct_freq_offset += 1
            
            # 3. 现实信道（频偏+相位噪声+时延变化）
            # 添加小幅度的参数变化以模拟现实条件
            trial_delay = true_delay + np.random.randint(-1, 2)
            trial_freq_offset = true_freq_offset + 0.3 * np.random.randn()
            
            received_realistic = apply_realistic_channel(
                spi_zc_seq, trial_delay, trial_freq_offset, snr_db, add_phase_noise=True
            )
            detected_delay_realistic, _, _ = spi_zc_detector(
                received_realistic, x0, x1, g0_idx, g1_idx
            )
            if abs(detected_delay_realistic - trial_delay) <= 1:
                correct_realistic += 1
        
        # 计算检测概率
        prob_ideal = correct_ideal / num_trials
        prob_freq_offset = correct_freq_offset / num_trials
        prob_realistic = correct_realistic / num_trials
        
        detection_prob_ideal.append(prob_ideal)
        detection_prob_freq_offset.append(prob_freq_offset)
        detection_prob_realistic.append(prob_realistic)
        
        if snr_db % 2 == 0:  # 每2dB打印一次
            print(f"SNR = {snr_db:3d} dB: 理想={prob_ideal:.3f}, 频偏={prob_freq_offset:.3f}, 现实={prob_realistic:.3f}")
    
    return (snr_range, detection_prob_ideal, detection_prob_freq_offset, 
            detection_prob_realistic, autocorr_power, spi_zc_seq)

# === 8. 结果可视化 ===
def plot_results(snr_range, prob_ideal, prob_freq_offset, prob_realistic, autocorr_power, spi_zc_seq):
    """绘制仿真结果"""
    
    # 创建主图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 检测概率曲线（主要结果）
    ax1.plot(snr_range, prob_ideal, 'b-o', linewidth=2, markersize=4, label='理想信道（仅时延+噪声）')
    ax1.plot(snr_range, prob_freq_offset, 'r-s', linewidth=2, markersize=4, label='频偏信道')
    ax1.plot(snr_range, prob_realistic, 'g-^', linewidth=2, markersize=4, label='现实信道（频偏+相位噪声）')
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('检测概率', fontsize=12)
    ax1.set_title('SPI-ZC同步性能（重现文章结果）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(-10, 10)
    
    # 2. 自相关函数
    ax2.plot(autocorr_power / np.max(autocorr_power), 'b-', linewidth=1.5)
    ax2.set_xlabel('时延 (样本)', fontsize=12)
    ax2.set_ylabel('归一化相关幅度', fontsize=12)
    ax2.set_title('SPI-ZC自相关函数', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 127)
    
    # 3. 自相关函数 (dB)
    autocorr_db = 10 * np.log10(autocorr_power / np.max(autocorr_power))
    ax3.plot(autocorr_db, 'r-', linewidth=1.5)
    ax3.set_xlabel('时延 (样本)', fontsize=12)
    ax3.set_ylabel('相关幅度 (dB)', fontsize=12)
    ax3.set_title('SPI-ZC自相关函数 (dB)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-40, 5)
    ax3.set_xlim(0, 127)
    
    # 4. 性能比较条形图
    snr_points = [-5, 0, 5, 10]
    indices = [np.where(snr_range == snr)[0][0] for snr in snr_points]
    
    x_pos = np.arange(len(snr_points))
    width = 0.25
    
    ideal_vals = [prob_ideal[i] for i in indices]
    freq_vals = [prob_freq_offset[i] for i in indices]
    realistic_vals = [prob_realistic[i] for i in indices]
    
    ax4.bar(x_pos - width, ideal_vals, width, label='理想信道', alpha=0.8)
    ax4.bar(x_pos, freq_vals, width, label='频偏信道', alpha=0.8)
    ax4.bar(x_pos + width, realistic_vals, width, label='现实信道', alpha=0.8)
    
    ax4.set_xlabel('SNR (dB)', fontsize=12)
    ax4.set_ylabel('检测概率', fontsize=12)
    ax4.set_title('不同SNR下的性能比较', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(snr_points)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
plt.show()

# === 9. 主程序 ===
if __name__ == "__main__":
    print("SPI-ZC上行同步仿真")
    print("基于文章: GNSS-Independent Uplink LEO Satellite Synchronization")
    print("=" * 70)
    
    # 设置随机种子以获得可重复结果
    np.random.seed(42)
    
    # 运行主仿真
    results = main_simulation()
    snr_range, prob_ideal, prob_freq_offset, prob_realistic, autocorr_power, spi_zc_seq = results
    
    # 绘制结果
    plot_results(snr_range, prob_ideal, prob_freq_offset, prob_realistic, autocorr_power, spi_zc_seq)
    
    # 结果汇总表格
    print(f"\n=== 详细结果表格 ===")
    results_df = pd.DataFrame({
        'SNR (dB)': snr_range,
        '理想信道': prob_ideal,
        '频偏信道': prob_freq_offset,
        '现实信道': prob_realistic
    })
    
    # 只显示部分结果以节省空间
    display_indices = range(0, len(snr_range), 2)  # 每2dB显示一次
    print(results_df.iloc[display_indices].to_string(index=False, float_format='%.3f'))
    
    # 性能总结
    print(f"\n=== 性能总结 ===")
    print(f"在SNR = 0 dB时:")
    idx_0db = np.where(snr_range == 0)[0][0]
    print(f"  理想信道检测概率: {prob_ideal[idx_0db]:.3f}")
    print(f"  频偏信道检测概率: {prob_freq_offset[idx_0db]:.3f}")
    print(f"  现实信道检测概率: {prob_realistic[idx_0db]:.3f}")
    
    print(f"\n在SNR = 5 dB时:")
    idx_5db = np.where(snr_range == 5)[0][0]
    print(f"  理想信道检测概率: {prob_ideal[idx_5db]:.3f}")
    print(f"  频偏信道检测概率: {prob_freq_offset[idx_5db]:.3f}")
    print(f"  现实信道检测概率: {prob_realistic[idx_5db]:.3f}")
    
    print(f"\n仿真完成！")
    print("=" * 70)