import msprime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import random
import pandas as pd
from scipy.special import softmax
import os
import json
import multiprocessing
import sys
from io import StringIO

def _generate_one_sample_for_multiprocessing(args):
    """Helper function for multiprocessing to allow seeding each worker."""
    # Unpack all arguments
    model_type, complexity, sequence_length, num_bins, rate_variation, add_variable_rates, seed = args
    
    # Each worker gets its own simulator with a unique seed.
    # Suppress the "initialized with seed" print message from workers.
    original_stdout = sys.stdout
    sys.stdout = captured_stdout = StringIO()
    try:
        # Create a new simulator instance in each worker process
        simulator = DataSimulator(seed=seed)
        # Generate one sample
        sample = simulator.generate_single_sample(
            model_type=model_type,
            complexity=complexity,
            sequence_length=sequence_length,
            num_bins=num_bins,
            rate_variation=rate_variation,
            add_variable_rates=add_variable_rates
        )
    finally:
        sys.stdout = original_stdout
    return sample

class DataSimulator:
    """
    一个用于生成复杂人口统计学历史和相关基因组序列的模拟器。
    """
    
    def __init__(self, seed=None):
        """
        初始化模拟器。
        
        参数:
            seed (int, optional): 随机种子。如果未提供，将随机生成一个。
        """
        if seed is None:
            seed = random.randint(1, 10000)
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"DataSimulator initialized with seed: {self.seed}")

    def _generate_demographic_model(self, random_params=True, complexity='high'):
        """
        生成随机的人口统计学模型
        """
        if random_params:
            if complexity == 'low':
                ne_range, num_events_range, time_interval_range, size_change_range = (1000, 30000), (3, 7), (100, 6000), (0.1, 6.0)
            elif complexity == 'medium':
                ne_range, num_events_range, time_interval_range, size_change_range = (500, 80000), (6, 15), (50, 10000), (0.05, 15.0)
            elif complexity == 'high':
                ne_range, num_events_range, time_interval_range, size_change_range = (100, 150000), (12, 25), (20, 15000), (0.01, 30.0)
            elif complexity == 'extreme':
                ne_range, num_events_range, time_interval_range, size_change_range = (50, 300000), (20, 50), (10, 20000), (0.005, 80.0)
            else:
                ne_range, num_events_range, time_interval_range, size_change_range = (500, 50000), (5, 10), (50, 8000), (0.05, 10.0)

            Ne = random.randint(*ne_range)
            num_events = random.randint(*num_events_range)
            demographic_events = []
            current_time = 0
            
            add_oscillations = random.random() < 0.2
            oscillation_period = random.randint(200, 2000) if add_oscillations else 0
            oscillation_amplitude = random.uniform(0.2, 1.2) if add_oscillations else 0

            for i in range(num_events):
                time_interval = random.randint(*time_interval_range)
                current_time += time_interval
                
                size_change = np.random.uniform(*size_change_range)
                if random.random() < 0.3:
                    size_change = random.choice([0.005, 0.01, 0.05, 20.0, 40.0, 60.0])

                if random.random() < 0.4:
                    base_size = Ne * size_change
                    for j in range(1, random.randint(2, 6)):
                        sub_time = current_time + j * time_interval / 5
                        decay_rate = np.random.uniform(0.6, 1.4)
                        sub_size = base_size * (decay_rate ** j)
                        demographic_events.append(msprime.PopulationParametersChange(time=sub_time, initial_size=sub_size, population_id=0))

                demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * size_change, population_id=0))

                if add_oscillations and i < num_events - 1:
                    base_size = Ne * size_change
                    num_oscillations = random.randint(2, 7)
                    for j in range(1, num_oscillations + 1):
                        oscillation_time = current_time + j * (oscillation_period / num_oscillations)
                        oscillation_factor = 1 + oscillation_amplitude * np.sin(j * np.pi / num_oscillations)
                        demographic_events.append(msprime.PopulationParametersChange(time=oscillation_time, initial_size=base_size * oscillation_factor, population_id=0))

            random.shuffle(demographic_events)
            demographic_events.sort(key=lambda x: x.time)
            return demographic_events, Ne
        else:
            Ne = 10000
            demographic_events = [
                msprime.PopulationParametersChange(time=10000, initial_size=1000, population_id=0),
                msprime.PopulationParametersChange(time=20000, initial_size=15000, population_id=0),
            ]
            return demographic_events, Ne

    def _simulate_sequence(self, demographic_events, Ne, sequence_length=1000000, sample_size=2, mutation_rate=1.5e-8, recombination_rate=1.2e-8, rate_variation='high', add_variable_rates=True):
        """
        使用msprime模拟DNA序列
        """
        demography = msprime.Demography(
            populations=[msprime.Population(initial_size=Ne)],
            events=demographic_events
        )

        if rate_variation == 'low':
            rate_range = (0.8, 1.2)
            hotspot_intensity = (1.5, 3.0)
            coldspot_intensity = (0.3, 0.8)
        elif rate_variation == 'medium':
            rate_range = (0.5, 2.0)
            hotspot_intensity = (2.0, 5.0)
            coldspot_intensity = (0.2, 0.6)
        elif rate_variation == 'high':
            rate_range = (0.2, 5.0)
            hotspot_intensity = (3.0, 10.0)
            coldspot_intensity = (0.05, 0.4)
        elif rate_variation == 'extreme':
            rate_range = (0.1, 10.0)
            hotspot_intensity = (5.0, 20.0)
            coldspot_intensity = (0.01, 0.3)
        else:
            rate_range = (0.5, 2.0)
            hotspot_intensity = (2.0, 5.0)
            coldspot_intensity = (0.2, 0.6)

        mut_rate_variation = np.random.uniform(*rate_range)
        rec_rate_variation = np.random.uniform(*rate_range)
        mutation_rate *= mut_rate_variation
        recombination_rate *= rec_rate_variation

        # 是否使用率变化映射
        use_rate_maps = add_variable_rates and random.random() < 0.5

        # 变异率地图
        if use_rate_maps:
            # 重组率地图
            positions = [0]
            rec_rates = []
            
            # 更复杂的分段设计
            if random.random() < 0.3:
                # 不均匀段长度的分段
                num_segments = random.randint(5, 15)
                current_pos = 0
                remaining_length = sequence_length
                
                for i in range(num_segments - 1):
                    # 剩余长度的某个比例
                    segment_fraction = random.uniform(0.05, 0.2)
                    segment_length = int(remaining_length * segment_fraction)
                    current_pos += segment_length
                    positions.append(current_pos)
                    remaining_length -= segment_length
                    
                    # 确定该段的重组率
                    if random.random() < 0.2:  # 热点
                        rec_rates.append(recombination_rate * random.uniform(*hotspot_intensity))
                    elif random.random() < 0.3:  # 冷点
                        rec_rates.append(recombination_rate * random.uniform(*coldspot_intensity))
                    else:  # 普通区域
                        rec_rates.append(recombination_rate * random.uniform(0.8, 1.5))
                
                # 添加最后一个位置
                positions.append(sequence_length)
                if random.random() < 0.2:
                    rec_rates.append(recombination_rate * random.uniform(*hotspot_intensity))
                elif random.random() < 0.3:
                    rec_rates.append(recombination_rate * random.uniform(*coldspot_intensity))
                else:
                    rec_rates.append(recombination_rate * random.uniform(0.8, 1.5))
            else:
                # 等分段的设计
                num_segments = random.randint(5, 15)
                segment_length = sequence_length / num_segments
    
                for i in range(num_segments):
                    # 热点区域
                    if random.random() < 0.2:
                        rec_rates.append(recombination_rate * random.uniform(*hotspot_intensity))
                    # 冷点区域
                    elif random.random() < 0.3:
                        rec_rates.append(recombination_rate * random.uniform(*coldspot_intensity))
                    # 普通区域
                    else:
                        rec_rates.append(recombination_rate * random.uniform(0.8, 1.5))
                    positions.append(min(int((i + 1) * segment_length), sequence_length))
            
            if positions[-1] != sequence_length:
                positions[-1] = sequence_length

            # 创建重组率地图和突变率地图
            recomb_map = msprime.RateMap(position=positions, rate=rec_rates)
            
            # 突变率地图
            # 可能使用不同的分段方式
            if random.random() < 0.5:
                mut_positions = positions.copy()
                mut_rates = []
                
                for i in range(len(positions) - 1):
                    if random.random() < 0.2:  # 突变热点
                        mut_rates.append(mutation_rate * random.uniform(*hotspot_intensity))
                    elif random.random() < 0.3:  # 突变冷点
                        mut_rates.append(mutation_rate * random.uniform(*coldspot_intensity))
                    else:  # 普通区域
                        mut_rates.append(mutation_rate * random.uniform(0.8, 1.5))
            else:
                # 使用完全不同的分段
                mut_positions = [0]
                mut_rates = []
                num_mut_segments = random.randint(4, 12)
                mut_segment_length = sequence_length / num_mut_segments
                
                for i in range(num_mut_segments):
                    if random.random() < 0.2:
                        mut_rates.append(mutation_rate * random.uniform(*hotspot_intensity))
                    elif random.random() < 0.3:
                        mut_rates.append(mutation_rate * random.uniform(*coldspot_intensity))
                    else:
                        mut_rates.append(mutation_rate * random.uniform(0.8, 1.5))
                    mut_positions.append(min(int((i + 1) * mut_segment_length), sequence_length))
                
                if mut_positions[-1] != sequence_length:
                    mut_positions[-1] = sequence_length
            
            mutation_map = msprime.RateMap(position=mut_positions, rate=mut_rates)
            
            # 模拟序列
            ts = msprime.sim_ancestry(samples=sample_size, demography=demography, recombination_rate=recomb_map, random_seed=random.randint(1, 10000))
            ts = msprime.sim_mutations(ts, rate=mutation_map, random_seed=random.randint(1, 10000))
        else:
            ts = msprime.sim_ancestry(samples=sample_size, demography=demography, sequence_length=sequence_length, recombination_rate=recombination_rate, random_seed=random.randint(1, 10000))
            ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random.randint(1, 10000))
        
        return ts

    def _extract_binary_sequence(self, ts, sequence_length):
        binary_seq = np.zeros(sequence_length, dtype=np.int8)
        for variant in ts.variants():
            pos = int(variant.site.position)
            if pos < sequence_length and variant.genotypes[0] != variant.genotypes[1]:
                binary_seq[pos] = 1
        return binary_seq

    def _calculate_tmrca_histogram(self, ts, num_bins=20, max_time=None):
        positions = []
        tmrca_values = []
        for tree in ts.trees():
            tmrca = tree.tmrca(0, 1)
            positions.append(tree.interval[0])
            tmrca_values.append(tmrca)
        
        if max_time is None:
            max_time = max(tmrca_values) * 1.1
        
        bin_edges = np.linspace(0, max_time, num_bins + 1)
        hist, _ = np.histogram(tmrca_values, bins=bin_edges, weights=np.diff(positions + [ts.sequence_length]))
        return hist, bin_edges, tmrca_values

    def _normalize_histogram(self, hist):
        return softmax(hist)

    def _generate_custom_demographic_model(self, model_type="bottleneck", complexity='high'):
        """
        生成自定义人口统计学模型
        """
        if complexity == 'low':
            num_events, time_scale, time_span = 4, 1000, 5000
        elif complexity == 'medium':
            num_events, time_scale, time_span = 8, 800, 10000
        elif complexity == 'high':
            num_events, time_scale, time_span = 15, 500, 20000
        elif complexity == 'extreme':
            num_events, time_scale, time_span = 30, 300, 40000
        else:
            num_events, time_scale, time_span = 8, 800, 10000

        demographic_events = []
        Ne = random.randint(5000, 25000)

        if model_type == "bottleneck":
            bottleneck_time = random.randint(500, 5000)
            recovery_time = bottleneck_time + random.randint(500, 5000)
            bottleneck_severity = random.uniform(0.01, 0.3)
            recovery_factor = random.uniform(1.1, 3.0)
            
            # 多个瓶颈事件
            if random.random() < 0.4:
                demographic_events = []
                num_bottlenecks = random.randint(2, 4)
                current_time = 0
                for i in range(num_bottlenecks):
                    bottleneck_start = current_time + random.randint(1000, 3000)
                    bottleneck_duration = random.randint(500, 2000)
                    bottleneck_depth = random.uniform(0.01, 0.3)
                    recovery_magnitude = random.uniform(1.1, 2.5)
                    
                    demographic_events.append(msprime.PopulationParametersChange(time=bottleneck_start, initial_size=Ne * bottleneck_depth, population_id=0))
                    demographic_events.append(msprime.PopulationParametersChange(time=bottleneck_start + bottleneck_duration, initial_size=Ne * recovery_magnitude, population_id=0))
                    
                    current_time = bottleneck_start + bottleneck_duration + random.randint(1000, 4000)
            else:
                demographic_events = [
                    msprime.PopulationParametersChange(time=bottleneck_time, initial_size=Ne * bottleneck_severity, population_id=0),
                    msprime.PopulationParametersChange(time=recovery_time, initial_size=Ne * recovery_factor, population_id=0),
                ]
        elif model_type == "expansion":
            Ne = random.randint(5000, 30000)
            current_size = Ne
            
            # 爆发式扩张
            if random.random() < 0.3:
                expansion_time = random.randint(1000, 5000)
                expansion_factor = random.uniform(3.0, 10.0)
                demographic_events = [
                    msprime.PopulationParametersChange(time=expansion_time, initial_size=current_size / expansion_factor, population_id=0)
                ]
                
                # 添加前期波动
                if random.random() < 0.5:
                    pre_events = random.randint(2, 5)
                    for i in range(pre_events):
                        pre_time = random.randint(int(expansion_time * 1.2), int(expansion_time * 2.5))
                        pre_factor = random.uniform(0.3, 0.9)
                        demographic_events.append(msprime.PopulationParametersChange(time=pre_time, initial_size=current_size / expansion_factor * pre_factor, population_id=0))
            else:
                for i in range(num_events):
                    time_point = (i + 1) * time_scale
                    growth_factor = 0.6 ** (1 + i/2)
                    new_size = current_size * growth_factor
                    demographic_events.append(msprime.PopulationParametersChange(time=time_point, initial_size=new_size, population_id=0))
                    current_size = new_size
        elif model_type == "decline":
            Ne = random.randint(1000, 8000)
            current_size = Ne
            
            # 急剧下降
            if random.random() < 0.3:
                decline_time = random.randint(1000, 5000)
                decline_factor = random.uniform(5.0, 20.0)
                demographic_events = [
                    msprime.PopulationParametersChange(time=decline_time, initial_size=current_size * decline_factor, population_id=0)
                ]
                
                # 添加后期额外下降
                if random.random() < 0.5:
                    post_events = random.randint(2, 4)
                    for i in range(post_events):
                        post_time = decline_time + random.randint(1000, 3000) * (i+1)
                        post_factor = random.uniform(1.5, 3.0)
                        demographic_events.append(msprime.PopulationParametersChange(time=post_time, initial_size=current_size * decline_factor * post_factor, population_id=0))
            else:
                for i in range(num_events):
                    time_point = (i + 1) * time_scale
                    decline_factor = 1.7 ** (1 + i/2)
                    new_size = current_size * decline_factor
                    demographic_events.append(msprime.PopulationParametersChange(time=time_point, initial_size=new_size, population_id=0))
                    current_size = new_size
        elif model_type == "complex":
            current_time = 0
            # 更高的随机性和变化频率
            for i in range(num_events):
                time_interval = random.randint(time_scale//3, time_scale*3)
                current_time += time_interval
                
                # 更极端的变化
                if random.random() < 0.2:
                    growth_factor = random.uniform(5.0, 15.0) if random.random() < 0.5 else random.uniform(0.01, 0.1)
                else:
                    growth_factor = random.uniform(1.5, 5.0) if random.random() < 0.5 else random.uniform(0.1, 0.7)
                
                demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * growth_factor, population_id=0))
                
                # 增加一些小幅波动
                if random.random() < 0.3:
                    mini_oscillations = random.randint(2, 5)
                    for j in range(mini_oscillations):
                        mini_time = current_time + random.randint(50, 200) * (j+1)
                        mini_factor = growth_factor * random.uniform(0.8, 1.2)
                        demographic_events.append(msprime.PopulationParametersChange(time=mini_time, initial_size=Ne * mini_factor, population_id=0))
        elif model_type == "cyclic":
            cycle_length = random.randint(3, 12)
            num_cycles = max(2, num_events // cycle_length)
            amplitude = random.uniform(0.5, 1.5)
            
            # 可变周期
            if random.random() < 0.4:
                demographic_events = []
                current_time = 0
                for cycle in range(num_cycles):
                    # 每个周期的周期长度略有不同
                    this_cycle_length = random.randint(max(2, cycle_length-2), cycle_length+2)
                    this_amplitude = amplitude * random.uniform(0.8, 1.2)
                    
                    for i in range(this_cycle_length):
                        phase = i / this_cycle_length
                        current_time += random.randint(time_scale//2, time_scale*3//2)
                        factor = 1 + this_amplitude * np.sin(2 * np.pi * phase)
                        demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
            else:
                for cycle in range(num_cycles):
                    base_time = cycle * cycle_length * time_scale
                    for i in range(cycle_length):
                        current_time = base_time + i * time_scale
                        factor = 1 + amplitude * np.sin(2 * np.pi * i / cycle_length)
                        demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
        elif model_type == "bottleneck_recovery":
            Ne = random.randint(8000, 30000)
            
            # 多阶段恢复
            if random.random() < 0.4:
                bottleneck_time = random.randint(500, 3000)
                bottleneck_severity = random.uniform(0.01, 0.2)
                recovery_events = random.randint(5, 12)
                
                demographic_events = [msprime.PopulationParametersChange(time=bottleneck_time, initial_size=Ne * bottleneck_severity, population_id=0)]
                
                # 先快速恢复，然后缓慢恢复
                fast_recovery = random.randint(2, 5)
                slow_recovery = recovery_events - fast_recovery
                
                # 快速恢复阶段
                fast_recovery_rate = (0.5 - bottleneck_severity) / fast_recovery
                for i in range(fast_recovery):
                    current_time = bottleneck_time + (i+1) * random.randint(time_scale//3, time_scale//2)
                    recovery_factor = bottleneck_severity + (i+1) * fast_recovery_rate
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * recovery_factor, population_id=0))
                
                # 缓慢恢复阶段
                slow_recovery_rate = (1.2 - 0.5) / slow_recovery
                for i in range(slow_recovery):
                    current_time += random.randint(time_scale//2, time_scale*2)
                    recovery_factor = 0.5 + (i+1) * slow_recovery_rate
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * recovery_factor, population_id=0))
            else:
                bottleneck_time = random.randint(500, 2000)
                bottleneck_severity = random.uniform(0.05, 0.2)
                bottleneck_duration = random.randint(200, 1000)
                recovery_events = num_events - 1
                demographic_events.append(msprime.PopulationParametersChange(time=bottleneck_time, initial_size=Ne * bottleneck_severity, population_id=0))
                recovery_start = bottleneck_time + bottleneck_duration
                recovery_step = (1.2 - bottleneck_severity) / recovery_events
                for i in range(recovery_events):
                    current_time = recovery_start + i * time_scale
                    recovery_factor = bottleneck_severity + (i + 1) * recovery_step
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * recovery_factor, population_id=0))
        elif model_type == "zigzag":
            current_time = 0
            increase = True
            
            # 不等间距的锯齿
            if random.random() < 0.4:
                demographic_events = []
                for i in range(num_events):
                    # 不等的时间间隔和变化幅度
                    time_interval = random.randint(time_scale//3, time_scale*3)
                    current_time += time_interval
                    
                    # 有时添加极端变化
                    if random.random() < 0.2:
                        factor = random.uniform(4.0, 10.0) if increase else random.uniform(0.05, 0.2)
                    else:
                        factor = random.uniform(1.5, 4.0) if increase else random.uniform(0.2, 0.7)
                        
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
                    
                    # 不总是切换方向
                    if random.random() < 0.8:
                        increase = not increase
            else:
                for i in range(num_events):
                    time_interval = random.randint(time_scale//2, time_scale*3//2)
                    current_time += time_interval
                    factor = random.uniform(1.5, 3.0) if increase else random.uniform(0.2, 0.7)
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
                    increase = not increase
        elif model_type == "step":
            current_time = 0
            step_size = random.randint(2, 5)
            step_direction = random.choice([-1, 1])
            
            # 不规则的台阶
            if random.random() < 0.4:
                demographic_events = []
                current_factor = 1.0
                
                for i in range(num_events):
                    # 不等的时间间隔
                    time_interval = random.randint(time_scale//2, time_scale*2)
                    current_time += time_interval
                    
                    # 随机决定是否改变方向
                    if i % step_size == 0 or random.random() < 0.2:
                        step_direction *= -1
                    
                    # 计算新的系数
                    step_magnitude = random.uniform(0.2, 1.0) if random.random() < 0.7 else random.uniform(1.0, 3.0)
                    
                    if step_direction > 0:
                        current_factor += step_magnitude
                    else:
                        current_factor = max(0.1, current_factor - step_magnitude)
                        
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * current_factor, population_id=0))
            else:
                for i in range(num_events):
                    time_interval = random.randint(time_scale//2, time_scale*3//2)
                    current_time += time_interval
                    if i % step_size == 0:
                        step_direction *= -1
                    if step_direction > 0:
                        factor = 1 + random.uniform(0.2, 0.8) * (i % step_size + 1) / step_size
                    else:
                        factor = 1 - random.uniform(0.2, 0.6) * (i % step_size + 1) / step_size
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
        elif model_type == "exponential":
            current_time = 0
            is_growth = random.choice([True, False])
            
            # 分段指数模型
            if random.random() < 0.4:
                demographic_events = []
                current_factor = 1.0
                
                segments = random.randint(2, 4)
                for segment in range(segments):
                    # 每段有不同的基数和时间点数
                    seg_events = random.randint(3, 7)
                    base = random.uniform(1.3, 1.8) if is_growth else random.uniform(0.5, 0.8)
                    
                    for i in range(seg_events):
                        time_interval = random.randint(time_scale//2, time_scale*2)
                        current_time += time_interval
                        
                        if is_growth:
                            current_factor *= base
                        else:
                            current_factor /= base
                        
                        demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * current_factor, population_id=0))
                    
                    # 有可能在段之间切换增长方向
                    if random.random() < 0.3:
                        is_growth = not is_growth
            else:
                base = 1.5 if is_growth else 0.7
                for i in range(num_events):
                    time_interval = random.randint(time_scale//2, time_scale*3//2)
                    current_time += time_interval
                    factor = base ** (num_events - i) if is_growth else base ** (i + 1)
                    demographic_events.append(msprime.PopulationParametersChange(time=current_time, initial_size=Ne * factor, population_id=0))
        elif model_type == "saw_tooth":
            # 增强锯齿波模型
            if random.random() < 0.4:
                demographic_events = []
                current_time = 0
                
                num_teeth = random.randint(3, 7)
                for tooth in range(num_teeth):
                    # 每个锯齿有不同的大小和形状
                    tooth_duration = random.randint(time_scale, time_scale*3)
                    crash_position = random.uniform(0.5, 0.9)  # 崩溃位于锯齿的哪个位置
                    rise_duration = int(tooth_duration * crash_position)
                    fall_duration = tooth_duration - rise_duration
                    
                    # 上升阶段
                    rise_steps = random.randint(3, 7)
                    for step in range(rise_steps):
                        step_time = current_time + rise_duration * step / rise_steps
                        step_factor = 1.0 + (step + 1) / rise_steps * random.uniform(0.5, 2.0)
                        demographic_events.append(msprime.PopulationParametersChange(time=step_time, initial_size=Ne * step_factor, population_id=0))
                    
                    # 崩溃阶段
                    crash_time = current_time + rise_duration
                    crash_factor = random.uniform(0.05, 0.3)
                    demographic_events.append(msprime.PopulationParametersChange(time=crash_time, initial_size=Ne * crash_factor, population_id=0))
                    
                    current_time += tooth_duration
            else:
                cycle_length = random.randint(3, 6)
                num_cycles = max(1, num_events // cycle_length)
                for cycle in range(num_cycles):
                    base_time = cycle * cycle_length * time_scale
                    crash_time = base_time + time_scale
                    crash_factor = random.uniform(0.1, 0.3)
                    demographic_events.append(msprime.PopulationParametersChange(time=crash_time, initial_size=Ne * crash_factor, population_id=0))
                    recovery_events = cycle_length - 1
                    for i in range(recovery_events):
                        recovery_time = crash_time + (i + 1) * time_scale
                        recovery_factor = crash_factor + (1.2 - crash_factor) * (i + 1) / recovery_events
                        demographic_events.append(msprime.PopulationParametersChange(time=recovery_time, initial_size=Ne * recovery_factor, population_id=0))
        else:
            return self._generate_demographic_model(random_params=True, complexity=complexity)

        if random.random() < 0.5:
            for _ in range(random.randint(1, 5)):
                time_point = random.randint(time_scale // 2, time_span)
                size_factor = random.uniform(0.1, 5.0)
                demographic_events.append(msprime.PopulationParametersChange(time=time_point, initial_size=Ne * size_factor, population_id=0))
            demographic_events.sort(key=lambda x: x.time)
            
        return demographic_events, Ne

    def generate_single_sample(self, model_type='random', complexity='high', sequence_length=1000000, num_bins=20, rate_variation='high', add_variable_rates=True):
        """
        生成单个模拟样本
        """
        if model_type == 'random':
            demographic_events, Ne = self._generate_demographic_model(random_params=True, complexity=complexity)
        else:
            demographic_events, Ne = self._generate_custom_demographic_model(model_type=model_type, complexity=complexity)

        ts = self._simulate_sequence(demographic_events, Ne, sequence_length, rate_variation=rate_variation, add_variable_rates=add_variable_rates)
        binary_seq = self._extract_binary_sequence(ts, sequence_length)
        hist, bin_edges, tmrca_values = self._calculate_tmrca_histogram(ts, num_bins)
        normalized_hist = self._normalize_histogram(hist)

        return {
            'model_type': model_type,
            'complexity': complexity,
            'Ne': Ne,
            'demographic_events': demographic_events,
            'ts': ts,
            'binary_seq': binary_seq,
            'histogram': hist,
            'normalized_histogram': normalized_hist,
            'bin_edges': bin_edges,
            'tmrca_values': tmrca_values,
            'sequence_length': sequence_length,
            'segregating_sites': np.sum(binary_seq)
        }

    def generate_dataset(self, num_samples, model_type='random', complexity='high', sequence_length=1000000, **kwargs):
        """
        生成包含多个样本的数据集 (并行版)
        """
        # Use the instance's random state to generate seeds for workers
        # to ensure reproducibility if the main simulator has a fixed seed.
        random.seed(self.seed)
        seeds = [random.randint(1, 10**8) for _ in range(num_samples)]

        # Extract arguments for generate_single_sample from kwargs, with defaults
        num_bins = kwargs.get('num_bins', 20)
        rate_variation = kwargs.get('rate_variation', 'high')
        add_variable_rates = kwargs.get('add_variable_rates', True)

        args_list = [
            (model_type, complexity, sequence_length, num_bins, rate_variation, add_variable_rates, seed)
            for seed in seeds
        ]

        num_workers = multiprocessing.cpu_count()
        print(f"Generating {num_samples} samples using {num_workers} CPU cores...")
        
        dataset = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap_unordered is generally faster as it doesn't have to wait for results in order
            for i, result in enumerate(pool.imap_unordered(_generate_one_sample_for_multiprocessing, args_list), 1):
                dataset.append(result)
                print(f"  Generated sample {i}/{num_samples}...", end='\\r', flush=True)
        
        print(f"\\nData generation complete. {len(dataset)} samples generated.")
        return dataset

    def plot_demographic_history(self, sample):
        """
        绘制单个样本的人口历史
        """
        plt.figure(figsize=(12, 6))
        demographic_events = sample['demographic_events']
        Ne = sample['Ne']
        times = [0]
        ne_values = [Ne]
        for event in demographic_events:
            times.append(event.time)
            ne_values.append(event.initial_size)
        
        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices]
        ne_values = np.array(ne_values)[sorted_indices]
        
        plt.step(times, ne_values, where='post', linewidth=2)
        plt.xlabel('Time (generations ago)')
        plt.ylabel('Effective Population Size (Ne)')
        plt.title(f'Demographic History - {sample["model_type"]} ({sample["complexity"]})')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()

    def export_to_psmc(self, sequences, output_dir='psmc_data'):
        """
        将序列导出为PSMC格式
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, seq in enumerate(sequences):
            psmc_seq = [str(np.sum(chunk)) for chunk in np.array_split(seq['binary_seq'], len(seq['binary_seq']) // 100)]
            
            filepath = os.path.join(output_dir, f"sequence_{i+1}.psmcfa")
            with open(filepath, "w") as f:
                f.write(f">sequence_{i+1}\n")
                f.write("".join(psmc_seq) + "\n")
        print(f"Exported {len(sequences)} sequences to {output_dir}")

    def save_dataset_files(self, sequences, output_dir='sequence_data'):
        """
        将数据集的每个样本分别保存为文件
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, seq in enumerate(sequences):
            prefix = os.path.join(output_dir, f"sample_{i+1}")
            np.save(f"{prefix}_binary.npy", seq['binary_seq'])
            np.save(f"{prefix}_norm_hist.npy", seq['normalized_histogram'])
            
            metadata = {
                'seed': self.seed,
                'model_type': seq['model_type'],
                'complexity': seq['complexity'],
                'Ne': seq['Ne'],
                'bin_edges': seq['bin_edges'].tolist()
            }
            with open(f"{prefix}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        print(f"Saved {len(sequences)} samples to {output_dir}")

    def generate_diverse_dataset(self, num_samples, sequence_length=1000000, include_all_models=True, complexity_distribution=None, **kwargs):
        """
        生成包含多种模型类型和复杂度的多样化数据集
        
        参数:
            num_samples (int): 要生成的样本总数
            sequence_length (int): 每个样本的序列长度
            include_all_models (bool): 是否确保包含所有模型类型
            complexity_distribution (dict, optional): 各复杂度级别的比例分布，例如 {'low': 0.1, 'medium': 0.3, 'high': 0.4, 'extreme': 0.2}
            **kwargs: 传递给generate_single_sample的其他参数
        
        返回:
            list: 包含多个样本的数据集
        """
        all_model_types = [
            "bottleneck", "expansion", "decline", "complex", "cyclic", 
            "bottleneck_recovery", "zigzag", "step", "exponential", "saw_tooth"
        ]
        
        complexity_levels = ['low', 'medium', 'high', 'extreme']
        
        # 默认复杂度分布
        if complexity_distribution is None:
            complexity_distribution = {'low': 0.1, 'medium': 0.3, 'high': 0.4, 'extreme': 0.2}
        
        dataset = []
        
        # 确保所有模型类型都至少有一个样本
        if include_all_models:
            print("包含所有模型类型的样本...")
            for model_type in all_model_types:
                # 随机选择复杂度级别，加权选择
                complexity = random.choices(
                    complexity_levels, 
                    weights=[complexity_distribution[level] for level in complexity_levels]
                )[0]
                
                print(f"生成 '{model_type}' 模型 (复杂度: {complexity})...")
                sample = self.generate_single_sample(
                    model_type=model_type,
                    complexity=complexity,
                    sequence_length=sequence_length,
                    **kwargs
                )
                dataset.append(sample)
            
            # 剩余样本随机生成
            remaining_samples = num_samples - len(all_model_types)
        else:
            remaining_samples = num_samples
            
        if remaining_samples > 0:
            print(f"生成剩余的 {remaining_samples} 个随机样本...")
            
            # 为剩余样本创建模型类型分布
            # 可以调整权重来偏好特定的模型类型
            model_weights = {
                "bottleneck": 1.2, 
                "expansion": 1.2, 
                "decline": 1.0, 
                "complex": 1.5, 
                "cyclic": 0.8, 
                "bottleneck_recovery": 1.0, 
                "zigzag": 0.8, 
                "step": 0.7, 
                "exponential": 0.8, 
                "saw_tooth": 0.7
            }
            
            model_types = list(model_weights.keys())
            weights = list(model_weights.values())
            
            for i in range(remaining_samples):
                # 随机选择模型类型，按权重
                model_type = random.choices(model_types, weights=weights)[0]
                
                # 随机选择复杂度级别，按权重
                complexity = random.choices(
                    complexity_levels, 
                    weights=[complexity_distribution[level] for level in complexity_levels]
                )[0]
                
                print(f"生成样本 {len(dataset)+1}/{num_samples}: '{model_type}' (复杂度: {complexity})...")
                sample = self.generate_single_sample(
                    model_type=model_type,
                    complexity=complexity,
                    sequence_length=sequence_length,
                    **kwargs
                )
                dataset.append(sample)
                
        return dataset

if __name__ == '__main__':
    # 示例：如何使用DataSimulator类
    
    # 1. 初始化模拟器，可以使用一个固定的种子以保证结果可复现
    simulator = DataSimulator(seed=42)

    # 2. 生成一个高复杂度的瓶颈模型样本并可视化
    print("--- 生成单个'瓶颈'模型样本 ---")
    single_sample = simulator.generate_single_sample(model_type='bottleneck', complexity='high')
    simulator.plot_demographic_history(single_sample)

    # 3. 生成一个包含5个样本的小型数据集，使用不同的随机模型
    print("\n--- 生成一个小型'随机'模型数据集 ---")
    dataset = simulator.generate_dataset(num_samples=5, model_type='random', complexity='high', sequence_length=500000)
    print(f"生成了包含 {len(dataset)} 个样本的数据集")

    # 4. 生成一个多样化的数据集
    print("\n--- 生成一个多样化的数据集 ---")
    diverse_dataset = simulator.generate_diverse_dataset(
        num_samples=20, 
        sequence_length=300000,
        include_all_models=True,
        complexity_distribution={'low': 0.1, 'medium': 0.3, 'high': 0.4, 'extreme': 0.2},
        rate_variation='high',
        add_variable_rates=True
    )
    print(f"生成了包含 {len(diverse_dataset)} 个样本的多样化数据集")

    # 5. 将生成的数据集保存到文件
    print("\n--- 保存数据集到文件 ---")
    simulator.save_dataset_files(diverse_dataset, output_dir='output/diverse_dataset')

    # 6. 将数据集导出为PSMC格式
    print("\n--- 导出数据集为PSMC格式 ---")
    simulator.export_to_psmc(diverse_dataset, output_dir='output/psmc_files')

    # 7. 可视化部分样本的人口历史
    print("\n--- 可视化部分样本 ---")
    sample_indices = random.sample(range(len(diverse_dataset)), min(5, len(diverse_dataset)))
    for idx in sample_indices:
        print(f"样本 {idx+1}: {diverse_dataset[idx]['model_type']} (复杂度: {diverse_dataset[idx]['complexity']})")
        simulator.plot_demographic_history(diverse_dataset[idx])

    print("\n模拟完成。") 