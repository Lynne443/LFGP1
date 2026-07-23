import pandas as pd
import numpy as np
from math import sin, pi
import os
import argparse


class ReferenceSeries:
    def __init__(self, data, type="sin", resolution=10, alpha=0.7, pr=False):
        self.L, self.C = data.shape
        self.type = type
        self.alpha = alpha
        self.pr = pr
        
        for n in range(2, self.L):
            if round(self.L / (n - 1)) < round(self.L / n) + resolution:
                break            
        self.boundary = round(self.L / n)
        
        self.frequency, self.fre_magnitude = self._fft(data)
        self.period, self.per_magnitude = self._convert_spectrum()

    
    def _fft(self, data):
        mean = np.mean(data)
        data = data - mean

        fft_result = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(self.L)
        frequency = freqs[freqs >= 0]
        fre_magnitude = np.abs(fft_result[freqs >= 0])
        
        return frequency, fre_magnitude        

    
    def _convert_spectrum(self):
        period = np.array([i for i in range(self.boundary)])
        per_magnitude = np.zeros((self.boundary, self.C))
        
        for per in period[4:]:
            indices = self.get_indices_by_per(per)
            per_magnitude[per] = np.max(self.fre_magnitude[indices, :], axis=0)
            
        return period, per_magnitude


    def get_indices_by_per(self, pbp, hbp=1):
        index = round(self.L * hbp / pbp)
        
        if round(self.L * hbp / (index - 2)) == pbp:
            start = index - 2
        elif round(self.L * hbp / (index - 2)) == pbp:
            start = index - 1
        else:
            start = index
            
        if round(self.L * hbp / (index + 2)) == pbp:
            end = index + 2
        elif round(self.L * hbp / (index + 1)) == pbp:
            end = index + 1
        else:
            end = index
         
        indices = np.array([n for n in range(start, end + 1)])
        
        return indices

                
    def channel_wise_pbp_extractor_topk(self, top_k=3):
        """
        Extract top-k dominant periods for each channel based on spectrum magnitude.
        
        Args:
            top_k: Number of dominant periods to extract per channel
            
        Returns:
            channel_pbps: List of lists, each channel has exactly top_k periods (padded with 0 if insufficient)
        """
        channel_pbps = []
        
        for channel in range(self.C):
            channel_magnitude = self.per_magnitude[:, channel]
            
            valid_start = 4
            valid_end = self.boundary // 2
            
            if valid_end <= valid_start:
                pbp_topk = [0] * top_k
            else:
                valid_magnitude = channel_magnitude[valid_start:valid_end]
                valid_periods = np.arange(valid_start, valid_end)
                
                sorted_indices = np.argsort(valid_magnitude)[::-1]
                top_k_indices = sorted_indices[:top_k]
                
                pbp_list = valid_periods[top_k_indices].tolist()
                
                if len(pbp_list) < top_k:
                    pbp_topk = pbp_list + [0] * (top_k - len(pbp_list))
                else:
                    pbp_topk = pbp_list
            
            channel_pbps.append(pbp_topk)
            
            if self.pr:
                print(f"Channel {channel}: top-{top_k} periods = {pbp_topk}")
        
        self.channel_pbps_topk = channel_pbps
        return channel_pbps
    
    
    def statistical_common_periods_from_topk(self, top_n=1):
        """
        Extract common periods by counting occurrence frequency across channels.
        
        Selection strategy:
        1. Sort by occurrence frequency
        2. Select the first one (highest frequency)
        3. Subsequent periods must be larger than the maximum already selected
        4. Repeat until top_n periods are selected
        
        This ensures sufficient diversity between selected periods.
        
        Args:
            top_n: Number of periods to select (default 1)
            
        Returns:
            common_periods: List of common periods (sorted by period value)
        """
        if not hasattr(self, 'channel_pbps_topk'):
            raise ValueError("Please run channel_wise_pbp_extractor_topk() first")
        
        period_channel_count = {}
        
        for channel_idx, channel_pbps in enumerate(self.channel_pbps_topk):
            for pbp in channel_pbps:
                if pbp > 0:
                    if pbp not in period_channel_count:
                        period_channel_count[pbp] = set()
                    period_channel_count[pbp].add(channel_idx)
        
        if len(period_channel_count) == 0:
            if self.pr:
                print("\nWarning: No valid periods found")
            return []
        
        period_stats = []
        for period, channels in period_channel_count.items():
            count = len(channels)
            ratio = count / self.C
            period_stats.append((period, count, ratio))
        
        period_stats.sort(key=lambda x: x[1], reverse=True)
        
        if self.pr:
            print(f"\nPeriod statistics (sorted by channel coverage):")
            print(f"{'Period':>6} | {'Channels':>8} | {'Coverage':>8}")
            print("-" * 30)
            for period, count, ratio in period_stats[:30]:
                print(f"{period:6d} | {count:8d} | {ratio:7.1%}")
            if len(period_stats) > 30:
                print(f"  ... (total {len(period_stats)} different periods)")
        
        common_periods = []
        
        for period, count, ratio in period_stats:
            if len(common_periods) == 0:
                common_periods.append(period)
                if self.pr:
                    print(f"\nSelected 1st period: {period} (appears {count} times, {ratio:.1%})")
            elif len(common_periods) < top_n:
                max_selected = max(common_periods)
                if period > max_selected:
                    common_periods.append(period)
                    if self.pr:
                        print(f"Selected {len(common_periods)}th period: {period} (appears {count} times, {ratio:.1%}, larger than max {max_selected})")
                else:
                    if self.pr and len(common_periods) == 1:
                        print(f"Skipping period {period} (not larger than max {max_selected})...")
            else:
                break
        
        common_periods = sorted(common_periods)
        
        if self.pr:
            print(f"\nStatistical common periods:")
            print(f"  - Strategy: Smart top_{top_n} (highest frequency, subsequent must be larger than max)")
            print(f"  - Common periods: {common_periods}")
            print(f"  - Number of periods: {len(common_periods)}")
            
            print(f"\nDetails of selected periods:")
            for period in common_periods:
                count = len(period_channel_count[period])
                ratio = count / self.C
                print(f"  - Period {period:3d}: {count:3d} channels ({ratio:6.1%})")
        
        self.statistical_common_periods = common_periods
        return common_periods
    
    
    def statistical_common_periods_reference_series_generator(self):
        """
        Generate reference series based on common periods from statistical_common_periods_from_topk.
        
        Returns:
            rs: Array of shape [L, len(statistical_common_periods)], one periodic wave per common period
        """
        assert self.type in ["sin", "swatooth", "reactangle", "pulse"], \
            "type must be one of sin, swatooth, reactangle, or pulse"
        
        if not hasattr(self, 'statistical_common_periods'):
            raise ValueError("Please run statistical_common_periods_from_topk() first")
        
        rs = []
        for period in self.statistical_common_periods:
            if self.type == "sin":
                wave = [sin(t * 2 * pi * 1 / period) for t in range(self.L)]
            elif self.type == "swatooth":
                wave = [t * 1 % period for t in range(self.L)]
            elif self.type == "reactangle":
                wave = [2 * t * 1 // period % 2 for t in range(self.L)]
            elif self.type == "pulse":
                wave = [1 if t * 1 % period == 0 else 0 for t in range(self.L)]
            rs.append(wave)
        
        if self.pr:
            print(f"\nGenerated reference series from statistical common periods:")
            print(f"  - Common periods: {self.statistical_common_periods}")
            print(f"  - Reference series shape: [{self.L}, {len(self.statistical_common_periods)}]")
        
        return np.array(rs).T
    
    
    def channel_wise_reference_series_generator_topk(self, top_k=3):
        """
        Generate fixed number (top_k) of periodic waves for each channel (not superimposed).
        Uses top_k periods stored in channel_pbps_topk for each channel.
        
        Args:
            top_k: Number of periods per channel
            
        Returns:
            channel_rs: Array of shape [L, C*top_k], exactly top_k waves per channel
                       Order: [ch0_freq0, ..., ch0_freq(k-1), ch1_freq0, ..., ch1_freq(k-1), ...]
                       If a period is 0 (padding), the corresponding wave is all zeros
        """
        assert self.type in ["sin", "swatooth", "reactangle", "pulse"], \
            "type must be one of sin, swatooth, reactangle, or pulse"
        
        if not hasattr(self, 'channel_pbps_topk'):
            self.channel_wise_pbp_extractor_topk(top_k=top_k)
        
        total_waves = self.C * top_k
        channel_rs = np.zeros((self.L, total_waves))
        
        wave_idx = 0
        for channel in range(self.C):
            pbp_list = self.channel_pbps_topk[channel]
            
            for pbp in pbp_list:
                if pbp == 0:
                    wave = [0.0 for _ in range(self.L)]
                else:
                    bp = (pbp, 1)
                    
                    if self.type == "sin":
                        wave = [sin(t * 2 * pi * bp[1] / bp[0]) for t in range(self.L)]
                    elif self.type == "swatooth":
                        wave = [t * bp[1] % bp[0] for t in range(self.L)]
                    elif self.type == "reactangle":
                        wave = [2 * t * bp[1] // bp[0] % 2 for t in range(self.L)]
                    elif self.type == "pulse":
                        wave = [1 if t * bp[1] % bp[0] == 0 else 0 for t in range(self.L)]
                
                channel_rs[:, wave_idx] = wave
                wave_idx += 1
        
        if self.pr:
            print(f"\nChannel-wise reference series shape (top-{top_k}): {channel_rs.shape}")
            print(f"Periods per channel: {top_k}")
            print(f"Total waves: {total_waves} = C({self.C}) * top_k({top_k})")
        
        return channel_rs


def example_usage(args=None):
    """
    Example usage of ReferenceSeries for generating reference series.
    
    Args:
        args: Command line arguments object, uses default if None
    """
    print("="*80)
    print("ReferenceSeries Example Usage")
    print("="*80)
    
    if args is not None:
        data_path = os.path.join(args.root_path, args.data_path)
        type_ = args.type
        ksp = args.ksp
        pr = True
    else:
        data_path = "./dataset/traffic/traffic.csv"
        type_ = "sin"
        ksp = 3
        pr = True
    
    print(f"\nLoading dataset: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    data = df.iloc[:, 1:].values
    print(f"Data shape: {data.shape} (L={data.shape[0]}, C={data.shape[1]})")
    
    print(f"\n" + "="*80)
    print(f"1. Create ReferenceSeries Object")
    print(f"="*80)
    print(f"Parameters: type={type_}, ksp={ksp}")
    
    reference_series = ReferenceSeries(data, type_, pr=pr)
    
    print(f"\n" + "="*80)
    print(f"2. Extract Top-{ksp} Periods per Channel")
    print(f"="*80)
    
    channel_pbps = reference_series.channel_wise_pbp_extractor_topk(top_k=ksp)
    
    print(f"\nTop-{ksp} periods per channel:")
    for ch_idx, pbps in enumerate(channel_pbps[:10]):
        print(f"  Channel {ch_idx}: {pbps}")
    if len(channel_pbps) > 10:
        print(f"  ... (total {len(channel_pbps)} channels, showing first 10)")
    
    channel_rs = reference_series.channel_wise_reference_series_generator_topk(top_k=ksp)
    print(f"\nChannel-wise reference series shape: {channel_rs.shape}")
    
    print(f"\n" + "="*80)
    print(f"3. Extract Common Periods from Top-{ksp}")
    print(f"="*80)
    
    print(f"\n" + "-"*80)
    print(f"Testing top_n=1 (select 1 most frequent period):")
    print(f"-"*80)
    common_periods_stat_1 = reference_series.statistical_common_periods_from_topk(top_n=1)
    
    print(f"\n" + "-"*80)
    print(f"Testing top_n=2 (select 2 periods, 2nd must be larger than 1st):")
    print(f"-"*80)
    common_periods_stat_2 = reference_series.statistical_common_periods_from_topk(top_n=2)
    
    print(f"\n" + "-"*80)
    print(f"Testing top_n=3 (select 3 periods, each must be larger than previous max):")
    print(f"-"*80)
    common_periods_stat_3 = reference_series.statistical_common_periods_from_topk(top_n=3)
    
    print(f"\n" + "="*80)
    print(f"4. Comparison of Different top_n Values")
    print(f"="*80)
    print(f"\ntop_n=1: {common_periods_stat_1}")
    print(f"top_n=2: {common_periods_stat_2}")
    print(f"top_n=3: {common_periods_stat_3}")
    
    print(f"\n" + "="*80)
    print(f"5. Generate Reference Series (using top_n=2)")
    print(f"="*80)
    common_rs = reference_series.statistical_common_periods_reference_series_generator()
    print(f"Common period reference series shape: {common_rs.shape}")
    
    print(f"\n" + "="*80)
    print(f"6. Usage in Data Loader")
    print(f"="*80)
    print(f"""
Usage Example:
----------------------------------------
from utils.Periodicity import ReferenceSeries

# 1. Create ReferenceSeries object
reference_series = ReferenceSeries(data, type='sin', pr=True)

# 2. Extract top-k periods per channel and generate channel-wise reference series
reference_series.channel_wise_pbp_extractor_topk(top_k=3)
channel_rs = reference_series.channel_wise_reference_series_generator_topk(top_k=3)
# channel_rs shape: [L, C*k] - k periodic waves per channel

# 3. Extract common periods statistically and generate reference series
# Smart selection of top_n periods (subsequent must be larger than max)
common_periods = reference_series.statistical_common_periods_from_topk(top_n=2)
common_rs = reference_series.statistical_common_periods_reference_series_generator()
# common_rs shape: [L, top_n] - periodic waves for top_n common periods

# Parameter explanation:
#   top_n=1: Select 1 most common period
#   top_n=2: Select 2 periods, 2nd must be larger than 1st
#   top_n=3: Select 3 periods, each must be larger than previous max
""")
    
    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)
    
    return reference_series


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ReferenceSeries Reference Series Generation Tool')
    
    parser.add_argument('--root_path', type=str, default='./dataset/electricity/', 
                       help='Root directory path for data')
    parser.add_argument('--data_path', type=str, default='electricity.csv', 
                       help='Data file name')
    
    parser.add_argument('--type', type=str, default='sin', 
                       choices=['sin', 'swatooth', 'reactangle', 'pulse'],
                       help='Reference series type')
    parser.add_argument('--ksp', type=int, default=3, 
                       help='Number of dominant periods to extract per channel')
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        print("\n" + "="*80)
        print("Running ReferenceSeries with Command Line Arguments")
        print("="*80)
        args = parse_args()
        print(f"Parsed arguments: {args}")
        example_usage(args)
    else:
        print("\n" + "="*80)
        print("Running Example with Default Parameters")
        print("="*80)
        print("Tip: You can use command line arguments, for example:")
        print("python utils/Periodicity.py --root_path ./dataset/electricity/ --data_path electricity.csv --ksp 3")
        print("="*80)
        example_usage()
