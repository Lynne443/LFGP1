import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)

class EmbLayer(nn.Module):

    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)

        x = self.ff_1(x)
        return x


class Emb(nn.Module):

    def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        d_model = d_model//4
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

    def forward(self, x):
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        return s_out


class FrequencyAdaptiveEmbedding(nn.Module):
    """
    基于傅里叶变换的自适应patching嵌入层
    通过分析输入序列的主频率，自适应确定patch长度
    """
    def __init__(self, seq_len, d_model, min_patch_len=4, max_patch_len=None, patch_step_ratio=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.min_patch_len = min_patch_len
        self.max_patch_len = max_patch_len if max_patch_len is not None else seq_len // 2
        self.patch_step_ratio = patch_step_ratio  # patch_step相对于patch_len的比例
        
    def get_frequency_patches(self, x):
        """
        基于傅里叶变换得到主频率，并据此确定patch长度
        x: [Batch, Variable, Sequence_Length]
        """
        batch_size, var_dim, _ = x.shape
        
        # 对每个变量进行傅里叶变换
        fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft)
        
        # 排除直流分量(即频率为0的分量)
        amplitude[:, :, 0] = 0
        
        # 获取主频率索引(对应幅值最大的频率)
        dominant_freq_idx = torch.argmax(amplitude, dim=-1)  # [B, V]
        
        # 计算对应的周期（序列长度除以频率）
        # 频率索引需要转换为实际频率 (索引i对应的频率为i/N)
        # 周期 = N / (idx / N) = N² / idx
        period = torch.div(self.seq_len * self.seq_len, dominant_freq_idx + 1, rounding_mode='floor')
        
        # 将周期作为patch长度，但需要限制在合理范围内
        patch_lens = torch.clamp(period, min=self.min_patch_len, max=self.max_patch_len)
        
        # 计算每个样本的patch步长
        patch_steps = torch.div(patch_lens, int(1/self.patch_step_ratio), rounding_mode='floor')
        patch_steps = torch.clamp(patch_steps, min=1)  # 确保步长至少为1
        
        return patch_lens, patch_steps
    
    def adaptive_patch_embedding(self, x, patch_lens, patch_steps):
        """
        根据计算得到的patch长度和步长进行patching和embedding
        为每个样本单独处理
        """
        batch_size, var_dim, seq_len = x.shape
        embedded_features = []
        
        for b in range(batch_size):
            sample_features = []
            
            for v in range(var_dim):
                # 获取当前样本当前变量的patch长度和步长
                patch_len = int(patch_lens[b, v].item())
                patch_step = int(patch_steps[b, v].item())
                
                # 计算patch数量
                num_patches = max(1, int((seq_len - patch_len) / patch_step + 1))
                
                # 创建当前样本当前变量的embedding层
                d_model_per_patch = self.d_model // num_patches
                if d_model_per_patch < 1:  # 处理极端情况
                    d_model_per_patch = 1
                    
                # patch划分
                x_patched = x[b, v].unfold(dimension=0, size=patch_len, step=patch_step)  # [num_patches, patch_len]
                
                # 使用线性层进行嵌入
                patch_embedding = nn.Linear(patch_len, d_model_per_patch).to(x.device)
                embedded_patches = patch_embedding(x_patched)  # [num_patches, d_model_per_patch]
                
                # 展平并投影到目标维度
                flattened = embedded_patches.flatten()  # [num_patches * d_model_per_patch]
                
                # 确保投影输入维度与目标维度匹配
                if flattened.shape[0] < self.d_model:
                    # 如果不足，则填充
                    padding = torch.zeros(self.d_model - flattened.shape[0], device=x.device)
                    flattened = torch.cat([flattened, padding])
                elif flattened.shape[0] > self.d_model:
                    # 如果超出，则截断
                    flattened = flattened[:self.d_model]
                
                sample_features.append(flattened)
            
            # 合并当前样本的所有变量特征
            sample_tensor = torch.stack(sample_features)  # [V, d_model]
            embedded_features.append(sample_tensor)
        
        # 合并所有样本
        result = torch.stack(embedded_features)  # [B, V, d_model]
        return result
    
    def forward(self, x):
        # 获取每个样本每个变量的patch长度和步长
        patch_lens, patch_steps = self.get_frequency_patches(x)
        
        # 进行自适应patching和embedding
        embedded = self.adaptive_patch_embedding(x, patch_lens, patch_steps)
        
        return embedded


class FrequencyAdaptiveEmbedding_Efficient(nn.Module):
    """
    基于傅里叶变换的自适应patching嵌入层的高效实现版本
    通过分析输入序列的主频率，自适应确定patch长度，对整个批次使用相同的patch配置
    """
    def __init__(self, seq_len, d_model, min_patch_len=4, max_patch_len=None, patch_step_ratio=0.5, num_freq_bands=3):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.min_patch_len = min_patch_len
        self.max_patch_len = max_patch_len if max_patch_len is not None else seq_len // 2
        self.patch_step_ratio = patch_step_ratio
        self.num_freq_bands = num_freq_bands
        
        # 预先创建可能的线性投影层
        self.patch_projectors = nn.ModuleDict()
        possible_patch_lens = []
        
        # 为可能的patch长度创建投影层
        for i in range(min_patch_len, self.max_patch_len + 1, 4):  # 每4个长度为一组，减少模型大小
            possible_patch_lens.append(i)
        
        for patch_len in possible_patch_lens:
            # 计算对应的patch数量
            patch_step = max(1, int(patch_len * self.patch_step_ratio))
            num_patches = max(1, int((seq_len - patch_len) / patch_step + 1))
            d_model_per_patch = max(1, self.d_model // num_patches)
            
            # 为每个可能的patch长度创建投影层
            self.patch_projectors[f"len_{patch_len}"] = nn.Sequential(
                nn.Linear(patch_len, d_model_per_patch),
                nn.GELU()
            )
            
            # 创建最终投影层
            final_in_dim = d_model_per_patch * num_patches
            if final_in_dim > self.d_model:
                self.patch_projectors[f"proj_{patch_len}"] = nn.Linear(final_in_dim, self.d_model)
        
        # 创建一个共享的频段嵌入层
        self.freq_band_embedding = nn.Linear(self.num_freq_bands, self.d_model)
    
    def get_median_patch_len(self, x):
        """
        计算整个批次的中位数patch长度
        x: [Batch, Variable, Sequence_Length]
        """
        # 对每个变量进行傅里叶变换
        fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft)
        
        # 排除直流分量
        amplitude[:, :, 0] = 0
        
        # 获取主频率索引
        dominant_freq_idx = torch.argmax(amplitude, dim=-1)  # [B, V]
        
        # 计算对应的周期
        period = torch.div(self.seq_len * self.seq_len, dominant_freq_idx + 1, rounding_mode='floor')
        
        # 将周期作为patch长度，取中位数
        median_patch_len = int(torch.median(period).item())
        patch_len = max(self.min_patch_len, min(median_patch_len, self.max_patch_len))
        
        # 找到最接近的预定义patch长度
        available_patch_lens = [int(k.split('_')[1]) for k in self.patch_projectors.keys() if k.startswith('len_')]
        closest_patch_len = min(available_patch_lens, key=lambda x: abs(x - patch_len))
        
        return closest_patch_len
    
    def get_frequency_bands(self, x):
        """
        提取频率段特征
        """
        # 对每个变量进行傅里叶变换
        fft = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft)
        
        # 将频谱分为几个频段
        freq_bands = []
        band_size = (self.seq_len // 2 + 1) // self.num_freq_bands
        
        for i in range(self.num_freq_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, self.seq_len // 2 + 1)
            
            # 计算每个频段的能量
            if start_idx < end_idx:
                band_energy = torch.mean(amplitude[:, :, start_idx:end_idx], dim=-1)
                freq_bands.append(band_energy)
            else:
                # 处理边界情况
                band_energy = torch.zeros_like(amplitude[:, :, 0])
                freq_bands.append(band_energy)
        
        # 堆叠所有频段特征
        freq_features = torch.stack(freq_bands, dim=-1)  # [B, V, num_freq_bands]
        
        return freq_features
        
    def forward(self, x):
        """
        x: [Batch, Variable, Sequence_Length]
        """
        batch_size, var_dim, _ = x.shape
        
        # 第1步：获取中位数patch长度
        patch_len = self.get_median_patch_len(x)
        patch_step = max(1, int(patch_len * self.patch_step_ratio))
        
        # 第2步：进行patching
        x_patched = x.unfold(dimension=-1, size=patch_len, step=patch_step)  # [B, V, num_patches, patch_len]
        num_patches = x_patched.size(2)
        
        # 第3步：使用预定义的投影层进行嵌入
        x_patched_reshaped = x_patched.reshape(-1, patch_len)  # [B*V*num_patches, patch_len]
        embedded_patches = self.patch_projectors[f"len_{patch_len}"](x_patched_reshaped)  # [B*V*num_patches, d_model_per_patch]
        
        # 第4步：重塑并展平
        d_model_per_patch = embedded_patches.size(-1)
        embedded_patches = embedded_patches.reshape(batch_size, var_dim, num_patches, d_model_per_patch)
        flattened = embedded_patches.reshape(batch_size, var_dim, num_patches * d_model_per_patch)
        
        # 第5步：如果需要，进行最终投影
        final_in_dim = num_patches * d_model_per_patch
        if final_in_dim > self.d_model:
            if f"proj_{patch_len}" in self.patch_projectors:
                output = self.patch_projectors[f"proj_{patch_len}"](flattened)  # [B, V, d_model]
            else:
                # 如果没有预定义的投影层，则截断
                output = flattened[:, :, :self.d_model]
        elif final_in_dim < self.d_model:
            # 如果维度不足，则填充
            padding = torch.zeros(batch_size, var_dim, self.d_model - final_in_dim, device=x.device)
            output = torch.cat([flattened, padding], dim=-1)
        else:
            output = flattened
        
        # 第6步：添加频段特征以增强表示
        freq_features = self.get_frequency_bands(x)
        freq_embedding = self.freq_band_embedding(freq_features)  # [B, V, d_model]
        
        # 融合patch嵌入和频段嵌入
        output = output + 0.1 * freq_embedding  # 低权重融合，以保持主要信息
        
        return output
    

class DataEmbedding_FreqInterpolate(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DataEmbedding_FreqInterpolate, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.fft(x, n=2 * L)
        x_fft_resampled = self.resample_fft(x_fft, self.d_model)
        return x_fft_resampled

    def resample_fft(self, x_fft, new_length):
        real_part = x_fft.real
        imag_part = x_fft.imag
        real_interpolated = F.interpolate(real_part, size=new_length, mode='linear', align_corners=False)
        imag_interpolated = F.interpolate(imag_part, size=new_length, mode='linear', align_corners=False)
        x_fft_resampled = torch.complex(real_interpolated, imag_interpolated)
        return x_fft_resampled


'''
class PatchLayer(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
    def get_patch_params(self, x):
        """计算patch相关参数"""
        # FFT分析获取主频率
        fft = torch.fft.fft(x)
        amp = torch.abs(fft)[..., :self.seq_len // 2]
        f = torch.argmax(amp, dim=-1)  # [B, V]
        
        # 计算patch长度
        p = torch.div(self.seq_len, f + 1, rounding_mode='floor')
        patch_len = torch.div(p, 2, rounding_mode='floor')
        patch_len = max(int(torch.min(patch_len)), 4)
        patch_step = patch_len // 2
        
        # 计算patch数量
        patch_num = int((self.seq_len - patch_len) / patch_step + 1)
        
        return patch_len, patch_step, patch_num
    
    def patch_and_embed(self, x, patch_len, patch_step, patch_num):
        """patch分割并embedding"""
        # patch分割
        x = x.unfold(dimension=-1, size=patch_len, step=patch_step)  # [B, V, num_patches, patch_len]
        
        # embedding
        d_model_temp = self.d_model // patch_num
        ff = nn.Linear(patch_len, d_model_temp).to(x.device)
        ff_1 = nn.Linear(d_model_temp * patch_num, self.d_model).to(x.device)
        
        x = ff(x)  # [B, V, num_patches, d_model_temp]
        x = x.flatten(start_dim=-2)  # [B, V, num_patches * d_model_temp]
        x = ff_1(x)  # [B, V, d_model]
        
        return x
        
    def forward(self, x):
        # x: [B, V, L]
        patch_len, patch_step, patch_num = self.get_patch_params(x)
        return self.patch_and_embed(x, patch_len, patch_step, patch_num)
'''
    

'''
class EmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step
        self.seq_len = seq_len

    def forward(self, x):
        # x: [B, V, L]
        # 只进行patch分割，不做维度变换
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        return x  # [B, V, num_patches, patch_len]


class Emb(nn.Module):
    def __init__(self, seq_len, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len  # 使用patch_len作为步长
        
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len)

    def forward(self, x):
        # 获取不同尺度的patch
        s_x1 = self.EmbLayer_1(x)  # [B, V, num_patches1, patch_len1]
        s_x2 = self.EmbLayer_2(x)  # [B, V, num_patches2, patch_len2]
        s_x3 = self.EmbLayer_3(x)  # [B, V, num_patches3, patch_len3]
        s_x4 = self.EmbLayer_4(x)  # [B, V, num_patches4, patch_len4]
        
        return [s_x1, s_x2, s_x3, s_x4]  # 返回patch列表
'''