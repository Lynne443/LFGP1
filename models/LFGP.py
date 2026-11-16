import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN

EPS = 1e-8


class AEF(nn.Module):
    
    def __init__(self, seq_len, pred_len, hidden_size, num_filters=2, dropout=0.0, 
                 is_channel_filter=False, freq_band_ratio=0.4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.dropout = nn.Dropout(dropout)
        self.is_channel_filter = is_channel_filter
        self.scale = 0.02
        self.freq_band_ratio = freq_band_ratio
        
        self.complex_weights = nn.ParameterList([
            nn.Parameter(torch.randn(seq_len // 2 + 1, 2, dtype=torch.float32), 
                        requires_grad=True)
            for _ in range(num_filters)
        ])
        
        initial_weights = torch.tensor([0.5, 0.5])
        self.filter_weights = nn.Parameter(initial_weights, requires_grad=True)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.pred_len)
        )

    def _compute_kneepoint(self, energy_per_freq, freq_bins, device):
        cumsum_energy = torch.cumsum(energy_per_freq, dim=1)
        total_energy = cumsum_energy[:, -1:] + EPS
        C_k = cumsum_energy / total_energy

        k_indices = torch.arange(freq_bins, device=device, dtype=torch.float32)
        x_k = k_indices / (freq_bins - 1)
        x_k = x_k.unsqueeze(0).expand(energy_per_freq.size(0), -1)

        g_k = C_k - x_k
        knee_indices = torch.argmax(g_k, dim=1, keepdim=True)
        
        knee_indices = knee_indices.clamp(
            min=int(freq_bins * 0.15),
            max=int(freq_bins * 0.85)
        )
        
        return knee_indices

    def _create_frequency_masks(self, knee_indices, freq_bins, device):
        B = knee_indices.size(0)
        freq_indices = torch.arange(freq_bins, device=device, dtype=torch.float32)
        freq_indices = freq_indices.unsqueeze(0).expand(B, -1)
        
        low_mask = freq_indices <= knee_indices.float()
        high_mask = ~low_mask
        
        return low_mask, high_mask

    def _select_top_frequencies(self, energy_per_freq, low_mask, high_mask, device):
        low_len = low_mask.sum(dim=1, keepdim=True).float()
        high_len = high_mask.sum(dim=1, keepdim=True).float()

        low_k_all = (low_len * self.freq_band_ratio).clamp(min=1).long()
        high_k_all = (high_len * self.freq_band_ratio).clamp(min=1).long()

        k_low_max = int(low_k_all.max().item())
        k_high_max = int(high_k_all.max().item())

        big_neg = -1e9

        # Vectorized patch entropy using prefix sums. The low- and high-
        # frequency candidate regions are divided independently into
        # consecutive patches of three bins.
        patch_size = 3
        batch_size, freq_bins = energy_per_freq.shape
        freq_index = torch.arange(freq_bins, device=device).unsqueeze(0)
        low_length = low_mask.sum(dim=1, keepdim=True)

        region_start = torch.where(
            low_mask,
            torch.zeros_like(low_length),
            low_length
        )
        region_end = torch.where(
            low_mask,
            low_length,
            torch.full_like(low_length, freq_bins)
        )
        patch_start = (
            region_start
            + ((freq_index - region_start) // patch_size) * patch_size
        )
        patch_end = torch.minimum(patch_start + patch_size, region_end)
        patch_count = patch_end - patch_start

        # For p_k = E_k / sum(E), use the equivalent entropy expression
        # H = log(sum(E)) - sum(E * log(E)) / sum(E). Prefix sums provide
        # each bin's patch statistics without Python loops or scatter_add_.
        zero_column = energy_per_freq.new_zeros(batch_size, 1)
        energy_prefix = torch.cat(
            [zero_column, torch.cumsum(energy_per_freq, dim=1)], dim=1
        )
        energy_log_energy = energy_per_freq * torch.log(
            energy_per_freq + EPS
        )
        energy_log_prefix = torch.cat(
            [zero_column, torch.cumsum(energy_log_energy, dim=1)], dim=1
        )

        patch_energy = (
            energy_prefix.gather(1, patch_end)
            - energy_prefix.gather(1, patch_start)
        )
        patch_energy_log_energy = (
            energy_log_prefix.gather(1, patch_end)
            - energy_log_prefix.gather(1, patch_start)
        )
        patch_entropy = (
            torch.log(patch_energy + EPS)
            - patch_energy_log_energy / (patch_energy + EPS)
        )

        normalized_entropy = patch_entropy / (
            torch.log(patch_count.to(energy_per_freq.dtype).clamp(min=2.0))
            + EPS
        )
        # A one-bin remainder patch has H = 0 and concentration C = 1.
        normalized_entropy = torch.where(
            patch_count > 1,
            normalized_entropy,
            torch.zeros_like(normalized_entropy)
        )
        concentration = (1.0 - normalized_entropy).clamp(0.0, 1.0)

        # Entropy-aware selection score: G = E * (1 - normalized entropy).
        selection_score = energy_per_freq * concentration

        energy_low = selection_score.masked_fill(~low_mask, big_neg)
        energy_high = selection_score.masked_fill(~high_mask, big_neg)

        _, low_idx = torch.topk(energy_low, k_low_max, dim=1)
        _, high_idx = torch.topk(energy_high, k_high_max, dim=1)

        arange_low = torch.arange(k_low_max, device=device).unsqueeze(0)
        select_low = arange_low < low_k_all

        arange_high = torch.arange(k_high_max, device=device).unsqueeze(0)
        select_high = arange_high < high_k_all

        freq_band_mask = torch.zeros_like(energy_per_freq)
        freq_band_mask.scatter_(1, low_idx, select_low.float())
        freq_band_mask.scatter_(1, high_idx, select_high.float())
        freq_band_mask = freq_band_mask.unsqueeze(1)

        return freq_band_mask

    @torch.no_grad()
    def adaptive_mask(self, x_freq):
        B, N, freq_bins = x_freq.shape
        device = x_freq.device

        energy_per_freq = torch.abs(x_freq).pow(2).mean(dim=1)
        
        knee_indices = self._compute_kneepoint(energy_per_freq, freq_bins, device)
        low_mask, high_mask = self._create_frequency_masks(knee_indices, freq_bins, device)
        freq_band_mask = self._select_top_frequencies(
            energy_per_freq, low_mask, high_mask, device
        )

        return freq_band_mask

    def forward(self, x):
        B, N, L = x.shape
        
        x = self.dropout(x)
        x_freq = torch.fft.rfft(x, dim=2, norm='ortho')
        
        final_mask = self.adaptive_mask(x_freq)
        
        complex_weight1 = torch.view_as_complex(self.complex_weights[0])
        y1 = x_freq * complex_weight1
        out1 = torch.fft.irfft(y1, n=L, dim=2, norm="ortho")
        out1 = self.dropout(out1)
        
        complex_weight2 = torch.view_as_complex(self.complex_weights[1])
        y2 = x_freq * complex_weight2 * final_mask
        out2 = torch.fft.irfft(y2, n=L, dim=2, norm="ortho")
        out2 = self.dropout(out2)
        
        weights = F.softmax(self.filter_weights, dim=0)
        filtered_output = weights[0] * out1 + weights[1] * out2
        
        freq_pred = self.prediction_head(filtered_output)
        freq_pred = freq_pred.transpose(1, 2)
        
        return freq_pred


class CAP(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        freq = getattr(configs, 'freq', 'h')
        if freq == 'h':
            self.time_dim = 4
        elif freq == 't' or freq == 'min':
            self.time_dim = 5
        else:
            self.time_dim = 4
        
        self.rda = getattr(configs, 'rda')
        self.rdb = getattr(configs, 'rdb') 
        self.ksize = getattr(configs, 'ksize')
        self.ksh = getattr(configs, 'ksh')
        self.ksp = getattr(configs, 'ksp')
        
        self.channel_fusion = nn.Conv1d(self.ksp, 1, kernel_size=1)
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        
        self.global_rs_enc = nn.Sequential(
            nn.Linear(self.ksh, self.enc_in // self.rda),
            nn.LayerNorm(self.enc_in // self.rda),
            nn.ReLU(),
            nn.Linear(self.enc_in // self.rda, self.enc_in // self.rdb),
            nn.LayerNorm(self.enc_in // self.rdb),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.seq_len,
                out_channels=self.seq_len,
                kernel_size=self.ksize,
                padding='same'
            ),
            nn.Linear(self.enc_in // self.rdb, self.enc_in),
        )
        
    def forward(self, x, share_signal, specific_signal):
        B, L, N = x.shape
        
        global_rs_embed = self.global_rs_enc(share_signal)

        specific_signal_reshaped = specific_signal.reshape(B, L, N, self.ksp)
        specific_signal_conv = specific_signal_reshaped.permute(
            0, 2, 3, 1
        ).reshape(B * N, self.ksp, L)
        channel_fused_conv = self.channel_fusion(specific_signal_conv)
        channel_fused = channel_fused_conv.reshape(B, N, L).permute(0, 2, 1)

        # Full CAP combines shared/global and channel-specific representations.
        time_embed = global_rs_embed + channel_fused
        time_out = self.time_proj(time_embed.transpose(1, 2)).transpose(1, 2)
        
        return time_out


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout
        self.freq_band_ratio = configs.freq_band_ratio
        self.beta = configs.beta
        
        self.revin_layer = RevIN(num_features=configs.enc_in)
        
        self.freq_branch = AEF(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_size=self.hidden_size,
            num_filters=2,
            dropout=self.dropout,
            is_channel_filter=False,
            freq_band_ratio=self.freq_band_ratio
        )
        
        self.time_branch = CAP(configs)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                share_signal=None, specific_signal=None, mask=None):
        x = x_enc
        
        z = self.revin_layer(x, 'norm')
        z_trans = z.transpose(1, 2)
        freq_pred = self.freq_branch(z_trans)
        
        B, L, N = x.shape
        share_signal = share_signal.reshape(B, L, -1)
        specific_signal = specific_signal.reshape(B, L, -1)
        time_pred = self.time_branch(x, share_signal, specific_signal)
        
        final_pred = self.beta * freq_pred + (1 - self.beta) * time_pred
        output = self.revin_layer(final_pred, 'denorm')
        
        return output
