import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Union

#CIWI-CKT: Chaos-Informed Wave Interference Meta-Learning and Cross-City Knowledge Transfer for Traffic Flow Forecasting
def _stat_chaos_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 20-dim statistical proxy for chaos features.
    Runs entirely on whichever device x lives on — no CPU/numpy round-trip.
    x: (B, T, D) or (B, D)
    returns: (B, 20)
    """
    if x.dim() == 3:
        ts = x.mean(dim=-1)
    else:
        ts = x
    B, T = ts.shape

    mu    = ts.mean(dim=1)
    sigma = ts.std(dim=1).clamp(min=1e-8)
    ts_n  = (ts - mu[:, None]) / sigma[:, None]

    diff1 = (ts[:, 1:] - ts[:, :-1]) if T > 1 else ts.new_zeros(B, 1)
    diff2 = (diff1[:, 1:] - diff1[:, :-1]) if T > 2 else ts.new_zeros(B, 1)

    t   = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)
    t_c = t - 0.5

    ac1 = (ts_n[:, :-1] * ts_n[:, 1:]).mean(dim=1) if T > 1 else ts.new_zeros(B)
    trend = (ts_n * t_c[None, :]).sum(dim=1) / (t_c.pow(2).sum() + 1e-8)
    half  = T // 2
    half_diff = ts_n[:, :half].mean(dim=1) - ts_n[:, half:].mean(dim=1)

    feats = torch.stack([
        mu,
        sigma,
        ts.var(dim=1),
        ts_n.abs().mean(dim=1),
        ts_n.pow(3).mean(dim=1),
        ts_n.pow(4).mean(dim=1) - 3,
        ac1,
        diff1.abs().mean(dim=1),
        diff1.pow(2).mean(dim=1),
        diff2.abs().mean(dim=1),
        ts.amax(dim=1) - ts.amin(dim=1),
        ts.amax(dim=1),
        ts.amin(dim=1),
        trend,
        half_diff,
        ts.median(dim=1).values,
        (ts > mu[:, None]).float().mean(dim=1),
        diff1.pow(2).sum(dim=1) / (T + 1e-8),
        ts_n.abs().amax(dim=1),
        sigma / (mu.abs() + 1e-8),
    ], dim=1)

    return feats.nan_to_num(0.0)


class ChaosAnalyzer:
    """Used only in datasets.py for one-time preprocessing — not called during training."""

    def __init__(self, embedding_dim=3, time_delay=1, max_embed_dim=10):
        self.embedding_dim = embedding_dim
        self.time_delay    = time_delay
        self.max_embed_dim = max_embed_dim
        self.feature_cache = {}

    def extract_chaos_features(self, time_series):
        if len(time_series) < 10:
            return np.zeros(20)
        ts = np.array(time_series).flatten()
        ts = ts[np.isfinite(ts)]
        if len(ts) < 10:
            return np.zeros(20)

        cache_key = hash(tuple(ts[:100])) if len(ts) > 100 else hash(tuple(ts))
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = np.zeros(20)
        features[0]  = self._largest_lyapunov_exponent(ts)
        features[1]  = self._hurst_exponent(ts)
        features[2]  = self._sample_entropy(ts)
        features[3]  = self._correlation_dimension(ts)
        features[4]  = self._box_counting_dimension(ts)
        features[5]  = self._recurrence_rate(ts)
        features[6]  = self._multifractal_spectrum(ts)
        features[7]  = self._approximate_entropy(ts)
        features[8]  = np.mean(ts)
        features[9]  = np.std(ts)
        features[10] = np.var(ts)
        features[11] = len(ts)

        embedded = self._phase_space_reconstruction(ts)
        if embedded.shape[0] > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(embedded)
            features[12] = np.mean(distances)
            features[13] = np.std(distances)
            features[14] = np.max(distances)
            features[15] = (np.min(distances[distances > 0])
                            if len(distances[distances > 0]) > 0 else 0.0)

        features[16] = self._coefficient_of_variation(ts)
        features[17] = self._autocorrelation_decay(ts)
        features[18] = self._trend_strength(ts)
        features[19] = self._seasonal_strength(ts)

        result = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(self.feature_cache) < 100:
            self.feature_cache[cache_key] = result
        return result

    def _coefficient_of_variation(self, ts):
        mean_val = np.mean(ts)
        return 0.0 if abs(mean_val) < 1e-8 else np.std(ts) / abs(mean_val)

    def _autocorrelation_decay(self, ts):
        if len(ts) < 10:
            return 0.0
        autocorrs = []
        for lag in range(1, min(10, len(ts) // 2)):
            if len(ts) > lag:
                corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        if not autocorrs:
            return 0.0
        x = np.arange(len(autocorrs))
        y = np.array(autocorrs)
        if len(x) > 2:
            decay_rate = -np.polyfit(x, np.log(y + 1e-8), 1)[0]
            return max(0.0, min(decay_rate, 5.0))
        return 0.0

    def _trend_strength(self, ts):
        if len(ts) < 3:
            return 0.0
        x = np.arange(len(ts))
        slope, _ = np.polyfit(x, ts, 1)
        return abs(slope) / (np.std(ts) + 1e-8)

    def _seasonal_strength(self, ts):
        if len(ts) < 12:
            return 0.0
        period   = min(12, len(ts) // 3)
        seasonal = np.zeros(len(ts))
        for i in range(len(ts)):
            seasonal[i] = np.mean(ts[max(0, i - period):min(len(ts), i + period + 1)])
        detrended    = ts - np.mean(ts)
        seasonal_var = np.var(seasonal - np.mean(seasonal))
        total_var    = np.var(detrended)
        return 0.0 if total_var < 1e-8 else seasonal_var / total_var

    def _phase_space_reconstruction(self, time_series):
        n = len(time_series)
        if n < self.embedding_dim * self.time_delay:
            return np.zeros((1, self.embedding_dim))
        num_points = n - (self.embedding_dim - 1) * self.time_delay
        indices    = (np.arange(num_points)[:, None]
                      + np.arange(self.embedding_dim)[None, :] * self.time_delay)
        return time_series[indices]

    def _largest_lyapunov_exponent(self, time_series):
        embedded = self._phase_space_reconstruction(time_series)
        if embedded.shape[0] < 10:
            return 0.0
        N          = embedded.shape[0]
        diff       = embedded[:, None, :] - embedded[None, :, :]
        distances  = np.sqrt(np.sum(diff ** 2, axis=2))
        lyapunov_sum, count = 0.0, 0
        for i in range(min(N - 20, 50)):
            mask = np.ones(N, dtype=bool)
            mask[max(0, i - 1):min(N, i + 2)] = False
            if np.any(mask):
                valid_distances = distances[i, mask]
                valid_indices   = np.where(mask)[0]
                min_idx         = valid_indices[np.argmin(valid_distances)]
                max_k = min(15, N - max(i, min_idx))
                if max_k > 1:
                    k_range    = np.arange(1, max_k)
                    i_valid    = i + k_range
                    j_valid    = min_idx + k_range
                    valid_mask = (i_valid < N) & (j_valid < N)
                    if np.any(valid_mask):
                        dists       = np.linalg.norm(embedded[i_valid[valid_mask]] - embedded[j_valid[valid_mask]], axis=1)
                        valid_dists = dists[dists > 0]
                        if len(valid_dists) > 3:
                            times = np.arange(1, len(valid_dists) + 1)
                            slope = np.polyfit(times, np.log(valid_dists), 1)[0]
                            lyapunov_sum += slope
                            count += 1
        return lyapunov_sum / max(count, 1)

    def _hurst_exponent(self, time_series):
        n = len(time_series)
        if n < 20:
            return 0.5
        sizes     = np.unique(np.logspace(1, np.log10(n // 2), 8).astype(int))
        rs_values = []
        for size in sizes:
            if size >= n:
                continue
            rs_window = []
            for i in range(n // size):
                window   = time_series[i * size:(i + 1) * size]
                detrended = window - np.mean(window)
                cumsum    = np.cumsum(detrended)
                R, S = np.ptp(cumsum), np.std(window)
                if S > 0:
                    rs_window.append(R / S)
            if rs_window:
                rs_values.append(np.mean(rs_window))
        if len(rs_values) < 3:
            return 0.5
        hurst = np.polyfit(np.log(sizes[:len(rs_values)]), np.log(rs_values), 1)[0]
        return max(0.0, min(hurst, 1.0))

    def _sample_entropy(self, time_series, m=2):
        N = len(time_series)
        if N < 10:
            return 0.0
        r = 0.2 * np.std(time_series)
        def _phi(m_):
            patterns  = np.array([time_series[i:i + m_] for i in range(N - m_ + 1)])
            phi_sum   = 0.0
            for i in range(min(N - m_ + 1, 200)):
                matches = np.sum(np.max(np.abs(patterns - patterns[i]), axis=1) <= r)
                if matches > 0:
                    phi_sum += np.log(matches / float(N - m_ + 1))
            return phi_sum / min(float(N - m_ + 1), 200.0)
        return _phi(m) - _phi(m + 1)

    def _correlation_dimension(self, time_series, max_r=None):
        embedded = self._phase_space_reconstruction(time_series)
        N = embedded.shape[0]
        if N < 20:
            return 1.0
        sample_size = min(N, 500)
        idx = np.random.choice(N, sample_size, replace=False)
        s   = embedded[idx]
        d   = np.sqrt(np.sum((s[:, None, :] - s[None, :, :]) ** 2, axis=2))
        if max_r is None:
            max_r = np.percentile(d[np.triu_indices(sample_size, k=1)], 90)
        r_values = np.logspace(np.log10(max_r / 100 + 1e-10), np.log10(max_r + 1e-10), 15)
        log_c    = []
        for r in r_values:
            count = max(np.sum(d < r) - sample_size, 1)
            log_c.append(np.log(count / (sample_size * (sample_size - 1))))
        start, end = int(0.2 * len(r_values)), int(0.8 * len(r_values))
        if end > start + 2:
            return max(0.1, min(np.polyfit(np.log(r_values[start:end]), log_c[start:end], 1)[0], 10.0))
        return 1.0

    def _box_counting_dimension(self, time_series):
        embedded = self._phase_space_reconstruction(time_series)
        if embedded.shape[0] < 10:
            return 1.0
        rng = np.ptp(embedded, axis=0)
        rng[rng == 0] = 1
        normalized   = (embedded - embedded.min(axis=0)) / rng
        box_sizes    = np.logspace(-2, 0, 8)
        box_counts   = [len(set(map(tuple, np.floor(normalized / bs).astype(int)))) for bs in box_sizes]
        if len(box_sizes) > 3:
            return max(0.1, min(np.polyfit(np.log(1 / box_sizes), np.log(box_counts), 1)[0], 3.0))
        return 1.0

    def _recurrence_rate(self, time_series, threshold_percent=10):
        embedded = self._phase_space_reconstruction(time_series)
        N = embedded.shape[0]
        if N < 10:
            return 0.0
        sample_size = min(N, 200)
        idx = np.random.choice(N, sample_size, replace=False)
        s   = embedded[idx]
        d   = np.sqrt(np.sum((s[:, None, :] - s[None, :, :]) ** 2, axis=2))
        return np.sum(d < np.percentile(d, threshold_percent)) / (sample_size ** 2)

    def _multifractal_spectrum(self, time_series):
        if len(time_series) < 32:
            return 0.0
        N        = len(time_series)
        q_values = np.arange(-3, 4, 2)
        tau_q    = []
        for q in q_values:
            scales          = np.unique(np.logspace(1, np.log10(N // 4), 6).astype(int))
            log_scales, log_flucts = [], []
            for scale in scales:
                if scale >= N:
                    continue
                flucts = []
                for i in range(min(N // scale, 50)):
                    seg = time_series[i * scale:(i + 1) * scale]
                    x   = np.arange(len(seg))
                    trend = np.polyval(np.polyfit(x, seg, 1), x)
                    f     = np.sqrt(np.mean((seg - trend) ** 2))
                    if f > 0:
                        flucts.append(f)
                if flucts:
                    flucts = np.array(flucts)
                    avg    = (np.exp(np.mean(np.log(flucts))) if q == 0
                              else np.mean(flucts ** q) ** (1 / q))
                    log_scales.append(np.log(scale))
                    log_flucts.append(np.log(avg))
            if len(log_scales) > 2:
                tau_q.append(np.polyfit(log_scales, log_flucts, 1)[0])
            else:
                tau_q.append(0.0)
        return float(np.max(tau_q) - np.min(tau_q)) if len(tau_q) > 1 else 0.0

    def _approximate_entropy(self, time_series, m=2, r=None):
        N = len(time_series)
        if N < 10:
            return 0.0
        if r is None:
            r = 0.2 * np.std(time_series)
        def _phi(m_):
            patterns = np.array([time_series[i:i + m_] for i in range(N - m_ + 1)])
            phi_sum  = 0.0
            for i in range(min(N - m_ + 1, 200)):
                matches = np.sum(np.max(np.abs(patterns - patterns[i]), axis=1) <= r)
                if matches > 0:
                    phi_sum += np.log(matches / float(N - m_ + 1))
            return phi_sum / min(float(N - m_ + 1), 200.0)
        return _phi(m) - _phi(m + 1)


class ChaosAwareEncoder(nn.Module):
    def __init__(self, input_dim: int, chaos_dim: int = 20):
        super().__init__()
        self.input_dim = input_dim
        self.chaos_dim = chaos_dim

        self.chaos_feature_processor = nn.Sequential(
            nn.Linear(20, chaos_dim),
            nn.LayerNorm(chaos_dim),
            nn.GELU(),
            nn.Linear(chaos_dim, chaos_dim),
            nn.Tanh()
        )
        self.learnable_statistics = nn.Sequential(
            nn.Linear(input_dim, chaos_dim),
            nn.LayerNorm(chaos_dim),
            nn.GELU(),
            nn.Linear(chaos_dim, max(chaos_dim // 2, 4)),
            nn.Tanh()
        )
        fusion_input_dim = chaos_dim + max(chaos_dim // 2, 4)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, chaos_dim),
            nn.LayerNorm(chaos_dim),
            nn.GELU(),
            nn.Linear(chaos_dim, chaos_dim),
            nn.Tanh()
        )
        self.lyapunov_estimator = nn.Sequential(
            nn.Linear(chaos_dim, 1),
            nn.Sigmoid()
        )
        self.attractor_mapper = nn.Sequential(
            nn.Linear(chaos_dim, chaos_dim * 2),
            nn.LayerNorm(chaos_dim * 2),
            nn.GELU(),
            nn.Linear(chaos_dim * 2, chaos_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x_mean = x.mean(dim=1)
        elif x.dim() == 2:
            x_mean = x
        else:
            raise ValueError(f"Unexpected input dimension: {x.shape}")

        if x_mean.size(-1) != self.input_dim:
            if not hasattr(self, 'adaptive_projection'):
                self.adaptive_projection = nn.Linear(x_mean.size(-1), self.input_dim).to(x.device)
            x_mean = self.adaptive_projection(x_mean)

        raw_chaos = _stat_chaos_proxy(x)
        processed_chaos    = self.chaos_feature_processor(raw_chaos)
        learnable_features = self.learnable_statistics(x_mean)
        fused_features     = self.feature_fusion(torch.cat([processed_chaos, learnable_features], dim=-1))
        lyapunov_exp       = self.lyapunov_estimator(fused_features)
        attractor_dims     = self.attractor_mapper(fused_features)

        return {
            'chaos_features':     fused_features,
            'lyapunov_exponent':  lyapunov_exp,
            'attractor_dimensions': attractor_dims,
            'raw_chaos_features': processed_chaos,
            'learnable_features': learnable_features,
        }


class AdvancedWaveGenerator(nn.Module):
    def __init__(self, input_dim: int, wave_components: int = 12, chaos_dim: int = 20):
        super().__init__()
        self.input_dim       = input_dim
        self.wave_components = wave_components
        self.chaos_dim       = chaos_dim

        self.chaos_encoder = ChaosAwareEncoder(input_dim, chaos_dim)
        enhanced_dim = input_dim + chaos_dim

        self.wave_amplitude_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enhanced_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            ) for _ in range(wave_components)
        ])
        self.frequency_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enhanced_dim, max(input_dim // 2, 4)),
                nn.GELU(),
                nn.Linear(max(input_dim // 2, 4), input_dim),
                nn.Tanh()
            ) for _ in range(wave_components)
        ])
        self.wave_numbers = nn.ParameterList([
            nn.Parameter(torch.ones(1, input_dim) * (i + 1) * 0.15)
            for i in range(wave_components)
        ])
        self.phase_shifts = nn.ParameterList([
            nn.Parameter(torch.randn(1, input_dim) * 0.05)
            for _ in range(wave_components)
        ])

    def forward(self, x: torch.Tensor,
                meta_context: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size, seq_len, input_dim = x.shape

        if input_dim != self.input_dim:
            if not hasattr(self, 'adaptive_input_projection'):
                self.adaptive_input_projection = nn.Linear(input_dim, self.input_dim).to(x.device)
            x_proj = self.adaptive_input_projection(x)
        else:
            x_proj = x

        chaos_outputs  = self.chaos_encoder(x_proj)
        time_coords    = torch.linspace(0, 2 * math.pi, seq_len, device=x.device).view(1, seq_len, 1)
        chaos_features = chaos_outputs['chaos_features'].unsqueeze(1).expand(-1, seq_len, -1)
        x_enhanced     = torch.cat([x_proj, chaos_features], dim=-1)

        if meta_context is not None:
            if meta_context.dim() == 1:
                meta_context = meta_context.unsqueeze(0)
            if meta_context.size(0) != batch_size:
                meta_context = meta_context.expand(batch_size, -1)
            if meta_context.size(-1) != self.chaos_dim:
                meta_features = (meta_context[:, :self.chaos_dim]
                                 if meta_context.size(-1) > self.chaos_dim
                                 else F.pad(meta_context, (0, self.chaos_dim - meta_context.size(-1))))
            else:
                meta_features = meta_context
            chaos_features_enhanced = chaos_outputs['chaos_features'] + 0.1 * meta_features
        else:
            chaos_features_enhanced = chaos_outputs['chaos_features']

        phase_offsets = [0, 0, math.pi / 3, math.pi / 6, math.pi / 2, math.pi / 4]
        use_sin = [True, False, True, False, True, False]
        waves   = []
        for i in range(self.wave_components):
            amplitude   = self.wave_amplitude_generators[i](x_enhanced)
            freq_mod    = self.frequency_modulators[i](x_enhanced.mean(dim=1, keepdim=True))
            wave_phase  = self.wave_numbers[i] * (1 + 0.2 * freq_mod) * time_coords + self.phase_shifts[i]
            phase_off   = phase_offsets[i % 6]
            wave        = (amplitude * torch.sin(wave_phase + phase_off)
                           if use_sin[i % 6]
                           else amplitude * torch.cos(wave_phase + phase_off))
            waves.append(wave)

        chaos_outputs['meta_enhanced_features'] = chaos_features_enhanced
        return waves, chaos_outputs


class PredictabilityModule(nn.Module):
    def __init__(self, input_dim: int, wave_components: int = 12, meta_dim: int = 64):
        super().__init__()
        self.input_dim       = input_dim
        self.wave_components = wave_components

        self.stability_analyzer = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(input_dim // 2, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.consistency_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
                nn.Tanh()
            ) for _ in range(wave_components)
        ])
        self.confidence_scorer = nn.Sequential(
            nn.Linear(meta_dim + input_dim + 20, meta_dim),
            nn.LayerNorm(meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, 1),
            nn.Sigmoid()
        )
        self.chaos_predictability_assessor = nn.Sequential(
            nn.Linear(20, meta_dim // 2),
            nn.GELU(),
            nn.Linear(meta_dim // 2, 1),
            nn.Sigmoid()
        )

    def analyze_wave_chaos(self, waves: List[torch.Tensor]) -> torch.Tensor:
        chaos_scores = []
        for wave in waves:
            chaos_feat = _stat_chaos_proxy(wave)
            score      = self.chaos_predictability_assessor(chaos_feat)
            chaos_scores.append(score)
        return torch.stack(chaos_scores, dim=1).mean(dim=1)

    def analyze_temporal_stability(self, waves: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([
            self.stability_analyzer(w.transpose(1, 2)).mean(dim=(1, 2))
            for w in waves
        ], dim=1)

    def enhance_pattern_consistency(self, waves: List[torch.Tensor]) -> List[torch.Tensor]:
        return [w + 0.1 * self.consistency_enhancer[i](w) for i, w in enumerate(waves)]

    def compute_predictability_score(self, waves: List[torch.Tensor],
                                     meta_context: torch.Tensor) -> torch.Tensor:
        wave_features = torch.stack([w.mean(dim=1) for w in waves], dim=1).mean(dim=1)
        combined      = torch.stack(waves, dim=1).mean(dim=1)
        chaos_tensor  = _stat_chaos_proxy(combined)
        confidence    = self.confidence_scorer(torch.cat([meta_context, wave_features, chaos_tensor], dim=-1))
        return confidence.squeeze(-1)

    def forward(self, support_waves: List[torch.Tensor],
                query_waves: List[torch.Tensor],
                meta_context: torch.Tensor) -> Dict[str, any]:
        support_chaos = self.analyze_wave_chaos(support_waves)
        query_chaos   = self.analyze_wave_chaos(query_waves)
        support_consistent = self.enhance_pattern_consistency(support_waves)
        query_consistent   = self.enhance_pattern_consistency(query_waves)
        all_waves           = support_consistent + query_consistent
        predictability      = self.compute_predictability_score(all_waves, meta_context)
        combined_chaos      = (support_chaos + query_chaos) / 2
        final_predictability = (predictability + combined_chaos) / 2

        return {
            'support':              support_consistent,
            'query':                query_consistent,
            'predictability_score': final_predictability,
            'chaos_predictability': combined_chaos,
            'stability_scores': {
                'support': self.analyze_temporal_stability(support_waves),
                'query':   self.analyze_temporal_stability(query_waves),
            },
        }


class MetaInterferenceProcessor(nn.Module):
    def __init__(self, input_dim: int, wave_components: int = 12, meta_dim: int = 64):
        super().__init__()
        self.input_dim       = input_dim
        self.wave_components = wave_components
        self.meta_dim        = meta_dim

        self.wave_projector = nn.Sequential(
            nn.Linear(input_dim, meta_dim),
            nn.LayerNorm(meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, meta_dim)
        )
        self.predictability_enhancer = PredictabilityModule(input_dim, wave_components, meta_dim)
        self.interference_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(meta_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            ) for _ in range(wave_components)
        ])
        self.pattern_enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.chaos_interference_modulator = nn.Sequential(
            nn.Linear(20 + meta_dim, meta_dim),
            nn.LayerNorm(meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, input_dim),
            nn.Tanh()
        )

    def compute_adaptive_interference(self, waves: List[torch.Tensor],
                                      meta_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_interference = sum(
            self.interference_weights[i](meta_context).unsqueeze(1) * waves[i]
            for i in range(len(self.interference_weights))
        )
        chaos_features   = _stat_chaos_proxy(combined_interference)
        chaos_modulation = self.chaos_interference_modulator(torch.cat([chaos_features, meta_context], dim=-1))
        modulated        = combined_interference + 0.1 * chaos_modulation.unsqueeze(1)
        return modulated, chaos_features

    def forward(self, support_waves: List[torch.Tensor],
                query_waves: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        wave_features = [self.wave_projector(w.mean(dim=1)) for w in support_waves + query_waves]
        meta_context  = torch.stack(wave_features).mean(dim=0)

        enhanced = self.predictability_enhancer(support_waves, query_waves, meta_context)

        support_interference, support_chaos = self.compute_adaptive_interference(enhanced['support'], meta_context)
        query_interference,   query_chaos   = self.compute_adaptive_interference(enhanced['query'],   meta_context)

        final_interference   = support_interference + query_interference
        final_chaos_features = _stat_chaos_proxy(final_interference)
        enhanced_pattern     = self.pattern_enhancer(final_interference.mean(dim=1))

        return {
            'interference_pattern':      final_interference,
            'meta_context':              meta_context,
            'enhanced_pattern':          enhanced_pattern,
            'predictability_score':      enhanced['predictability_score'],
            'interference_chaos_features': final_chaos_features,
            'support_chaos_features':    support_chaos,
            'query_chaos_features':      query_chaos,
        }


class WaveAttention(nn.Module):
    def __init__(self, input_dim: int, frequency_bands: int = 16, num_heads: int = 8):
        super().__init__()
        self.input_dim       = input_dim
        self.frequency_bands = frequency_bands
        self.num_heads       = num_heads

        self.multiscale_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
            for _ in range(3)
        ])
        self.temporal_fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, interference_features: torch.Tensor) -> torch.Tensor:
        attended_features = [
            attn(interference_features, interference_features, interference_features)[0]
            for attn in self.multiscale_attention
        ]
        return self.temporal_fusion(torch.cat(attended_features, dim=-1)) + interference_features


class MetaLearningPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, meta_dim: int = 64):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.meta_dim   = meta_dim

        self.support_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.meta_prototype_generator = nn.Sequential(
            nn.Linear(hidden_dim, meta_dim), nn.LayerNorm(meta_dim), nn.GELU(),
            nn.Linear(meta_dim, meta_dim)
        )
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, support_features: torch.Tensor, query_features: torch.Tensor,
                external_meta_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if support_features.size(-1) != self.input_dim:
            if not hasattr(self, 'support_projection'):
                self.support_projection = nn.Linear(support_features.size(-1), self.input_dim).to(support_features.device)
            support_features = self.support_projection(support_features)
        if query_features.size(-1) != self.input_dim:
            if not hasattr(self, 'query_projection'):
                self.query_projection = nn.Linear(query_features.size(-1), self.input_dim).to(query_features.device)
            query_features = self.query_projection(query_features)

        support_encoded   = self.support_encoder(support_features)
        query_encoded     = self.query_encoder(query_features)
        support_prototype = self.meta_prototype_generator(support_encoded.mean(dim=1, keepdim=True))

        if external_meta_context is not None:
            if external_meta_context.size(-1) != self.meta_dim:
                if not hasattr(self, 'meta_projection'):
                    self.meta_projection = nn.Linear(external_meta_context.size(-1), self.meta_dim).to(external_meta_context.device)
                external_meta_context = self.meta_projection(external_meta_context)
            meta_context = external_meta_context + support_prototype.squeeze(1)
        else:
            meta_context = support_prototype.squeeze(1)

        adapted_context = meta_context.unsqueeze(1).expand(-1, query_encoded.size(1), -1)
        return self.adaptive_fusion(torch.cat([query_encoded, adapted_context], dim=-1))


class OptimizedInputProjection(nn.Module):
    def __init__(self, target_dim: int, max_input_dim: int = 2000):
        super().__init__()
        self.target_dim    = target_dim
        self.max_input_dim = max_input_dim
        self.projection    = nn.Linear(max_input_dim, target_dim)
        self.final_norm    = nn.LayerNorm(target_dim)
        self.final_transform = nn.Sequential(nn.GELU(), nn.Linear(target_dim, target_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, input_dim = x.shape
        if input_dim > self.max_input_dim:
            x = x[:, :, :self.max_input_dim]
        elif input_dim < self.max_input_dim:
            x = F.pad(x, (0, self.max_input_dim - input_dim))
        return self.final_transform(self.final_norm(self.projection(x)))



class CityPrototypeBank(nn.Module):
    """
    Cross-city knowledge transfer via learnable per-city prototype embeddings.
    """
    def __init__(self, city_names: List[str], prototype_dim: int):
        super().__init__()
        self.city_names  = list(city_names)
        self.city_to_idx = {name: i for i, name in enumerate(self.city_names)}
        n                = len(self.city_names)
        self.prototype_dim = prototype_dim

        self.prototypes = nn.Embedding(n, prototype_dim)
        nn.init.normal_(self.prototypes.weight, 0.0, 0.02)

        # Current city attends to ALL city prototypes to gather transferable info
        n_heads = max(1, prototype_dim // 16)
        self.cross_attn = nn.MultiheadAttention(
            prototype_dim, num_heads=n_heads, batch_first=True, dropout=0.0
        )
        # Gate decides how much cross-city context to mix into own prototype
        self.transfer_gate = nn.Sequential(
            nn.Linear(prototype_dim * 2, prototype_dim),
            nn.Sigmoid()
        )
        # Projects batch features to prototype space for alignment loss
        self.feature_projector = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.Tanh()
        )

    def register_city(self, city_name: str):
        """Dynamically add a new city (e.g. a target city seen at fine-tune time)."""
        if city_name not in self.city_to_idx:
            idx = len(self.city_names)
            self.city_names.append(city_name)
            self.city_to_idx[city_name] = idx
            # Initialise new prototype as mean of existing ones
            new_proto = self.prototypes.weight.data.mean(dim=0, keepdim=True)
            new_embed = nn.Embedding(idx + 1, self.prototype_dim)
            new_embed.weight.data[:idx] = self.prototypes.weight.data
            new_embed.weight.data[idx]  = new_proto
            self.prototypes = new_embed

    def forward(self, city_name: str,
                batch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        city_name      : name of the city this batch belongs to
        batch_features : (B, prototype_dim) encoded representation of the batch
        """
        if city_name not in self.city_to_idx:
            self.register_city(city_name)

        B      = batch_features.size(0)
        device = batch_features.device

        if self.prototypes.weight.device != device:
            self.prototypes = self.prototypes.to(device)

        idx        = self.city_to_idx[city_name]
        city_proto = self.prototypes.weight[idx]          # (D,)
        all_protos = self.prototypes.weight               # (N, D)

        # -- cross-city attention --
        q         = city_proto.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        kv        = all_protos.unsqueeze(0)               # (1, N, D)
        cross_ctx, _ = self.cross_attn(q, kv, kv)
        cross_ctx = cross_ctx.squeeze(0).squeeze(0)       # (D,)

        # -- gated fusion --
        gate        = self.transfer_gate(torch.cat([city_proto, cross_ctx], dim=-1))
        transferred = gate * city_proto + (1.0 - gate) * cross_ctx   # (D,)

        # -- alignment loss --
        proj_feat        = self.feature_projector(batch_features)         # (B, D)
        mean_feat        = proj_feat.mean(dim=0)                          # (D,)
        self_consistency = 1.0 - F.cosine_similarity(
            F.normalize(city_proto.unsqueeze(0), dim=-1),
            F.normalize(mean_feat.unsqueeze(0),  dim=-1)
        ).squeeze()

        p_norm    = F.normalize(all_protos, dim=-1)                       # (N, D)
        sim_mat   = p_norm @ p_norm.T                                     # (N, N)
        n_cities  = len(self.city_names)
        off_diag  = sim_mat[~torch.eye(n_cities, dtype=torch.bool, device=device)]
        diversity = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(0.0, device=device)

        alignment_loss = torch.clamp(self_consistency + 0.1 * diversity, 0.0, 5.0)

        return {
            'city_prototype':     city_proto.unsqueeze(0).expand(B, -1),   # (B, D)
            'cross_city_context': transferred.unsqueeze(0).expand(B, -1),  # (B, D)
            'alignment_loss':     alignment_loss,
        }


class ChaosWaveFSL(nn.Module):
    def __init__(self,
                 target_dim: int = 16,
                 hidden_dim: int = 16,
                 output_dim: int = 1,
                 wave_components: int = 12,
                 frequency_bands: int = 16,
                 num_heads: int = 8,
                 prediction_horizon: int = 12,
                 chaos_dim: int = 20,
                 meta_dim: int = 64,
                 kernel_sizes: List[int] = None,
                 max_input_dim: int = 2000,
                 city_names: List[str] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]
        if city_names is None:
            city_names = ['default']

        self.target_dim         = target_dim
        self.hidden_dim         = hidden_dim
        self.output_dim         = output_dim
        self.prediction_horizon = prediction_horizon
        self.meta_dim           = meta_dim

        self.input_projection          = OptimizedInputProjection(target_dim, max_input_dim)
        self.support_wave_generator    = AdvancedWaveGenerator(target_dim, wave_components, chaos_dim)
        self.query_wave_generator      = AdvancedWaveGenerator(target_dim, wave_components, chaos_dim)
        self.meta_interference_processor = MetaInterferenceProcessor(target_dim, wave_components, meta_dim)
        self.enhanced_wave_attention   = WaveAttention(target_dim, frequency_bands, num_heads)
        self.meta_learning_predictor   = MetaLearningPredictor(target_dim, hidden_dim, meta_dim)

        # Cross-city knowledge transfer module
        self.city_prototype_bank = CityPrototypeBank(city_names, meta_dim)
        # Projects cross-city context to same dim as meta_context for injection
        self.city_context_injector = nn.Sequential(
            nn.Linear(meta_dim * 2, meta_dim),
            nn.LayerNorm(meta_dim),
            nn.GELU(),
        )

        self.temporal_convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(target_dim, target_dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(target_dim),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.feature_integration = nn.Sequential(
            nn.Linear(target_dim * len(kernel_sizes), target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Linear(target_dim * 2, target_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prediction_horizon * output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, support_data: torch.Tensor,
                query_data: torch.Tensor,
                meta_context: Optional[torch.Tensor] = None,
                city_name: Optional[str] = None) -> Dict[str, torch.Tensor]:

        support_features = self.input_projection(support_data)
        query_features   = self.input_projection(query_data)

        support_waves, _ = self.support_wave_generator(support_features, meta_context)
        query_waves,   _ = self.query_wave_generator(query_features,   meta_context)

        interference_results   = self.meta_interference_processor(support_waves, query_waves)
        interference_features  = interference_results['interference_pattern']
        meta_context_extracted = interference_results['meta_context']   # (B, meta_dim)

        # -- Cross-city knowledge transfer --
        # Inject city prototype + cross-city attended context into meta_context
        # so every subsequent module is conditioned on city-level knowledge.
        city_name_resolved = city_name or 'default'
        city_outputs = self.city_prototype_bank(city_name_resolved, meta_context_extracted)
        enriched_meta = self.city_context_injector(
            torch.cat([meta_context_extracted, city_outputs['cross_city_context']], dim=-1)
        )                                                                # (B, meta_dim)

        attended_features = self.enhanced_wave_attention(interference_features)
        adapted_features  = self.meta_learning_predictor(support_features, attended_features, enriched_meta)

        multiscale_features = [
            conv(adapted_features.transpose(1, 2)).transpose(1, 2)
            for conv in self.temporal_convolutions
        ]
        integrated_features = self.feature_integration(torch.cat(multiscale_features, dim=-1))
        final_features      = integrated_features + adapted_features
        predictions         = self.prediction_head(final_features[:, -1, :])
        predictions         = predictions.view(-1, self.prediction_horizon, self.output_dim)

        return {
            'predictions':              predictions,
            'interference_pattern':     interference_results['interference_pattern'],
            'meta_context':             enriched_meta,
            'predictability_score':     interference_results['predictability_score'],
            'interference_chaos_features': interference_results['interference_chaos_features'],
            'city_prototype':           city_outputs['city_prototype'],
            'cross_city_context':       city_outputs['cross_city_context'],
            'city_alignment_loss':      city_outputs['alignment_loss'],
        }


class ChaosWaveLoss(nn.Module):
    def __init__(self,
                 prediction_weight: float = 1.0,
                 harmony_weight: float = 0.15,
                 interference_weight: float = 0.08,
                 smoothness_weight: float = 0.05,
                 consistency_weight: float = 0.08,
                 meta_weight: float = 0.1,
                 predictability_weight: float = 0.05,
                 city_alignment_weight: float = 0.1):
        super().__init__()
        self.prediction_weight     = prediction_weight
        self.harmony_weight        = harmony_weight
        self.interference_weight   = interference_weight
        self.smoothness_weight     = smoothness_weight
        self.consistency_weight    = consistency_weight
        self.meta_weight           = meta_weight
        self.predictability_weight = predictability_weight
        self.city_alignment_weight = city_alignment_weight

    def prediction_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.clamp(predictions, -5.0, 5.0)
        targets     = torch.clamp(targets, -5.0, 5.0)
        if predictions.shape != targets.shape:
            if predictions.dim() == 3 and targets.dim() == 3:
                n = min(predictions.size(1), targets.size(1))
                predictions, targets = predictions[:, :n], targets[:, :n]
            elif predictions.dim() == 3 and targets.dim() == 2:
                predictions = predictions.mean(dim=1)
            elif predictions.dim() == 2 and targets.dim() == 3:
                targets = targets.mean(dim=1)
        return (0.5 * F.mse_loss(predictions, targets)
                + 0.3 * F.l1_loss(predictions, targets)
                + 0.2 * F.huber_loss(predictions, targets, delta=0.5))

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions          = torch.nan_to_num(outputs['predictions'],         nan=0.0)
        targets              = torch.nan_to_num(targets,                        nan=0.0)
        interference_pattern = outputs.get('interference_pattern', torch.zeros_like(predictions))
        meta_context         = outputs.get('meta_context', torch.zeros(predictions.size(0), 64, device=predictions.device))
        predictability_score = outputs.get('predictability_score', torch.ones(predictions.size(0), device=predictions.device))
        meta_context         = torch.nan_to_num(meta_context, nan=0.0)
        predictability_score = torch.nan_to_num(predictability_score, nan=0.5)

        pred_loss         = torch.clamp(self.prediction_loss(predictions, targets), 0.0, 100.0)
        interference_loss = torch.clamp(
            torch.mean(torch.abs(interference_pattern[:, 1:] - interference_pattern[:, :-1])) * 0.1, 0.0, 10.0)
        meta_loss         = torch.clamp(torch.norm(meta_context, p=2, dim=-1).mean() * 0.1, 0.01, 10.0)
        smoothness_loss   = torch.clamp(
            torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1])) * 0.1
            if predictions.size(1) > 1 else torch.tensor(0.0, device=predictions.device), 0.0, 10.0)
        predictability_loss = torch.clamp(
            F.mse_loss(predictability_score, torch.ones_like(predictability_score) * 0.8) * 0.1, 0.0, 10.0)

        city_alignment_loss = torch.clamp(
            outputs.get('city_alignment_loss', torch.tensor(0.0, device=predictions.device)),
            0.0, 5.0
        )

        total_loss = (self.prediction_weight      * pred_loss
                      + self.harmony_weight       * meta_loss
                      + self.interference_weight  * interference_loss
                      + self.smoothness_weight    * smoothness_loss
                      + self.predictability_weight * predictability_loss
                      + self.city_alignment_weight * city_alignment_loss)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.1, device=predictions.device, requires_grad=True)

        return {
            'total_loss':           total_loss,
            'prediction_loss':      pred_loss,
            'interference_loss':    interference_loss,
            'meta_loss':            meta_loss,
            'smoothness_loss':      smoothness_loss,
            'predictability_loss':  predictability_loss,
            'city_alignment_loss':  city_alignment_loss,
        }


def create_enhanced_model(config: Dict) -> ChaosWaveFSL:
    model_config = config.get('model', {})
    task_config  = config.get('task', {})
    data_config  = config.get('data', {})
    target_dim   = int(model_config.get('hidden_dim', 16))
    chaos_dim    = int(model_config.get('chaos_dim', 20))
    meta_dim     = int(model_config.get('meta_dim', 64))
    city_names   = list(data_config.get('data_keys', ['default']))
    print(f"Creating model with target_dim={target_dim}, chaos_dim={chaos_dim}, meta_dim={meta_dim}")
    print(f"City prototype bank: {city_names}")
    return ChaosWaveFSL(
        target_dim         = target_dim,
        hidden_dim         = target_dim,
        output_dim         = int(model_config.get('output_dim', 1)),
        wave_components    = int(model_config.get('wave_components', 12)),
        frequency_bands    = int(model_config.get('frequency_bands', 16)),
        num_heads          = int(model_config.get('num_heads', 8)),
        prediction_horizon = int(task_config.get('pred_num', 12)),
        chaos_dim          = chaos_dim,
        meta_dim           = meta_dim,
        kernel_sizes       = model_config.get('kernel_sizes', [3, 5, 7, 9]),
        city_names         = city_names,
    )


def create_enhanced_loss(config: Dict) -> ChaosWaveLoss:
    model_config = config.get('model', {})
    return ChaosWaveLoss(
        prediction_weight     = float(model_config.get('prediction_weight',     1.0)),
        harmony_weight        = float(model_config.get('harmony_weight',        0.15)),
        interference_weight   = float(model_config.get('interference_weight',   0.08)),
        smoothness_weight     = float(model_config.get('smoothness_weight',     0.05)),
        consistency_weight    = float(model_config.get('consistency_weight',    0.08)),
        meta_weight           = float(model_config.get('meta_weight',           0.1)),
        predictability_weight = float(model_config.get('predictability_weight', 0.05)),
        city_alignment_weight = float(model_config.get('city_alignment_weight', 0.1)),
    )