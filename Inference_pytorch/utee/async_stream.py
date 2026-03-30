import math
import numpy as np
import torch


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def compute_stream_stats(data, base_frequency, observation_window, jitter,
                         min_precision=4.0, max_precision=12.0):
    array = _to_numpy(data).astype(np.float64, copy=False)
    base_frequency = max(float(base_frequency), 1.0)
    observation_window = max(float(observation_window), 1.0)
    jitter = max(float(jitter), 0.0)
    if array.size == 0:
        return {
            "event_rate": 0.0,
            "duty_cycle": 0.5,
            "effective_frequency": base_frequency,
            "effective_precision": float(min_precision),
            "pulse_budget": 1.0,
            "snr": 1.0,
            "adc_latency_scale": 1.0,
            "adc_energy_scale": 1.0,
            "accum_latency_scale": 1.0,
            "accum_energy_scale": 1.0,
            "buffer_latency_scale": 1.0,
            "buffer_energy_scale": 1.0,
            "ic_latency_scale": 1.0,
            "ic_energy_scale": 1.0,
            "leakage_scale": 1.0,
            "latency_scale": 1.0,
            "energy_scale": 1.0,
        }

    max_abs = float(np.max(np.abs(array)))
    if max_abs < 1e-12:
        return {
            "event_rate": 0.0,
            "duty_cycle": 0.5,
            "effective_frequency": base_frequency * 0.5,
            "effective_precision": float(min_precision),
            "pulse_budget": 1.0,
            "snr": 1.0,
            "adc_latency_scale": max(observation_window / 32.0, 1.0),
            "adc_energy_scale": 0.25,
            "accum_latency_scale": 0.5,
            "accum_energy_scale": 0.2,
            "buffer_latency_scale": max(observation_window / 48.0, 0.5),
            "buffer_energy_scale": 0.15,
            "ic_latency_scale": max(observation_window / 48.0, 0.5),
            "ic_energy_scale": 0.1,
            "leakage_scale": max(observation_window / 32.0, 1.0),
            "latency_scale": max(observation_window / 32.0, 1.0),
            "energy_scale": 0.25,
        }

    normalized = np.clip(array / max_abs, -1.0, 1.0)
    event_threshold = max(0.15 * max_abs, 1e-6)
    event_rate = float(np.mean(np.abs(array) >= event_threshold))
    contrast = float(np.std(normalized))
    activity = min(1.0, max(event_rate, contrast))
    duty_cycle = float(np.mean((normalized + 1.0) / 2.0))

    effective_frequency = base_frequency * (0.5 + 0.5 * activity)
    frequency_ratio = effective_frequency / base_frequency
    pulse_budget = max(1.0, effective_frequency * observation_window * max(event_rate, 0.05))

    jitter_sigma = jitter / math.sqrt(pulse_budget)
    signal_strength = max(0.05, event_rate * (0.5 + contrast) * (0.5 + abs(duty_cycle - 0.5)))
    snr = max((signal_strength ** 2) / max(jitter_sigma ** 2, 1e-12), 1.0)
    effective_precision = np.clip(0.5 * math.log2(snr + 1.0) + 2.0, min_precision, max_precision)
    precision_ratio = max(effective_precision / max(max_precision, 1.0), 0.25)

    adc_latency_scale = max(0.2, (observation_window / 32.0) * (0.75 + 0.75 * precision_ratio) / max(frequency_ratio, 0.2) * (1.0 + 0.75 * jitter))
    adc_energy_scale = max(0.05, (0.4 + 0.6 * duty_cycle) * frequency_ratio * (0.7 + 0.8 * precision_ratio) * (1.0 + jitter))
    accum_latency_scale = max(0.2, (0.7 + 0.6 * precision_ratio) * (0.5 + 0.5 * event_rate))
    accum_energy_scale = max(0.05, (0.4 + 0.6 * event_rate) * (0.6 + 0.6 * precision_ratio))
    buffer_latency_scale = max(0.2, (observation_window / 40.0) * (0.6 + 0.6 * event_rate) / max(frequency_ratio, 0.25))
    buffer_energy_scale = max(0.05, (0.25 + 0.75 * event_rate) * frequency_ratio)
    ic_latency_scale = max(0.2, (observation_window / 48.0) * (0.5 + event_rate) / max(frequency_ratio, 0.25))
    ic_energy_scale = max(0.05, (0.2 + 0.8 * event_rate) * (0.7 + 0.3 * duty_cycle) * frequency_ratio)
    leakage_scale = max(0.5, observation_window / 32.0)

    latency_scale = max(0.25, 0.55 * adc_latency_scale + 0.2 * accum_latency_scale + 0.15 * buffer_latency_scale + 0.1 * ic_latency_scale)
    energy_scale = max(0.1, 0.5 * adc_energy_scale + 0.2 * accum_energy_scale + 0.15 * buffer_energy_scale + 0.15 * ic_energy_scale)

    return {
        "event_rate": event_rate,
        "duty_cycle": duty_cycle,
        "effective_frequency": effective_frequency,
        "effective_precision": float(effective_precision),
        "pulse_budget": float(pulse_budget),
        "snr": float(snr),
        "adc_latency_scale": float(adc_latency_scale),
        "adc_energy_scale": float(adc_energy_scale),
        "accum_latency_scale": float(accum_latency_scale),
        "accum_energy_scale": float(accum_energy_scale),
        "buffer_latency_scale": float(buffer_latency_scale),
        "buffer_energy_scale": float(buffer_energy_scale),
        "ic_latency_scale": float(ic_latency_scale),
        "ic_energy_scale": float(ic_energy_scale),
        "leakage_scale": float(leakage_scale),
        "latency_scale": float(latency_scale),
        "energy_scale": float(energy_scale),
    }


def quantize_tensor(tensor, bits):
    bits = max(int(round(bits)), 1)
    if bits >= 16:
        return tensor

    max_abs = tensor.detach().abs().max()
    if float(max_abs) < 1e-12:
        return tensor

    qmax = float((2 ** (bits - 1)) - 1)
    if qmax < 1:
        return tensor.sign() * max_abs

    scale = max_abs / qmax
    return torch.round(tensor / scale) * scale


def apply_stream_noise(tensor, stats, jitter):
    max_abs = tensor.detach().abs().max()
    if float(max_abs) < 1e-12 or jitter <= 0:
        return tensor

    pulse_budget = max(float(stats.get("pulse_budget", 1.0)), 1.0)
    noise_std = float(jitter) * float(max_abs) / math.sqrt(pulse_budget)
    return tensor + torch.randn_like(tensor) * noise_std


def write_stream_metadata(path, stats):
    ordered_keys = [
        "event_rate",
        "duty_cycle",
        "effective_frequency",
        "effective_precision",
        "pulse_budget",
        "snr",
        "adc_latency_scale",
        "adc_energy_scale",
        "accum_latency_scale",
        "accum_energy_scale",
        "buffer_latency_scale",
        "buffer_energy_scale",
        "ic_latency_scale",
        "ic_energy_scale",
        "leakage_scale",
        "latency_scale",
        "energy_scale",
    ]
    with open(path, "w") as handle:
        for key in ordered_keys:
            handle.write(f"{key}={float(stats.get(key, 0.0))}\n")
