# Async Stream NeuroSim Prototype

This document describes a lightweight asynchronous stream-based CIM extension built on top of the original `DNN+NeuroSim V1.4` flow.

The objective of this prototype is simple:

- keep the original NeuroSim hardware estimation path intact
- add a clean async-stream frontend for simple CNN inference
- export per-layer stream statistics
- rescale hardware metrics in the C++ backend using stream-aware factors

This is intended for early-stage architecture exploration, not signoff-level circuit verification.

## What Was Added

The prototype introduces a new inference path for simple CNN evaluation:

- New inference mode: `ASYNC`
- New dataset option: `mnist`
- New simple network: `StreamCNN`
- New Python-side stream behavior model
- New per-layer stream metadata export
- New C++-side stream-aware hardware scaling

The goal is to support early exploration of event-driven / asynchronous stream-based CIM behavior without rewriting the entire NeuroSim backend.

## Main Idea

The implementation preserves the original NeuroSim hardware estimation flow and augments it with per-layer stream statistics.

At a high level:

1. A simple CNN runs in Python using stream-aware layers.
2. Each layer estimates stream activity statistics such as:
   - event rate
   - duty cycle
   - effective frequency
   - effective precision
   - pulse budget
3. The standard NeuroSim traces are still generated:
   - `weight*.csv`
   - `input*.csv`
4. Additional per-layer metadata files are generated:
   - `meta*.txt`
5. The C++ NeuroSim backend reads those metadata files and rescales:
   - ADC latency / energy
   - accumulation latency / energy
   - buffer latency / energy
   - interconnect latency / energy
   - leakage

This makes the prototype stream-aware while preserving the original directory structure and most of the original code path.

## What This Prototype Can Do

The current prototype can:

- run a simple `MNIST` CNN (`StreamCNN`)
- estimate per-layer async stream behavior
- export `weight`, `input`, and `metadata` traces
- feed those traces into the original NeuroSim C++ backend
- report stream-aware hardware changes at the module level:
  - ADC / sensing
  - accumulation
  - buffer
  - interconnect
  - leakage

## Files Added

- `Inference_pytorch/utee/async_stream.py`
  - Computes lightweight async-stream statistics and writes metadata files.
- `Inference_pytorch/modules/async_stream_infer.py`
  - Defines `AsyncStreamConv2d` and `AsyncStreamLinear`.
- `Inference_pytorch/models/StreamCNN.py`
  - A minimal CNN for `MNIST`.
- `Inference_pytorch/NeuroSIM/NetWork_StreamCNN.csv`
  - Network structure file for the C++ backend.

## Files Modified

- `Inference_pytorch/inference.py`
  - Adds `ASYNC` mode, `mnist` dataset support, stream parameters, and `StreamCNN` selection.
- `Inference_pytorch/models/dataset.py`
  - Adds `MNIST` loader and a writable dataset directory fallback.
- `Inference_pytorch/utee/hook.py`
  - Exports per-layer stream metadata and passes async arguments to the C++ backend.
- `Inference_pytorch/NeuroSIM/Param.h`
- `Inference_pytorch/NeuroSIM/Param.cpp`
  - Add async stream configuration fields.
- `Inference_pytorch/NeuroSIM/main.cpp`
  - Reads stream metadata and applies module-level hardware scaling.

## Current Model Scope

The prototype is intentionally limited:

- Supported async model: `StreamCNN`
- Supported async dataset: `mnist`
- Supported mode: `ASYNC`

This is a first-order behavioral extension. It is not a transistor-accurate asynchronous circuit simulator.

## Supported Stream Parameters

The following options are available in `inference.py`:

- `--stream_frequency`
  - Base stream frequency in Hz.
- `--stream_window`
  - Observation / aggregation window.
- `--stream_jitter`
  - Stream jitter factor used in effective precision and noise estimation.
- `--stream_energy_per_pulse`
  - Relative pulse energy scaling factor.

## Example Workflow

### 1. Train the simple CNN

```bash
cd Inference_pytorch
python train_streamcnn.py --epochs 10 --batch_size 128 --lr 1e-3
```

### 2. Run async inference and hardware estimation

```bash
python inference.py --dataset mnist --model StreamCNN --mode ASYNC --inference 1 --stream_frequency 1e4 --stream_window 32 --stream_jitter 0.02
```

### 3. Inspect outputs

Useful outputs include:

- software inference accuracy
- per-layer stream statistics
- per-layer latency / energy breakdown
- chip-level summary
- energy efficiency and throughput

## How To Train StreamCNN

To obtain meaningful classification accuracy, first train the simple `MNIST` model:

```bash
cd Inference_pytorch
python train_streamcnn.py --epochs 10 --batch_size 128 --lr 1e-3
```

By default, the training script saves the best checkpoint to:

```bash
Inference_pytorch/log/StreamCNN_mnist.pth
```

The inference path already looks for this checkpoint automatically when `--model StreamCNN` is selected.

## Reference Result

The following reference result was obtained after training `StreamCNN` on `MNIST` and running:

```bash
python inference.py --dataset mnist --model StreamCNN --mode ASYNC --inference 1 --stream_frequency 1e4 --stream_window 32 --stream_jitter 0.02
```

### Software accuracy

- Test accuracy: `9852 / 10000 (99%)`
- Test loss: `0.0472`

### Chip-level hardware summary

- `Chip clock period`: `197.764 ns`
- `Chip pipeline-system-clock-cycle (per image)`: `1.80541e+06 ns`
- `Chip pipeline-system readDynamicEnergy (per image)`: `113808 pJ`
- `Chip pipeline-system leakage Energy (per image)`: `25087.4 pJ`
- `Chip pipeline-system leakage Power`: `13.8957 uW`
- `Energy Efficiency`: `3.27379 TOPS/W`
- `Throughput`: `553.892 FPS`

### Example per-layer async stream behavior

| Layer | Event Rate | Frequency (Hz) | ENOB (bit) | ADC Lat/Energy Scale | Accum Lat/Energy Scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| Layer 1 | 0.964037 | 9820.18 | 12.0000 | 1.55038 / 1.04567 | 1.27662 / 1.17411 |
| Layer 2 | 0.163358 | 5948.64 | 11.1036 | 2.46382 / 0.63990 | 0.73011 / 0.57530 |
| Layer 3 | 0.411838 | 7059.19 | 12.0000 | 2.15676 / 0.80671 | 0.91769 / 0.77652 |
| Layer 4 | 0.381719 | 6908.59 | 12.0000 | 2.20378 / 0.78749 | 0.89812 / 0.75484 |

### Interpretation of the reference result

- The first layer has very high event activity because it directly observes the input image.
- Later layers exhibit lower effective frequency and lower event density.
- ADC remains the dominant energy consumer in the current prototype.
- The prototype already captures a useful async trade-off:
  - lower event activity can reduce dynamic energy
  - lower event activity can also increase the time needed to accumulate enough evidence

These values should be treated as architecture-exploration results rather than final circuit signoff numbers.

## Output Artifacts

The following files are generated under `Inference_pytorch/layer_record_StreamCNN/`:

- `weight*.csv`
- `input*.csv`
- `meta*.txt`
- `trace_command.sh`

The metadata files contain stream-aware scaling factors consumed by `NeuroSIM/main`.

## How The Circuit Performance Change Is Modeled

The current version models circuit performance changes at the module level:

- `adc_latency_scale`, `adc_energy_scale`
- `accum_latency_scale`, `accum_energy_scale`
- `buffer_latency_scale`, `buffer_energy_scale`
- `ic_latency_scale`, `ic_energy_scale`
- `leakage_scale`

These factors are derived from stream statistics and applied in `NeuroSIM/main.cpp` after the original NeuroSim layer performance is computed.

The current implementation follows this pattern:

1. compute stream statistics in Python
2. export per-layer metadata
3. run the original NeuroSim hardware estimation
4. rescale module-level latency and energy terms in C++

This keeps the implementation compact while still exposing meaningful circuit-level trends.

## How To Read The Async Log

When async inference is enabled, each layer prints both the original NeuroSim hardware breakdown and the stream-aware scaling summary.

### Key stream statistics

- `Async Stream Event Rate`
  - Fraction of activations treated as meaningful events.
- `Async Stream Duty Cycle`
  - Average duty cycle of the encoded stream.
- `Async Stream Frequency`
  - Effective stream frequency after activity-aware scaling.
- `Async Stream ENOB`
  - Effective precision estimate derived from pulse budget and jitter.
- `Async Stream Pulse Budget`
  - Approximate number of useful stream pulses available for aggregation.

### Key circuit scaling terms

- `ADC latency/energy scale`
  - Captures how stream behavior changes sensing / readout cost.
- `Accum latency/energy scale`
  - Captures how accumulation circuits scale with stream activity.
- `Buffer latency/energy scale`
  - Captures changes in local storage / buffering overhead.
- `IC latency/energy scale`
  - Captures changes in interconnect activity and transfer cost.
- `Leakage scale`
  - Captures the impact of stream timing on leakage-related cost.

## How To Interpret A Successful Run

A good async run usually means:

1. The model accuracy is reasonable
2. Per-layer stream statistics are printed
3. Per-layer hardware breakdown is printed
4. Chip-level summary is produced without crashing

For example, after training `StreamCNN`, a run with ~99% `MNIST` accuracy indicates:

- the model weights are valid
- the async-stream path is active
- the reported latency / energy changes are meaningful

## Typical Interpretation Of The Results

In the current prototype, the first layer often has:

- very high event rate
- frequency close to the base stream frequency
- larger ADC and accumulation cost

This is expected because the first layer directly processes raw image input.

Later layers often show:

- lower event rate
- lower effective frequency
- lower energy in buffer / interconnect
- sometimes higher effective latency because stream aggregation becomes slower

This is also expected and matches the intended async trade-off:

- lower activity can reduce dynamic energy
- but lower activity can also increase time required to accumulate enough stream evidence

## What To Focus On In The Summary

The most useful final metrics are:

- `Chip pipeline-system-clock-cycle (per image)`
- `Chip pipeline-system readDynamicEnergy (per image)`
- `Chip pipeline-system leakage Energy (per image)`
- `Energy Efficiency TOPS/W`
- `Throughput FPS`

The most useful per-layer metrics are:

- `ADC readLatency / readDynamicEnergy`
- `Accumulation readLatency / readDynamicEnergy`
- `Other Peripheries readLatency / readDynamicEnergy`
- all `Async Stream ... scale` terms

## Current Practical Observation

With a trained `StreamCNN`, the current prototype already supports:

- meaningful classification accuracy
- stable async-stream hardware evaluation
- per-layer circuit trend analysis
- chip-level latency / energy / throughput reporting

At this stage, the prototype is suitable for relative studies such as:

- frequency sweeps
- observation window sweeps
- jitter sweeps
- energy vs. latency trade-off studies

## Suggested Experiment Directions

Useful next experiments include:

- sweep `--stream_frequency`
- sweep `--stream_window`
- sweep `--stream_jitter`
- compare trained vs. untrained StreamCNN
- compare per-layer ADC dominance across settings

## Important Limitations

- `StreamCNN` is currently untrained by default, so accuracy is low unless trained weights are provided.
- The async model changes hardware estimation, not only algorithm behavior.
- The C++ backend still uses the original NeuroSim physical model as the base and applies stream-aware scaling on top.
- This prototype is suitable for rapid architecture exploration, not final silicon signoff.
- During training, async layers fall back to standard differentiable CNN execution. Stream-aware effects are applied during inference-time hardware evaluation.

## Current Limitations

The current prototype intentionally keeps the implementation simple:

- only `StreamCNN` is supported in `ASYNC` mode
- the stream model is first-order and behavioral
- the backend still starts from the original NeuroSim physical model
- module-level async scaling is applied after base hardware estimation
- absolute numbers are useful, but relative comparisons are more reliable at this stage

## Suggested Next Steps

- Train `StreamCNN` on `MNIST` and save `log/StreamCNN_mnist.pth`
- Extend async layers to `VGG8`
- Refine stream-to-circuit scaling formulas using a more detailed ASC / sigma-delta model
