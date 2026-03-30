# Async Stream NeuroSim Prototype

This document summarizes the lightweight asynchronous stream-based CIM prototype added on top of the original `DNN+NeuroSim V1.4` flow.

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

The implementation keeps the original NeuroSim hardware estimation flow and extends it with per-layer stream statistics.

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

## How To Run

Example:

```bash
cd Inference_pytorch
python inference.py --dataset mnist --model StreamCNN --mode ASYNC --inference 1 --stream_frequency 1e4 --stream_window 32 --stream_jitter 0.02
```

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

## Recommended Workflow

1. Train the simple CNN on `MNIST`
2. Run async inference with `--mode ASYNC`
3. Inspect:
   - test accuracy
   - per-layer stream statistics
   - per-layer circuit scaling
   - chip-level latency / energy / throughput

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

For example, after training `StreamCNN`, a run with ~99% MNIST accuracy indicates:

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

## Suggested Next Steps

- Train `StreamCNN` on `MNIST` and save `log/StreamCNN_mnist.pth`
- Extend async layers to `VGG8`
- Refine stream-to-circuit scaling formulas using a more detailed ASC / sigma-delta model
