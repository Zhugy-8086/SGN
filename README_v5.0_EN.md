# SGN-Lite

Sparse Graph Network Lite -- Pure Integer Competitive/Hebbian Learning Recognition System

## Introduction

SGN-Lite is a lightweight recognition engine with **floating-point-free, backpropagation-free core recognition paths**. Its core mechanism is integerized competitive learning + template matching, designed for algorithm verification in embedded MCU environments.

**Core Features**:
- **Core engine fully integerized** (DiscreteCoordinate discrete coordinate system)
- Competitive learning + Hebbian enhancement/weakening
- Dual-graph overlay gate recognition (v4.4)
- Graph-mode hierarchical memory (v5.0)
- Strategy plugin architecture

> **Note**: The input layer (vector rendering, noise generation, contrast stretching) involves external analog signals and retains floating-point operations. The core recognition layer (neuron matching, Hebbian learning, competitive sorting) is strictly integerized.

## Quick Start

```bash
cd engine
python main.py
```

Press Enter to start training. After training, you can run tests, visualization, and noise analysis.

### Command Line Arguments

```bash
python main.py                          # Interactive training
python main.py --batch                  # Batch mode
python main.py --auto 50                # Auto mode (50ms/step)
python main.py --mode compact           # Compact mode
python main.py --mode blackbox          # Blackbox mode
python main.py --config config.json     # Load configuration
python main.py --no-color               # Disable color output
```

## Architecture

```
Core Layer
  sgn_core.py          Core engine (competition/verification/learning/template merging)

Extension Layer
  sgn_hooks.py         Event bus/hook system
  sgn_config.py        Configuration registry/DiscreteCoordinate
  sgn_commands.py      Command registry

Strategy Layer
  sgn_input.py         Input pipeline/noise model/vector rendering
  sgn_layers.py        Layer extraction/edge extraction/block encoding
  sgn_strategies.py    Layer strategy/verification strategy/matching strategy
  sgn_metrics.py       Evaluation metrics (accuracy/confusion/noise robustness)
  sgn_storage.py       Storage backend (JSON/SQLite)
  sgn_graph.py         Graph data structure (v5.0)
  sgn_stack.py         Graph construction and projection (v5.0)
  sgn_merge.py         Cross-graph merging (v5.0)
  sgn_graph_match.py   Graph matching and inference (v5.0)

Application Layer
  main.py              Entry point
  sgn_interactive.py   Interactive menu
  sgn_training.py      Training loop
  sgn_test.py          Inference/batch/confusion/noise tests
  sgn_visual.py        Visualization/statistics/dashboard
  sgn_report.py        Chart export
  sgn_persist.py       Model persistence
  sgn_utils.py         Utility functions/color output/logging
```

## Runtime Modes

| Mode | Description |
|------|-------------|
| full | Full logging, output every step (default) |
| compact | Compact, only checkpoint summaries |
| blackbox | Blackbox, zero output during training |

## Input Sources

| Type | Description |
|------|-------------|
| pattern | Built-in 4x4 characters (0-F) |
| vector | Vector graphics (line/circle/sine/catear/mixed) |
| file | Load from CSV/JSON files |
| 8x8 standard chars | 0-9 + A-Z (STANDARD_CHARS_8x8) |

Vector graphics support grid sizes: 4/8/16/32/64. Set via `--vector-grid` or control panel `[g]`.

## Key Configuration Items

| Configuration | Default Value | Adjustable Range | Description |
|---------------|---------------|------------------|-------------|
| MAX_NEURONS | 256 | 1 ~ 4096 | Maximum neuron count |
| MAX_TEMPLATES | 500 | 1 ~ 10000 | Maximum template library capacity |
| MAX_ITERATIONS | 100000 | 1 ~ 1000000 | Maximum training steps |
| SEED | 42 | 0 ~ 99999 | Random seed |
| TOP_K | 6 | 1 ~ 128 | Competitive Top-K |
| MAX_LOCKOUT | 120 | 1 ~ 1000 | Lockout threshold |
| ENABLE_GATE_MATCHING | False | -- | Gate matching (v4.4) |
| ENABLE_GRAPH_MODE | False | -- | Graph mode (v5.0) |

> **Note**: The "Default Value" column shows system initial values. The "Adjustable Range" column shows allowed minimum/maximum values. Actual steady-state template count is typically around 80, no need to reach the upper limit.

## Graph Mode (v5.0)

Graph mode attaches a hierarchical memory system above neurons, addressing three inherent defects of template recognition:
1. Hierarchy deficiency -> Multi-layer graph structure
2. Spatial structure loss -> Connected domains + position normalization
3. Overfitting tendency -> Consistency filtering + feedback loop

```bash
# Enable graph mode
# Set ENABLE_GRAPH_MODE = True in Control Panel -> Advanced Options
# Or via configuration file
```

Core mechanisms:
- Neuron competition -> Projected as graph nodes
- Multi-view merging -> Consistency filtering (high-level nodes >= 2 projections)
- Feedback iteration -> Error map rescanning
- Layer demotion -> Forgetting mechanism

## Evaluation Metrics

Available tests after training:
- `[t]` Batch test: Recognition rate
- `[c]` Confusion matrix: Per-class accuracy
- `[n]` Noise test: Composite/Gaussian/salt-and-pepper/block occlusion robustness
- `[s]` Statistics: Neuron/template/verification pass rates
- `[g]` Dashboard: Comprehensive status

## Version History

- v4.3: Long-term parameter refactoring, bug fixes, 8x8 standard character library
- v4.4: Dual-graph overlay gate recognition, edge extraction, block encoding
- v5.0: Graph-mode hierarchical memory (sgn_graph/stack/merge/graph_match)
- v5.0-fix: Vector circle distance calculation bug fix, rendering gradient direction fix
- v5.0-vis: Visualization enhancements (10-level color scale/8-level heatmap/template intensity mode/shape annotation)
- v5.0-catear: New cat ear vector graphics (4 orientations x 3 spacings x 3 radii x 3 opening angles x 2 forms = 216 types)
- v5.0-grid: VECTOR_GRID extended to support 64x64

## Current Version Statement (A Priori Machine Status)

> **SGN-Lite v5.0 is currently in an a priori machine state, not a complete prototype.**
>
> **Version Number Note**: The current version number (v5.0) follows the a priori machine iteration rules. Once formally定型 as a prototype, the version number will be reset to zero and released as an independent project. A priori machines and prototypes belong to different project stages and must not be confused.
>
> The essential difference between an a priori machine and a prototype: the maturity of framework vs. infrastructure. Like a fetus vs. a newborn -- strictly speaking both are living entities, but at completely different developmental stages. SGN is currently still in the "fetal stage": the core framework is erected, but key infrastructure remains incomplete.
>
> **Currently Implemented**: Memory-level optimization code (graph-mode hierarchical memory, template merging, competitive learning mechanisms)
>
> **Not Yet Complete**:
> - Time variable is currently only a step counter, not true temporal dimension processing
> - Neuron infrastructure (e.g., dynamic topology, cross-layer signal transmission) not yet extended
> - Deep integration between graph mode and neurons not yet completed
>
> **Platform Positioning Statement**: SGN is currently a PC-based verification framework codebase, **not an MCU deployment version**. Future development includes MCU adaptation, but current code cannot run directly on embedded devices. Downloading this code provides an algorithm verification platform, not deployable product firmware.
>
> **Concept Statement**: SGN is not a traditional SNN (Spiking Neural Network) or DNN (Deep Neural Network). It possesses an independent conceptual system and terminology definitions. Some concepts may appear similar to existing neural network paradigms, but have essential differences. The current priority is memory architecture, not the complete functionality of the neuron itself.
>
> **Usage Recommendation**: SGN can be used as a "memory a priori framework" for experimentation and extension, but should not be deployed directly as a mature neural network solution.

## Dependencies

- Python 3.10+
- Optional: matplotlib (chart export)
- Optional: sqlite3 (SQLite storage backend)

## License

Apache License 2.0
