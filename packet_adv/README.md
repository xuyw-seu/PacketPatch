# packet_adv/ — Core Experimental Code

This directory contains the three core Python scripts for the PacketPatch pipeline.

## Files Overview

| File | Purpose | Order |
|------|---------|-------|
| `adv_fine_tuning_cls.py` | Train PatchGenerator (SpanBERT) using SCP + Span-MBM tasks | **Step 1** |
| `generate.py` | Generate adversarial packets using a trained PatchGenerator | **Step 2** |
| `gen_onnx.py` | Convert trained model to ONNX/TensorRT for accelerated inference | **Step 3** (optional) |

---

## Step 1: adv_fine_tuning_cls.py — Train PatchGenerator

### What it does

This script trains the **PatchGenerator** model (referred to as `SpanBERT` in code) using two joint self-supervised tasks:

1. **SCP (Same Category Prediction)**: Determines whether two packet segments belong to the same traffic category. Implemented via the NSP (Next Sentence Prediction) head.
2. **Span-MBM (Span Masked Byte Modeling)**: Reconstructs consecutive spans of masked bytes using bidirectional context. Implemented via the MLM (Masked Language Modeling) head.

**Total loss**: `Loss_NSP + Loss_MLM / 10`

### Key Components

- **`create_span_masks()`** (line 139): Generates contiguous span masks with 80%/10%/10% [MASK]/random/keep replacement strategy
- **`ETBERTDataset`** (line 200): Custom Dataset implementing packet pairing, dynamic partition, and sequence assembly
- **`MlmTargetGeneration`** (line 504): Output layer for Span-MBM — predicts probability distributions over vocabulary
- **`SpanBERT`** (line 754): PyTorch Lightning model combining embedding + encoder + dual task heads

### Before Running

Edit the following in `adv_fine_tuning_cls.py`:

| Line | Parameter | What to change |
|------|-----------|----------------|
| 47-50 | `tr_x_path`, `tr_y_path`, `val_x_path`, `val_y_path` | Dataset file paths |
| 56 | `num_classes` | Number of traffic classes in your dataset |
| 73-75 | `--pretrained_model_path`, `--vocab_path` | ET-BERT pretrained model and vocabulary paths |
| 102 | `--config_path` | Path to `bert_base_config.json` |
| 878 | `instances_num` | Number of samples in your training set |

### Running

```bash
cd packet_adv
python adv_fine_tuning_cls.py
```

### Output

- Model checkpoints (`.ckpt` files) saved in the specified log directory
- Training metrics logged to TensorBoard

---

## Step 2: generate.py — Generate Adversarial Packets

### What it does

Loads a trained PatchGenerator checkpoint and generates adversarial perturbations for test data packets. This implements the complete perturbation generation pipeline described in the paper:

1. Extract packet content around the header-payload boundary (byte 40)
2. Insert [MASK] tokens with dummy header from a different class
3. Single forward pass through PatchGenerator
4. Apply **argmin** selection on Pad1 and **argmax** selection on Pad2
5. Assemble and output adversarial packets

### Generation Strategies (`fun` parameter)

| Value | Description | Paper Reference |
|-------|-------------|-----------------|
| `fun1` | Single segment, all argmin | — |
| `fun1_argmax` | Single segment, all argmax | — |
| `fun2` | Dual segment, all argmax | — |
| `fun3` | Dual segment (50/50 split), Pad1 argmin + Pad2 argmax | Ablation V1 |
| `fun4` | Dual segment (random split), Pad1 argmin + Pad2 argmax | — |
| **`fun5`** | **Dual segment + random dummy header, Pad1 argmin + Pad2 argmax** | **Default** |
| `fun6` | fun5 with NSP-based decision for Pad2 | — |
| `random` | Random vocabulary sampling baseline | — |
| `random_dummy_head` | Random header replacement baseline | Ablation V7 |
| `random_pad` | Random padding baseline | — |

### Before Running

Edit the following in `generate.py`:

| Line | Parameter | What to change |
|------|-----------|----------------|
| 42-44 | `test_x_path`, `test_y_path`, `save_path` | Test set paths and output directory |
| 50 | `fun` | Generation strategy (default: `'fun5'`) |
| 60 | `BWO` | Bandwidth overhead ratio (default: `0.1` = 10%) |
| 68-77 | `--load_model_path` | Path to trained `.ckpt` checkpoint |
| 71 | `--vocab_path` | Vocabulary file path |

### Running

```bash
cd packet_adv
python generate.py
```

### Output

- BERT-format adversarial samples (`.pickle` file) — uncomment lines 480-481
- Deep packet format adversarial samples (`.npy` file, 1500 bytes normalized) — uncomment lines 499-502
- Average processing time per packet printed to console

---

## Step 3 (Optional): gen_onnx.py — Model Acceleration

### What it does

Converts the trained PyTorch model to ONNX format and compiles it into a TensorRT engine for FP16/INT8 accelerated inference. The paper reports **9.626 ms** perturbation generation time using TensorRT FP16.

### Before Running

Edit the following in `gen_onnx.py`:

| Line | Parameter | What to change |
|------|-----------|----------------|
| 132 | Model path | Trained `.ckpt` checkpoint |
| 145 | `onnx_filename` | Output ONNX file path |
| 207 | Engine path | Output TensorRT engine path |
| 21 | `mode` | `'fp16'` or `'int8'` (int8 requires calibration data) |
| 18-19 | `val_x_path`, `val_y_path` | Calibration dataset (INT8 mode only) |

### Running

```bash
cd packet_adv
python gen_onnx.py
```

### Output

- `*.onnx` — ONNX model file
- `*.trt` — TensorRT engine file

---

## Dependencies Between Steps

```
[Prepare Data] → [Step 1: Train] → [Step 2: Generate] → [Evaluate on Target Models]
                                      ↓
                              [Step 3: Accelerate] (replaces Step 2 inference)
```

- Step 1 must be completed before Step 2 (Step 2 loads the model trained in Step 1)
- Step 3 is independent: it loads the same trained model for conversion
- Different datasets (TOR/VPN/USTC) require separate trained models

## Notes

1. All hardcoded file paths must be updated to match your local environment before running
2. The code uses PyTorch Lightning 1.x API; for version 2.x, some parameters may need adjustment
3. Dataset preprocessing code (pcap → pickle) is NOT included in this directory; see [dataset/README.md](../dataset/README.md)
4. The original `readme.txt` suggests: "Copy the packet_adv folder into the ET-BERT project directory for use"
