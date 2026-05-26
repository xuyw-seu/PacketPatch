# uer/ — Universal Encoder Representations Framework

This directory contains the UER (Universal Encoder Representations) training framework, which provides the underlying infrastructure for model building, training, and inference. PacketPatch uses this framework as its backbone, extending it with custom training targets for adversarial packet generation.

## Directory Structure

```
uer/
├── README.md                          # This file
│
├── encoders/                          # Sequence encoders
│   ├── __init__.py                    # Encoder registry (str2encoder)
│   ├── transformer_encoder.py         # Transformer encoder (used by PacketPatch)
│   ├── rnn_encoder.py                 # RNN/LSTM/GRU/BiRNN encoders
│   └── cnn_encoder.py                 # Gated CNN encoder
│
├── layers/                            # Neural network layers
│   ├── __init__.py                    # Layer registry (str2embedding)
│   ├── embeddings.py                  # Word/Position/Segment embeddings
│   ├── transformer.py                 # Single Transformer layer
│   ├── multi_headed_attn.py           # Multi-head self-attention
│   ├── position_ffn.py               # Position-wise feed-forward network
│   ├── layer_norm.py                  # Layer normalization
│   ├── relative_position_embedding.py # Relative position embeddings
│   └── synthesizer.py                 # Synthesizer attention variant
│
├── targets/                           # Training target heads
│   ├── __init__.py                    # Target registry (str2target)
│   ├── bert_target.py                 # BERT target (MLM + NSP)
│   ├── mlm_target.py                  # Masked Language Modeling
│   ├── cls_target.py                  # Classification target
│   ├── lm_target.py                   # Language Modeling
│   ├── bilm_target.py                 # Bidirectional LM
│   ├── albert_target.py               # ALBERT target (MLM + SOP)
│   ├── seq2seq_target.py              # Sequence-to-sequence
│   ├── t5_target.py                   # T5 target
│   ├── nsp_target.py                  # Next Sentence Prediction
│   ├── prefixlm_target.py             # Prefix LM
│   ├── packet_distance_target.py      # Packet distance prediction
│   └── packet_reording_target.py      # Packet reordering prediction
│
├── models/                            # Model definition
│   └── model.py                       # Generic Model class (embedding + encoder + target)
│
├── utils/                             # Utility modules
│   ├── __init__.py                    # Utility registry (tokenizers, datasets, optimizers)
│   ├── tokenizers.py                  # Tokenizer implementations (BERT, Char, Space)
│   ├── vocab.py                       # Vocabulary class
│   ├── data.py                        # Dataset and DataLoader classes
│   ├── optimizers.py                  # Optimizer and scheduler utilities
│   ├── config.py                      # Hyperparameter loading from JSON
│   ├── constants.py                   # Special token constants
│   ├── act_fun.py                     # Activation functions
│   ├── misc.py                        # Miscellaneous utilities
│   ├── seed.py                        # Random seed setting
│   └── subword.py                     # Subword tokenization utilities
│
├── model_builder.py                   # Dynamic model construction
├── model_loader.py                    # Pretrained weight loading
├── model_saver.py                     # Model checkpoint saving
├── trainer.py                         # Training loop (single-GPU, multi-GPU, distributed)
└── opts.py                            # CLI argument definitions
```

## Key Design

The UER framework follows a modular design pattern:

```python
# Building blocks
embedding = str2embedding[args.embedding](args, vocab_size)  # Word/Position/Segment
encoder   = str2encoder[args.encoder](args)                  # Transformer/RNN/CNN
target    = str2target[args.target](args, vocab_size)        # MLM/NSP/CLS/...

# Assemble model
model = Model(args, embedding, encoder, target)
```

Each component is registered in a dictionary (`str2embedding`, `str2encoder`, `str2target`) that maps string names to class constructors, enabling dynamic model building from configuration.

## PacketPatch-Specific Modifications

PacketPatch adds the following custom targets to the UER framework:

| File | Description |
|------|-------------|
| `targets/packet_distance_target.py` | Packet distance prediction (NSP variant for packets) |
| `targets/packet_reording_target.py` | Packet reordering prediction |

The training logic for these targets is implemented in `packet_adv/adv_fine_tuning_cls.py` rather than using the standard UER trainer, as PacketPatch requires custom data loading and training loop logic.

## Configuration

The `bert_base_config.json` at the repository root defines the BERT architecture:

```json
{
    "emb_size": 768,
    "feedforward_size": 3072,
    "hidden_size": 768,
    "hidden_act": "gelu",
    "heads_num": 12,
    "layers_num": 12,
    "dropout": 0.1
}
```

> **Note**: PacketPatch specifically uses `word_pos_seg` embedding (token + position + segment) and `transformer` encoder with `fully_visible` attention mask.

## References

- UER framework: https://github.com/dbiir/UER-py
- ET-BERT (pretrained model source): https://github.com/linwhitehat/ET-BERT
