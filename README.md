# Egocentric Vision - Ego4D NLQ Benchmark

This repository contains implementations and experiments for the **Ego4D Natural Language Queries (NLQ)** benchmark, including baseline models and extension approaches for episodic memory tasks in egocentric video understanding.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Available Notebooks](#available-notebooks)
- [Dataset](#dataset)
- [Models](#models)
- [Extension Approaches](#extension-approaches)
- [Citation](#citation)

## 🎯 Overview

This project implements and extends baseline models for the Ego4D Natural Language Queries benchmark. The task involves localizing temporal windows in egocentric videos based on natural language queries (e.g., "Where did I put my keys?").

**Key Features:**
- Multiple baseline implementations (VSLNet, VSLBase, 2D-TAN)
- Custom extension approaches with narration-based query generation
- Comprehensive data analysis and visualization
- Integration with Ego4D dataset and Omnivore features

## 📁 Project Structure

```
.
├── AML_1_Ego4D_NLQ_Benchmark.ipynb                    # Base VSLNet benchmark
├── AML_1_Ego4D_NLQ_Benchmark_2DTAN.ipynb              # 2D-TAN baseline
├── AML_1_Ego4D_NLQ_Benchmark_VSLBASE.ipynb            # VSLBase baseline
├── AML_1_Ego4D_NLQ_Benchmark_extension_approach_1.ipynb  # Extension method 1
├── AML_1_Ego4D_NLQ_Benchmark_extension_approach_2.ipynb  # Extension method 2
├── episodic-memory/                                   # Cloned baseline repos
│   └── NLQ/
│       ├── VSLNet/                                    # VSLNet implementation
│       ├── VSLBASE/                                   # VSLBase implementation
│       └── 2D-TAN/                                    # 2D-TAN implementation
├── extension_1/                                       # Custom extension scripts
│   ├── 1_extract_usable_narrations.py
│   ├── 2_extraxt_narration_windows.py
│   ├── 3_from_questions_to_narration_format.py
│   ├── 4_from_narration_format_to_nlq_format.py
│   ├── 5_split_annotations.py
│   └── query_generation/                              # Query generation pipeline
└── datasets/                                          # Dataset directory (created during setup)
    └── ego4d_data/
        └── v1/
            ├── annotations/
            └── omnivore_video_swinl_fp16/
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- AWS Account with Ego4D access

### 1. Install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### 2. Configure AWS Credentials

```bash
aws configure set aws_access_key_id YOUR_ACCESS_KEY_ID
aws configure set aws_secret_access_key YOUR_SECRET_ACCESS_KEY
```

**Note:** Get your credentials by signing the Ego4D License at [ego4ddataset.com](https://ego4ddataset.com)

### 3. Install Ego4D CLI

```bash
pip install ego4d
```

### 4. Download Dataset

Download annotations and Omnivore features:

```bash
ego4d --output_directory="datasets/ego4d_data/" \
      --datasets annotations omnivore_video_swinl_fp16 \
      --benchmarks nlq \
      --version v1 \
      -y
```

### 5. Clone Episodic Memory Repository

```bash
git clone https://github.com/EGO4D/episodic-memory
cd episodic-memory
git checkout nlq_fixes_and_fp16_support
```

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

Additional packages:
```bash
pip install torch torchvision
pip install transformers
pip install matplotlib wordcloud
```

## 📓 Available Notebooks

### Baseline Models

1. **[AML_1_Ego4D_NLQ_Benchmark.ipynb](AML_1_Ego4D_NLQ_Benchmark.ipynb)**
   - VSLNet baseline implementation
   - Training and evaluation pipeline
   - Data statistics and visualization

2. **[AML_1_Ego4D_NLQ_Benchmark_VSLBASE.ipynb](AML_1_Ego4D_NLQ_Benchmark_VSLBASE.ipynb)**
   - VSLBase variant with enhanced features
   - Compatible with EgoVLP features

3. **[AML_1_Ego4D_NLQ_Benchmark_2DTAN.ipynb](AML_1_Ego4D_NLQ_Benchmark_2DTAN.ipynb)**
   - 2D Temporal Adjacent Networks
   - Alternative architecture for temporal localization

### Extension Approaches

4. **[AML_1_Ego4D_NLQ_Benchmark_extension_approach_1.ipynb](AML_1_Ego4D_NLQ_Benchmark_extension_approach_1.ipynb)**
   - Narration-based query generation (Method 1)
   - Uses video narrations to create synthetic queries

5. **[AML_1_Ego4D_NLQ_Benchmark_extension_approach_2.ipynb](AML_1_Ego4D_NLQ_Benchmark_extension_approach_2.ipynb)**
   - Narration-based query generation (Method 2)
   - Alternative approach with improved query diversity

## 📊 Dataset

The project uses the **Ego4D v1** dataset:

- **Training annotations**: [`nlq_train.json`](datasets/ego4d_data/v1/annotations/nlq_train.json)
- **Validation annotations**: [`nlq_val.json`](datasets/ego4d_data/v1/annotations/nlq_val.json)
- **Test annotations**: [`nlq_test_unannotated.json`](datasets/ego4d_data/v1/annotations/nlq_test_unannotated.json)
- **Features**: Omnivore Video SwinL FP16 embeddings

### Dataset Statistics

Run the data analysis cells in any notebook to see:
- Query distribution
- Temporal window statistics
- Video clip duration analysis
- Query word frequency

## 🤖 Models

### VSLNet (Span-based Localizing Network)

Based on [VSLNet (ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.585.pdf)

**Key Features:**
- Span-based prediction
- Cross-modal interaction
- Efficient temporal localization

**Training:**
```bash
cd episodic-memory/NLQ/VSLNet
python main.py --task nlq_official_v1 --predictor bert --mode train
```

### 2D-TAN (2D Temporal Adjacent Networks)

Based on [2D-TAN (AAAI 2020)](https://arxiv.org/abs/1912.03590)

**Key Features:**
- 2D temporal feature maps
- Multi-scale moment detection
- Learnable query-guided attention

**Training:**
```bash
cd episodic-memory/NLQ/2D-TAN
python moment_localization/train.py \
    --cfg experiments/ego4d/2D-TAN-40x40-K9L4-pool-window-std-omnivore.yaml
```

## 🔬 Extension Approaches

### Extension Pipeline

The [`extension_1/`](extension_1/) directory contains scripts for generating synthetic queries from video narrations:

1. **Extract Usable Narrations**: [`1_extract_usable_narrations.py`](extension_1/1_extract_usable_narrations.py)
2. **Extract Narration Windows**: [`2_extraxt_narration_windows.py`](extension_1/2_extraxt_narration_windows.py)
3. **Convert to Question Format**: [`3_from_questions_to_narration_format.py`](extension_1/3_from_questions_to_narration_format.py)
4. **Convert to NLQ Format**: [`4_from_narration_format_to_nlq_format.py`](extension_1/4_from_narration_format_to_nlq_format.py)
5. **Split Annotations**: [`5_split_annotations.py`](extension_1/5_split_annotations.py)

### Query Generation

Located in [`extension_1/query_generation/`](extension_1/query_generation/):
- Reconstruct chunks: [`4_reconstruct_chunks.py`](extension_1/query_generation/4_reconstruct_chunks.py)
- Merge JSON outputs for large-scale processing

## 🎓 Citation

If you use this code or the Ego4D dataset, please cite:

```bibtex
@article{Ego4D2021,
  title={Ego4D: Around the World in 3,000 Hours of Egocentric Video},
  author={Grauman, Kristen and others},
  journal={arXiv preprint arXiv:2110.07058},
  year={2021}
}

@inproceedings{Zhang2020VSLNet,
  title={Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language},
  author={Zhang, Haonan and Sun, Aixin and Jing, Wei and Zhou, Joey Tianyi},
  booktitle={ACL},
  year={2020}
}
```

## 📝 Resources

- **Ego4D Website**: [https://ego4d-data.org](https://ego4d-data.org)
- **Ego4D Documentation**: [https://ego4d-data.org/docs](https://ego4d-data.org/docs)
- **NLQ Benchmark**: [https://ego4d-data.org/docs/benchmarks/episodic-memory/](https://ego4d-data.org/docs/benchmarks/episodic-memory/)
- **EvalAI Challenge**: [https://eval.ai/web/challenges/challenge-page/1629/overview](https://eval.ai/web/challenges/challenge-page/1629/overview)
- **Baseline Repository**: [https://github.com/EGO4D/episodic-memory](https://github.com/EGO4D/episodic-memory)

## 📜 License

This project is released under the MIT License. See the Ego4D dataset license for dataset usage terms.

## 🤝 Contributing

Contributions are welcome! Please follow the contribution guidelines in [episodic-memory/NLQ/2D-TAN/CONTRIBUTING.md](episodic-memory/NLQ/2D-TAN/CONTRIBUTING.md).

## 🔒 Security

For security issues, please refer to [episodic-memory/NLQ/2D-TAN/SECURITY.md](episodic-memory/NLQ/2D-TAN/SECURITY.md).