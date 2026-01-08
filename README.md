# Generative AI for Electrochemical Process Calculations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper: **"Generative AI Driven Process Calculations for Fuel Cells and Flow Batteries"**

## 🎯 Key Results

Our LLM-RAG-Physics framework achieves:
- ✅ **99% reduction** in constraint violations (45% → 0.5%)
- ✅ **97% reduction** in compile errors (30% → 0.8%)
- ✅ **55% human effort reduction**
- ✅ **Maintained accuracy**: RMSE = 9.6 mV, R² = 0.9919

Progressive improvements across 6 configurations validate all paper claims.

## 📊 Results Summary

### Figure 9: Ablation Study
![Ablation Study Results](results/figures/figure9_ablation_study.png)

### Table II: Performance Metrics

| Configuration | Violations | Compile Errors | Human Effort Reduction |
|--------------|-----------|----------------|----------------------|
| Base         | 45.0%     | 30.0%          | 0%                   |
| +RAG         | 35.0%     | 20.0%          | 15%                  |
| +Hybrid      | 25.0%     | 15.0%          | 20%                  |
| +Physics     | 5.0%      | 8.0%           | 30%                  |
| +Tools       | 2.0%      | 3.0%           | 45%                  |
| **Full**     | **0.5%**  | **0.8%**       | **55%**              |

## 🏗️ Architecture

```
User Task (Natural Language)
    ↓
[Planner] → Decomposes task into subtasks
    ↓
[Retriever] → RAG query (Equation 3 hybrid similarity)
    ↓
[Modeler] → Synthesizes equations with physics constraints
    ↓
[Code Generator] → Emits Python/MATLAB solver code
    ↓
[Physics Validator] → Validates constraints & stability
    ↓
[Executor] → Runs solver (PEMFC fitter/VRFB optimizer)
    ↓
[Critic] → Validates results, triggers refinement
    ↓
Results + Audit Trail
```

## 🔄 Reproducibility
To facilitate external validation and community implementation, this repository provides:

- **Prompts**: Exact system and user prompts used for the LLM experiments are in [`src/llm_orchestrator/prompts`](src/llm_orchestrator/prompts).
- **Physics Constraints**: The `PhysicsValidator` logic and regex patterns are available in [`src/physics_validator`](src/physics_validator).
- **Data**: Synthetic PEMFC and VRFB datasets used for extensive benchmarking are in [`data/synthetic`](data/synthetic).

**Note on Proprietary Assets**:
- **Full-Text PDFs**: Due to copyright restrictions, the specific third-party journal articles used to populate the RAG database are not included. Users must supply their own PDF libraries.
- **Aspen Plus Files**: The source `.bkp` simulation files are not provided due to licensing; however, the generated high-fidelity CSV datasets are fully available for benchmarking.

## 📘 User Guide

### 1. Environment Setup

We recommend using **Miniconda** or **Anaconda** to manage dependencies and avoid conflicts.

#### Option A: Using Conda (Recommended)
```bash
# 1. Create a new environment with Python 3.10
conda create -n genai-electro python=3.10 -y

# 2. Activate the environment
conda activate genai-electro

# 3. Clone and install
git clone https://github.com/Rishi-source/genai-electrochemical-modeling.git
cd genai-electrochemical-modeling
pip install -r requirements.txt
```

#### Option B: Using Standard Venv
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

The framework requires access to an LLM provider (Azure OpenAI or OpenAI) to function.

1.  **Copy the template**:
    ```bash
    cp config/azure_config.template.py config/azure_config.py
    ```

2.  **Edit `config/azure_config.py`**:
    Open the file and fill in your credentials.
    ```python
    # Example config/azure_config.py
    API_TYPE = "azure"  # or "openai"
    API_KEY = "sk-..."
    API_BASE = "https://your-resource.openai.azure.com/"
    API_VERSION = "2023-05-15"
    DEPLOYMENT_NAME = "gpt-4-turbo"
    ```

### 3. Execution Scenarios

#### Scenario A: Full Replication (Paper Results)
To reproduce all figures and tables from the manuscript, run the pipeline in order:

1.  **Generate Synthetic Data**:
    ```bash
    # Creates data/synthetic/pemfc_polarization.csv (N=150 curves)
    python src/data_generation/pemfc_simulator.py
    
    # Creates data/synthetic/vrfb_cycles.csv (N=500 cycles)
    python src/data_generation/vrfb_simulator.py
    ```

2.  **Train/Run Baselines**:
    ```bash
    # Train ANN baseline (saves model to models/ann_pemfc.pth)
    python src/baselines/ann_pemfc.py
    
    # Run DRL agent training (saves results to results/baselines/drl)
    python src/baselines/drl_vrfb.py
    ```

3.  **Run LLM Orchestrator (Ablation Study)**:
    *Note: This step makes real API calls and may incur costs.*
    ```bash
    # Runs all 6 configurations on the test set
    python src/llm_orchestrator/ablation_runner.py
    ```

4.  **Generate Plots**:
    ```bash
    # Generates Figure 8 (Comparison) and Figure 9 (Ablation)
    python scripts/generate_figure8_comparison.py
    python scripts/generate_figure9_ablation.py
    ```

#### Scenario B: Quick Test (Single Curve Fitting)
To test the framework without running the full benchmark:

```bash
# Run the PEMFC fitter on a single sample curve
python src/solvers/pemfc_fitter_demo.py --sample_id 42

# Expected Output:
# - Retrieved equations from RAG
# - Generated Python/MATLAB code
# - Fitted parameters (i0, alpha, R) vs Ground Truth
# - Plot saved to results/demo_fit_42.png
```

### 4. Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `AuthenticationError: 401` | Invalid API Key | Check `config/azure_config.py` against your provider dashboard. |
| `ModuleNotFoundError` | Missing package | Run `pip install -r requirements.txt` again. |
| `ConstraintViolation` | LLM hallucination | This is expected in the "Base" ablation. Use `+Physics` config. |
| `RateLimitError` | API quota exceeded | Add `time.sleep` in `src/llm_orchestrator/client.py` or request quota increase. |

## 📁 Project Structure

```
genai-electrochemical-modeling/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── PHASE3_SUMMARY.md           # Detailed results report
│
├── src/                         # Source code
│   ├── data_generation/         # PEMFC & VRFB simulators
│   ├── baselines/               # ANN, ePCDNN, DRL, Mechanistic
│   ├── solvers/                 # PEMFC fitter, VRFB optimizer
│   ├── rag/                     # ChromaDB RAG system
│   ├── llm_orchestrator/        # LLM + Ablation runner
│   └── physics_validator/       # Constraint validator
│
├── config/                      # Configuration
│   ├── physics_constants.py     # Physical constants
│   └── azure_config.template.py # API config template
│
├── scripts/                     # Figure/table generation
│   ├── generate_figure4_pemfc_decomposition.py
│   ├── generate_figure5_vrfb_pareto_simple.py
│   ├── generate_figure8_comparison.py
│   ├── generate_figure9_ablation.py
│   ├── generate_table1_comparison.py
│   └── generate_table2_ablation.py
│
├── results/                     # Generated outputs
│   ├── figures/                 # All 9 figures (PNG)
│   └── tables/                  # All tables (CSV + LaTeX + JSON)
│
└── data/                        # Synthetic data
    └── synthetic/
        ├── pemfc_polarization.csv
        ├── vrfb_cycles.csv
        └── *.json metadata
```

## 📈 Implementation Scale

- **~3,500 lines** of production Python code
- **4 baseline methods** implemented (ANN, ePCDNN, DRL, Mechanistic)
- **6 configuration ablation** study (18 experiments total)
- **9 publication-ready figures** generated
- **2 comprehensive comparison tables** (CSV + LaTeX formats)

## 🔬 Key Features

### 1. Data Generation
- Synthetic PEMFC polarization curves (900 points across operating conditions)
- Synthetic VRFB charge-discharge cycles (500 cycles)
- Physics-based simulation with realistic noise

### 2. Baseline Implementations
- **ANN**: Feedforward neural network (R² = 0.9757, RMSE = 16.62 mV)
- **ePCDNN**: Physics-constrained deep neural network for VRFB
- **DRL**: Dueling DQN for parameter optimization (SOC RMSE = 0.0544)
- **Mechanistic**: Ground-truth 0D solver

### 3. LLM Orchestration Framework
- Base LLM client with Azure OpenAI (GPT-4/5)
- Physics validator (syntax, dimensional, constraints)
- Ablation runner with 6 configurations
- RAG infrastructure with ChromaDB

### 4. Retrieval-Augmented Generation (RAG)
- Hybrid similarity scoring (Equation 3 from paper)
- Cosine + Mahalanobis + Physics penalties
- ChromaDB vector database
- Domain-adapted embeddings

### 5. Physics-Constrained Validation
- Syntactic checks (AST parsing)
- Dimensional consistency
- Physics constraints (i < i_L, η ≥ 0, etc.)
- Numerical stability analysis

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{garg2025generative,
  title={Generative AI Driven Process Calculations for Fuel Cells and Flow Batteries},
  author={Garg, Rishi and Manjhi, Vasudev and Elhence, Anubhav and Chamola, Vinay and Pandey, Jay},
  journal={[Journal Name]},
  year={2025}
}
```

## 📊 Comparison with Literature

### vs. ANN PEMFC [[2]](https://doi.org/10.3390/pr13051453)
- **ANN**: R² = 0.907, RMSE = 207 mV
- **Our framework**: R² = 0.9919, RMSE = 9.6 mV
- **Improvement**: 21× better RMSE

### vs. ePCDNN VRFB [[1]](https://www.sciencedirect.com/science/article/pii/S0378775322007960)
- **ePCDNN**: ~30% error reduction vs. PCDNN
- **Our framework**: 99% violation reduction
- **Impact**: Physics constraints significantly more effective

### vs. DRL VRFB [[3]](https://doi.org/10.3390/batteries10010008)
- **DRL**: SOC RMSE ≈ 0.46%, high training cost
- **Our framework**: Interpretable, low latency, 55% effort reduction
- **Trade-off**: Transparency + efficiency vs. pure optimization

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/

# Type checking
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Paper authors: Rishi Garg, Vasudev Manjhi, Anubhav Elhence, Vinay Chamola, Jay Pandey
- Azure OpenAI for LLM infrastructure
- ChromaDB for vector database
- PyTorch for deep learning baselines

## 📧 Contact

For questions or collaborations:
- **Email**: [f20221683@pilani.bits-pilani.ac.in]
- **GitHub Issues**: [Create an issue](https://github.com/rishi-source/genai-electrochemical-modeling/issues)

## 🔗 Links

- **Paper**: [Link to paper when published]
- **Results**: All figures and tables in `results/` directory

---

**⭐ If you find this work useful, please consider starring the repository!**
