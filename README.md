# DAphyloSMC – A delayed‑acceptance sequential Monte Carlo for phylogenetic Bayesian Inference

[//]: # ([![PyPI version]&#40;https://badge.fury.io/py/DASMC.svg&#41;]&#40;https://pypi.org/project/DASMC/&#41;)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Description

In Bayesian phylogenetics, estimating the posterior distribution over tree space is computationally expensive due to repeated likelihood evaluations. **DAphyloSMC** is a Python package that implements a **delayed‑acceptance sequential Monte Carlo (DA‑SMC)** framework to accelerate phylogenetic inference.

The package extracts over 35 topological and branch‑length features from tree proposals (e.g., eSPR, stNNI) and uses a random forest to predict the likelihood change. This surrogate enables a delayed‑acceptance MCMC kernel that pre‑filters costly likelihood evaluations, substantially reducing computational time while maintaining robust posterior estimates.

DAphyloSMC seamlessly integrates this delayed‑acceptance MCMC kernel into an SMC sampler, providing an efficient, off‑the‑shelf tool for Bayesian phylogenetics on large‑scale sequence data.

For a detailed description of the methodology, please refer to our preprint (arXiv: pending).

## Table of Contents

- [Installation](#installation)
- [Supported Platforms](#Supported-Platforms)
- [System Dependencies](#System-Dependencies)
- [Verification](#Verification)
- [Preparing the data](#Preparing-the-data)
- [Usage](#usage)
  - [Step 1: Pilot SMC (feature collection)](#Step-1-Pilot-SMC-feature-collection)
  - [Step 2: Random Forest Training](#Step-2-Random-Forest-Training)
  - [Step 3: Final DA-SMC run](#Step-3-Final-DA-SMC-run)
  - [Common arguments for dasmc-run](#Common-arguments-for-dasmc-run)
  - [Arguments for dasmc-train](#Arguments-for-dasmc-train)
- [Dependencies](#dependencies)
- [Included Modified Packages](#included-modified-packages)
- [License](#license)
- [Citation](#citation-temporary)
- [Contact](#Contact)
## Installation

[//]: # (You can install DASMC directly from PyPI:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install DASMC)
You can install the latest development version from GitHub:
```bash
git clone https://github.com/wentYu/DAphyloSMC.git
cd DAphyloSMC
pip install .
```
After installation, a command-line tool dasmc will be available.

## Supported Platforms
DAphyloSMC relies on the embedded `p4` library, which includes Linux‑specific binary extensions (`.so` files). Therefore, DAphyloSMC **only supports Linux and macOS (x86_64)**.

**Windows is not supported.** Installation on Windows will succeed, but importing the package will fail due to the incompatible binary extension.

## System Dependencies

DASMC requires the following **C libraries** for certain optimization and scientific computing tasks:

- [NLopt](https://nlopt.readthedocs.io/) – for nonlinear optimization
- [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) – for special functions and numerical routines

These libraries must be installed **on your system** (not via pip). Please follow the official `p4` installation guide for your operating system:

👉 [**p4 Installation Instructions**](https://p4.nhm.ac.uk/install.html)

For **Ubuntu/Debian** users, the essential commands are typically:

```bash
sudo apt-get install libgsl-dev libnlopt-dev python3-dev
```
For **macOS** users with Homebrew:
```bash
brew install gsl nlopt
```
For other platforms or advanced setups (., using Conda, or installing without `sudo`), please refer to the `p4` installation page linked above.

After installing these system dependencies, you can use DAphyloSMC normally.

## Verification

To confirm that DAphyloSMC and its embedded `p4` module are properly installed, run the following Python commands:

```python
import DASMC
from DASMC import p4

print("DASMC version:", DASMC.__version__)
print("p4 module location:", p4.__file__)
```
If you see no error messages and the output shows the path to p4 (inside your site-packages or development directory), the installation was successful.

## Preparing the data

DAphyloSMC requires input files in **Nexus format** (extension `.nex`). Place your `.nex` file(s) inside a subdirectory named `data/` under your current working directory. 

If you have cloned the DAphyloSMC GitHub repository, a sample dataset `primates.nex` is already provided in the `data/` directory. To use this sample, simply run DAphyloSMC commands from the repository root. For your own datasets, simply create the `data/` directory and put your `.nex` file(s) there.
## Usage

DAphyloSMC provides two command-line tools: `dasmc-run` (for SMC sampling) and `dasmc-train` (for training the random forest classifier). The typical workflow consists of three consecutive steps.

### Step 1: Pilot SMC (feature collection)

Run a small‑scale pilot SMC to collect features for training the random forest. The features are saved to `./pilot_output/`.

**Example:**
```bash
dasmc-run -f 1 -d primates -m test1 -n 50 -i 200 -g 0 -p 0.02
```
**Key arguments:**

- `-f 1` : Pilot run (feature collection mode).  
- `-d primates` : Dataset name. The input file must be named `primates.nex`.  
- `-m test1` : Identifier for this set of works.  
- `-n 50` : Number of particles (use a small value for pilot).  
- `-i 200` : Number of SMC iterations (use a small value for pilot).  
- `-g 0` : Disable SYM/GTR model (0 = use JC69 or K2P, 1 = use SYM or GTR).  
- `-p 0.02` : Proposal probability for evolution rate parameters. Setting this value >0 enables the K2P model (instead of JC69).

### Step 2: Random Forest Training

Train the random forest classifier using the collected features. The trained model is saved to `./RF_output/`.

**Example:**
```bash
dasmc-train -d primates -m test1
```
**Important:** The `-d` and `-m` arguments must exactly match the ones used in the pilot run.

### Step 3: Final DA-SMC run

Run the full DA-SMC (delayed‑acceptance SMC) using the pre‑trained random forest model. The final results (posterior parameter sets, tree collections, consensus trees, etc.) are written to `./DASMC_output/`.

**Example:**
```bash
dasmc-run -d primates -m test1 -n 500 -g 0 -p 0.02 -s 1 -a 0.999
```
**Additional (optional) arguments:**
- `-s 1` : enable self‑adaptive annealing schedule
- `-a 0.999` : temperature controller α (only relevant if `-s 1`)

   **Recommended range of -a:** 
  `0.99` to `0.999999`. Closer to `1` yields longer, more accurate runs; closer to `0.99` gives shorter runs. Adjust based on your computational budget.


  If you use `-s` 1 (self‑adaptive annealing), the `-i` (number of iterations) parameter is not required and will be ignored – the iteration count is determined automatically. In that case, you can omit `-i` from the command.

  Oppositely, if `-s` is omitted, the program will use a fixed annealing schedule, and `-i` is required.
>**Note**: The model settings (e.g., -g and -p) must be identical between the pilot run and the final DA-SMC run for a given dataset. The dataset name (-d) and the identifier (-m) must also be consistent throughout the whole workflow.

### Common arguments for `dasmc-run`

| Argument                      | Description                                                                       | Default  |
|-------------------------------|-----------------------------------------------------------------------------------|----------|
| `--feature`, `-f`             | Pilot run (1) or formal DA-SMC run (0).                                           | 0        |
| `--dataset`, `-d`             | Exact base name (without `.nex` extension) of the input file.                     | (required) |
| `--mark`, `-m`                | Identifier for this set of works.                                                 | (required) |
| `--turn`, `-t`                | Which turn in the repeated run (Optional, usually used in formal DA-SMC, not pilot run).                                          | 0        |
| `--random_seed`, `-r`         | Random seed (0 = use default seeds).                                              | 0        |
| `--gtr`, `-g`                 | Use model SYM/GTR (1) or JC69/K2P (0).                                            | 0        |
| `--proposal_kappa_prob`, `-p` | Probability of evolution rate parameters proposal (set to 0 if using JC69).       | 0.02     |
| `--proposal_pi_prob`, `-pp`   | Probability of base frequency parameters proposal (set to 0 if **not** using GTR). | 0        |
| `--kappa`, `-k`               | Initial kappa (if **not** using GTR).                                             | 1.0      |
| `--brlen-prob`, `-b`          | Ratio of branch length proposal and topology proposal.                            | 0.5      |
| `--prior-lambda`, `-p`        | Branch length exponential prior lambda.                                           | 10.0     |
| `--etbrPExt`, `-e`            | eSPR extend probability. (Named after `p4`'s eTBR; same meaning in eSPR.)         | 0.6      |
| `--self_adaptive`, `-s`       | Use self‑adaptive SMC.                                                            | False    |
| `--alpha`, `-a`               | Self‑adaptive SMC temperature controller alpha (if self‑adaptive).                | 0.999    |
| `--nParticles`, `-n`          | Number of particles in one iteration.                                             | (int)    |
| `--iterations`, `-i`          | Total iterations (if not self‑adaptive).                                          | (int)    |
| `--K`, `-K`                   | First rejection threshold K.                                                      | (float)  |
| `--delta`, `-del`             | Likelihood tuning bias δ.                                                         | (float)  |
| `--parallel_cores`, `-pc`     | Cores used for parallelization.                                                   | 1        |

For a complete list of all parameters, including those for advanced use cases or under development, please see the inline documentation in the source code file `DASMC/SMC.py`.
### Arguments for `dasmc-train`

The `dasmc-train` command accepts the following arguments:

| Argument                     | Description                                                                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`,`-d`             | Dataset name (must match pilot run).                                                                                                          |
| `--mark`,`-m`                | Identifier for this set of works (must match pilot run).                                                                                      |
| `--advise_Delta`,`-a`        | If set to `1`, the program will also suggest δ (likelihood tuning bias) under different precision for the subsequent DA‑SMC run. Default `0`. |
| `--K`, `-K`                  | First rejection threshold K. Required when `--advise_Delta` is enabled; must match the value used in the pilot run.                           |
| `-nParticles`, `-n` | Number of particles for the δ suggestion. Required when `--advise_Delta` is enabled; must match the pilot run's `-nP`.                        |

> **Note:** If you enable `--advise_Delta`, you must also provide `-K` and `-nP` consistent with the pilot run. The suggested δ will be printed to the console.
## Dependencies

DAphyloSMC requires Python 3.8 or later. Key dependencies include:

| Package | Minimum Version |
|---------|----------------|
| `numpy` | 1.12.0 |
| `scipy` | 1.5.0 |
| `pandas` | 1.0.0 |
| `matplotlib` | 3.0.0 |
| `seaborn` | 0.11.0 |
| `scikit-learn` | 0.24.0 |
| `ete3` | 3.0.0 |
| `Dendropy` | 4.5.0 |
| `joblib` | 1.0.0 |
| `sumt` | 3.8.0 |

## Included Modified Packages

This package includes a **modified version** of the `p4` library (original author: Peter Foster). The original library is distributed under `GPL-2.0`. The modified source code is located in `DASMC/p4/`.

Modifications include:
- Activated the eSPR move in the source code and fixed a bug about it.
- Added the stNNI move.
- Added support for feature extraction of eSPR and stNNI move.

All original copyright notices and license terms are retained. A copy of the original license is included in `DASMC/p4/LICENSE`.

If you need the unmodified version, please visit the original project at https://github.com/pgfoster/p4-phylogenetics.

## License

DAphyloSMC is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either **version 3 of the License**, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

The complete license text is available in the `LICENSE` file in the root directory of this repository.

> **Note:** This package includes a modified version of the `p4` library. For details, see [Included Modified Packages](#included-modified-packages).

## Citation (temporary)
A preprint describing this work is currently under review. Please check back later or contact the authors for the proper citation format.

## Contact
For questions, bug reports, or suggestions, please [open an issue](https://github.com/wentYu/DAphyloSMC/issues) on GitHub.
You can also contact the maintainer via email: [yuwt2024@shanghaitech.edu.cn](mailto:yuwt2024@shanghaitech.edu.cn).
