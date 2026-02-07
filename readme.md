This repository contains code for analyzing the problem of detecting reviewer-author collusion rings from bidding datasets. For more explanation on the analyses, please refer to the associated paper: [On the Detection of Reviewer-Author Collusion Rings From Paper Bidding](https://arxiv.org/abs/2402.07860).

**Install:** `make install` or `pip install -e .` Run all commands below from the repo root.

Before running any code:
- Make empty directories titled `datasets/` and `results/`.
- Download the files from [here](https://drive.google.com/drive/folders/1Dpol_eSSQ-6-mYfusxae48KQKHOesIi7?usp=sharing) and place them in the `datasets/` directory. The file `aamas_2021.csv` is sourced from [PrefLib](https://www.preflib.org/dataset/00037). The file `wu_tensor_data.pl` is sourced from [(Wu et al., 2021)](https://github.com/facebookresearch/secure-paper-bidding). The other files are constructed by `scripts/construct_authorships.py` and `scripts/synthesize_aamas_text.py`, which have additional data dependencies not included in this repository.
- Run `make compile` (or `cpp/compile_count_cliques_c.sh`) to compile the C++ subroutines.

In all scripts, the argument `aamas_sub3` refers to the AAMAS dataset and the argument `wu` refers to the S2ORC dataset from the writeup. Other arguments specify the setting (unipartite/bipartite), size and density parameters, detection method, etc. Run from repo root, e.g. `python scripts/clique_eval.py aamas_sub3`. The following scripts run the analyses:
- `scripts/clique_eval.py` runs the exact clique-counting analyses.
- `scripts/detection_eval.py` runs the detection algorithm analyses. Code for detection methods TellTail and Fraudar was sourced from [(Hooi et al., 2020)](https://bhooi.github.io/projects/telltail/) and [(Hooi et al., 2016)](https://bhooi.github.io/projects/fraudar/index.html) respectively.
- `scripts/success_eval.py` runs the colluder success analyses.
