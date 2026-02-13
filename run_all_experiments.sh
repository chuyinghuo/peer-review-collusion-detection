#!/usr/bin/env bash
set -e

cd /Users/chuyinghuo/Downloads/peer-review-collusion-detection

# OPTIONAL: activate your venv if you use one
# source .venv/bin/activate

# 1) unipartite clique, aamas_sub3
python scripts/clique_eval.py aamas_sub3 \
  -t 1440 -kn 2 -kx 13 -gn 0.5 -gx 1.0 -gs 0.1

# 2) unipartite clique, wu
python scripts/clique_eval.py wu \
  -t 1440 -kn 2 -kx 13 -gn 0.5 -gx 1.0 -gs 0.1

# 3) bipartite clique, aamas_sub3
python scripts/clique_eval.py aamas_sub3 \
  -t 1440 -kn 2 -kx 13 -gn 0.5 -gx 1.0 -gs 0.1 -bp

# 4) bipartite clique, wu
python scripts/clique_eval.py wu \
  -t 1440 -kn 2 -kx 13 -gn 0.5 -gx 1.0 -gs 0.1 -bp

# 5) densest_subgraph, aamas_sub3
python scripts/detection_eval.py aamas_sub3 \
  -m densest_subgraph -kn 2 -kx 35 -gn 0.5 -gx 1.0 -gs 0.1

# 6) densest_subgraph, wu
python scripts/detection_eval.py wu \
  -m densest_subgraph -kn 2 -kx 35 -gn 0.5 -gx 1.0 -gs 0.1

# 7) fraudar, aamas_sub3
python scripts/detection_eval.py aamas_sub3 \
  -m fraudar -g B -bp -kn 2 -kx 34 -gn 0.2 -gx 1.0 -gs 0.2

# 8) fraudar, wu
python scripts/detection_eval.py wu \
  -m fraudar -g B -bp -kn 2 -kx 34 -gn 0.2 -gx 1.0 -gs 0.2

# 9–12) success_eval (unipartite + bipartite, both datasets)

# unipartite success (G₁)
python scripts/success_eval.py aamas_sub3 \
  -kn 2 -kx 35 -gn 0.5 -gx 1.0 -gs 0.1 -t 10

python scripts/success_eval.py wu \
  -kn 2 -kx 35 -gn 0.5 -gx 1.0 -gs 0.1 -t 10

# bipartite success (G₂)
python scripts/success_eval.py aamas_sub3 \
  -bp -kn 2 -kx 34 -gn 0.2 -gx 1.0 -gs 0.2 -t 10

python scripts/success_eval.py wu \
  -bp -kn 2 -kx 34 -gn 0.2 -gx 1.0 -gs 0.2 -t 10
