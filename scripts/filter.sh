# calclate paraphrase score
mkdir -p data/sim_scores
python scripts/calc_sim.py data/orig/ref data/pseudo_references data/sim_scores $TUNED_MODEL_DIR

# filter pseudo-reference
mkdir -p data/filtered_pseudo_references
python scripts/filter.py data/sim_scores data/filtered_pseudo_references --theta 0.5
