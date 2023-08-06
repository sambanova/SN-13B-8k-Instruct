# SN-13B-8k-Instruct
This repo contains the reproducibility information for the numbers listed in the SN-13B-8k-Instruct blogpost.

## Setup Eleuther AI LM Evaluation Harness
1. git clone https://github.com/EleutherAI/lm-evaluation-harness.git
2. Checkout the commit of LM Eval Harness that we used to collect the results: `git checkout fe803c2920a85f6afb74ea05d1d2f98ec27f1a63`
3. Follow the setup instructions specified in the repository's README.

## ZeroScrolls Reproducibility
1. Add ZeroScrolls task code.
2. pip install lifelines
3. pip install unidecode

## Scrolls Reproducibility
1. Replace the `'\n'` with your model's end of text token in the `until` list for all `greedy_until` requests.
