# Code for Estimating Finite Point Processes with Score Matching

## Reproducing Table 2

To reproduce Table 2, run (SAHP):

    bash stackoverflow_sahp.sh --seeds 1,2,3 --data "stackoverflow" --method mle --with_survival 1

and (THP):

    bash stackoverflow.sh --seeds 1,2,3 --data "stackoverflow" --method mle --with_survival 1

## Reproducing Figures 1 and 2

To reproduce Figures 1 and 2, run:

    python Main.py

Then set the default dataset to `"half-sin-hawkes"`, `"exp-decay"`, or `"conditional_logistic"`, and select different estimators as needed.

Training on the synthetic datasets is straightforward, so results are generally not sensitive to hyperparameters. In most cases, the default settings should produce similar qualitative results.
