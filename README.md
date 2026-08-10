All ten runs received the same starting files, prompt, model configuration, and
one-turn budget.

`F` is mean predictive skill across a ten-scenario bank.

## Starter

The starter median-imputes numeric data, adds missingness indicators,
standardizes numeric columns, one-hot encodes `segment`, and fits one regularized
logistic regression on train plus validation. It is reliable but does not use
validation for architecture selection, recency selection, blending, or
calibration.

- Selection F: `0.012464`
- Final F: `0.031155`
- Reward: `0.152154`
- Validity: 10/10 selection, 10/10 final

## Run 01 — broad shift-aware ensemble

Run 01 used robust median/IQR scaling, missingness features, bounded values,
outlier shrinkage, and detection of a strongly reversing feature pair. It built
linear, nonlinear, recent-history, time-aware, and segment-local logistic
models alongside histogram boosting, Extra Trees, and a smoothed segment prior.
Validation selected one variant per model family, chose a probability blend,
and fitted regularized logit calibration before refitting the selected members
on all labeled rows.

- Selection F: `0.196844`, rank 1
- Final F: `0.204762`, rank 3
- Reward: `1.000000`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: the most explicit treatment of growing measurement
  contamination and train-to-test correlation reversal

This was the highest fully valid selection run, so its exact archived source
became the frozen golden reference.

## Run 02 — segment-specific mixtures

Run 02 combined robust preprocessing, missing/outlier flags, quadratic and
hinge features, global and per-segment logistic models, random forests, and
histogram boosting. It compared full and reduced feature sets after looking for
unstable target associations. Instead of using one global blend, it learned
different validation-weighted model mixtures for different segments and
reweighted validation toward the test segment distribution. Its calibration
was deliberately shrunk toward no adjustment.

- Selection F: `0.180018`, rank 6
- Final F: `0.192031`, rank 6
- Reward: `0.937827`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: prediction-time ensemble weights varied by segment

## Run 03 — permutation screening

Run 03 constructed robust linear, engineered nonlinear, tree, and segment-prior
candidates. It used an out-of-time Extra Trees permutation test: a validation
feature was shuffled, and features whose presence appeared actively harmful
could be removed. Candidate probabilities received discrete temperature
calibration. The best variant from each family entered a greedy blend of at
most three members before refitting.

- Selection F: `0.175925`, rank 8
- Final F: `0.187908`, rank 7
- Reward: `0.917688`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: validation permutation screening for harmful inputs

## Run 04 — old/refit hybrid

Run 04 used median imputation, missing flags, winsorization, robust scaling,
polynomial and segment-interaction features, logistic regression, Extra Trees,
and histogram boosting. It could drop one feature after detecting both a
correlation reversal and a scale change. Validation compared single models and
two-family blends with bounded Platt calibration. Final probabilities mixed 20%
from the original train-only fit with 80% from the train-plus-validation refit.

- Selection F: `0.184182`, rank 4
- Final F: `0.196248`, rank 5
- Reward: `0.958419`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: retained part of the old model to reduce refit shift

## Run 05 — rich features with excessive runtime

Run 05 produced robust numeric features, missing/extreme indicators, time and
segment features, polynomial, trigonometric, pairwise, and segment-contrast
expansions. It evaluated logistic, histogram-boosting, and Extra Trees families,
with uniform or exponential recency weighting and validation-selected
temperature scaling. It retained the strongest families and refit them for
test prediction.

- Selection F: `0.146587`, only 8/10 valid and therefore ineligible
- Final F: `0.208968`, rank 2 among final-valid runs
- Reward: `1.000000`
- Validity: two selection timeouts; 10/10 final
- Distinguishing feature: strong final prediction quality, but the broad search
  exceeded the time limit during selection

Run 05 is excluded from the controlled-valid diversity summary even though its
final raw score exceeded the frozen golden.

## Run 06 — quantile features and a hard recent window

Run 06 used median imputation followed by a Gaussian quantile transform,
missingness indicators, segment encoding, nonlinear expansions, and optional
time. It compared logistic regression, random forests, and Extra Trees. It used
unlabeled test covariates to detect a correlation sign reversal, while
validation decided whether removing either feature actually helped. A recent
candidate retained the newest 72% of history. A greedy blend admitted only
models that improved held-out log loss, followed by damped logit calibration.

- Selection F: `0.181927`, rank 5
- Final F: `0.198213`, rank 4
- Reward: `0.968018`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: quantile normalization plus a fixed recent-history
  window

## Run 07 — segment calibration and frozen/refit blending

Run 07 used robust scaling, extreme-value handling, segment interactions,
linear models, global and segment-local gradient boosting, random forests, and
Extra Trees. Some candidates received recency and future-segment reweighting.
Validation selected a small convex ensemble. Calibration contained a global
slope and intercept plus shrunken segment offsets. Its final prediction mixed
30% from models fitted only on the original training data with 70% from models
refitted on train plus validation.

- Selection F: `0.192367`, rank 3
- Final F: `0.187341`, rank 8
- Reward: `0.914922`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: segment-level calibration combined with a 30/70
  frozen/refit ensemble

## Run 08 — compact linear/tree blend

Run 08 used robust scale-free features, missing and extreme-value flags,
segment interactions, and event time. It detected one large train-to-test
correlation reversal and discarded the member whose target association appeared
less stable. It compared regularized logistic regression and Extra Trees,
weighted validation rows toward the test segment mixture, selected a one- or
two-model blend, optionally mixed in the historical prior, and applied
conservative logit calibration.

- Selection F: `0.195192`, rank 2
- Final F: `0.212289`, rank 1
- Reward: `1.000000`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: the smallest model-family set among the top runs, and
  the best final raw score

Run 08 did not replace the golden because final-bank results were not used for
retrospective selection.

## Run 09 — calibrated family champions

Run 09 built robust linear and nonlinear feature blocks with missing/extreme
flags, pair terms, segment interactions, and optional recent history. It compared
logistic regression, random forests, Extra Trees, and gradient boosting. It
screened for an unstable feature pair, selected one champion per family, refit
those champions, corrected for the change caused by refitting, and averaged
their probabilities using validation-loss softmax weights.

- Selection F: `0.178853`, rank 7
- Final F: `0.180386`, rank 9
- Reward: `0.880953`
- Validity: 10/10 selection, 10/10 final
- Distinguishing feature: explicit compensation for the calibration shift
  introduced by refitting

This was the lowest reward among programs valid on every selection and final
scenario.

## Run 10 — ambitious segment-local search with timeouts

Run 10 used robust feature mapping, unstable-column detection, continuous
recency weighting, time and engineered interactions, and both global and local
segment models. Its candidates included logistic regression, histogram
boosting, random forests, Extra Trees, and segment-local forests. Validation
greedily added models to a blend and fitted probability calibration before
refitting the chosen members.

- Selection F: `0.172639`, rank 9
- Final F: `0.110058`; no final rank because only 7/10 completed
- Reward: `0.537494`
- Validity: 10/10 selection; three final timeouts
- Distinguishing feature: the largest reliability drop between selection and
  final because several expensive final fits exceeded 45 seconds

## Overall pattern

The same prompt did not produce ten unrelated paradigms. Most runs converged on
a common high-level pattern: robust preprocessing, linear and tree candidates,
chronological validation, blending, and calibration. Differentiation came from
how each run handled unstable features, recency, segments, refitting, and
runtime.

That convergence explains both findings:

- meaningful raw-score and ranking differences remained;
- strong-run rewards clustered near the frozen reference, producing partial
  saturation.
