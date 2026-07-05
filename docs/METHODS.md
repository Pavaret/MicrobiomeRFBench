# Methods

Machine-learning prediction of methotrexate (MTX) response from the baseline gut
microbiome, with permutation-based significance and a repeated-seed confidence
band on model discrimination.

## Data and encoding

For each disease group (rheumatoid arthritis, RA; psoriatic arthritis, PsA) and
feature set, a feature-by-sample table was transposed to a sample-by-feature
matrix. Two feature sets were analysed: functional pathway relative abundances
(HUMAnN3; `pathway_only`) and taxonomic species relative abundances (MetaPhlAn4
species-genome bins) augmented with two alpha-diversity indices, observed richness
and the Shannon index (`species_diversity`). The classification target was the MTX
response label, encoded as inefficiency = 1 (non-responder) and remission = 0
(responder). Non-feature metadata rows were removed by name; features with no
variance across the retained samples were dropped, and missing feature values were
set to zero.

## Nested leave-one-out cross-validation

Model discrimination was estimated with nested leave-one-out cross-validation
(LOO-CV). In each outer fold a single patient was held out. Within the training
partition an impurity-based random-forest feature screen retained the 50
highest-importance features (200-tree random forest, balanced class weights,
maximum depth 5), and the retained features were passed to an inner
hyperparameter search: a random-forest classifier (balanced class weights) tuned
by grid search over the number of trees (300, 500), maximum depth (unlimited, 4,
6), minimum samples per leaf (1, 2, 3) and the number of features considered per
split (√p, 0.3·p), scored by the area under the receiver-operating-characteristic
curve (AUROC) with three-fold stratified inner cross-validation. The tuned model
predicted the held-out patient's probability of inefficiency. Concatenating the
held-out predictions across all outer folds yielded one out-of-fold probability
per patient, and the AUROC of these out-of-fold probabilities against the true
labels was the model's observed discrimination.

All randomised components — the feature screen, the inner-cross-validation
shuffling and the random forests — were seeded deterministically from a single
base seed, so that a run is fully reproducible.

## Permutation significance

The statistical significance of each model was assessed by a label-permutation
test on the base-seed model (base seed 42). The response labels were randomly
permuted 1,000 times; for each permutation the entire nested LOO-CV was re-run on
the permuted labels and its AUROC recorded, forming the null distribution of
discrimination expected under no true association. The one-sided permutation
p-value was `(#{null AUROC ≥ observed AUROC} + 1) / (number of permutations + 1)`.

## Repeated-seed estimation of the AUROC confidence band

To quantify the run-to-run stability of model discrimination and to attach a
confidence band to the ROC curve, the complete nested LOO-CV was repeated one
hundred times using one hundred different base seeds (42–141). The set of held-out
samples and the patients themselves were identical across repeats; only the
stochastic components of the pipeline varied with the seed, each derived
deterministically from the base seed. Consequently the variability summarised
below reflects the algorithm's run-to-run stochasticity rather than
patient-sampling uncertainty; the base-seed repeat (seed 42) is the model the
permutation test evaluated, so the two analyses describe the same fitted model.

For each repeat the out-of-fold predicted probabilities and the corresponding
AUROC were computed. The one hundred ROC curves were each linearly interpolated
onto a common grid of 201 false-positive rates spanning 0 to 1, and at every grid
point the mean and standard deviation of the true-positive rate across repeats
were obtained. The plotted band is the two-sided 95% confidence interval of the
mean ROC, defined per grid point as the mean true-positive rate ±
t₍₉₉, ₀.₉₇₅₎ × SEM, where SEM is the standard deviation across the one hundred
repeats divided by √100 and t₍₉₉, ₀.₉₇₅₎ = 1.984 is the 97.5th percentile of
Student's t distribution with ninety-nine degrees of freedom; the band was clipped
to the unit interval. The reported discrimination is the mean AUROC across the one
hundred repeats with the analogous t-based 95% confidence interval. For the
threshold-dependent confusion matrix, the out-of-fold probabilities were averaged
across repeats and samples with a mean predicted probability of inefficiency of at
least 0.5 were classified as inefficiency.

## Dispersion of feature importance across repeats

Within each repeat, a stability-weighted random-forest importance score was
computed for every feature as the mean per-outer-fold impurity importance, with
folds in which the feature was not selected contributing zero. Features were then
ranked by their mean score across repeats, and the standard deviation across
repeats was reported as an error bar in the feature-importance figure, so that the
plot conveys both the typical importance of each feature and the consistency of
that importance across reruns. The accompanying table additionally records, per
feature, the mean selection frequency and the individual per-seed importance
scores.

## Software

All analyses used Python 3.12 with scikit-learn 1.6.1, NumPy, pandas, joblib and
SciPy (the last used only to evaluate the t critical value; a hardcoded t table
serves as a fallback if SciPy is unavailable). Figures were produced with
Matplotlib and exported as SVG. Exact pinned versions are in `requirements.txt`.
