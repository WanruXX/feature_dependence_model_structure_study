# Feature Dependence and Model Structure

A synthetic classification study separating feature dependence, target interactions, observation dependence and fitting/tuning allocation. The selected suite contains seven cases. Both generators produce **55 columns: 54 varying features and one constant**. Every case has 20,000 rows, allocated as 12,000 fitting, 4,000 tuning and 4,000 test rows.

See [ANALYSIS.md](ANALYSIS.md) for observed results and bounded conclusions, and [selected_study.ipynb](selected_study.ipynb) for the runnable study. The complete checked export is [results.json](results/selected/20261024T025852680255Z/results.json); [comparison.csv](results/selected/20261024T025852680255Z/comparison.csv) contains the aggregate table.

## Questions and Summary

- **Does feature dependence determine the predictive structure needed?** No. Within each generator, the same features support additive and interaction targets with different model contrasts. Input dependence and target interactions are distinct.
- **When does interaction capacity help?** Both product-target cases show large gains for interaction-capable models over static additive variants. Static gates do not substitute for cross-feature representation.
- **Does an additive target guarantee the additive model wins?** No. Attention has the lowest observed loss on Gaussian additive; MLP has the lowest on structured additive. Target expressiveness alone does not determine finite-budget fitting performance.
- **Is one flexible architecture consistently best?** No. Attention leads the Gaussian interaction case, while low-rank leads structured interaction. These are case-specific observations, not a universal ordering.
- **Do redundancy-informed gates help?** Conditionally. The reconstructability gate improves over additive and shuffled controls in the persistent-prefix case, but not over additive in the independent-prefix case. Its persistent-case mean advantage over the fixed gate is tiny, so learned correction has no established robust advantage.
- **Do observation dependence and allocation matter separately?** Paired prefix cases show that changing label dependence can change the model comparison while features and target probabilities stay fixed. The selected cases do not isolate allocation's effect.
- **Are redundant inputs themselves beneficial?** Not established. Gate comparisons are not comparisons with and without redundant inputs; the study does not identify a general benefit or harm from redundancy.

See [Conclusions](ANALYSIS.md#conclusions) for supporting cases and limits. These answers are descriptive: core cases have one seed and prefix cases three, with no statistical-significance or future-only forecasting claim.

## Selected Cases

Feature indices below are zero-based. Core cases use all 14 models at seed 42. The two prefix cases use six models at seeds 73, 74 and 75.

| Case | Features | Target logit before intercept | Signal columns | Label-uniform persistence | Development allocation |
| --- | --- | --- | --- | ---: | --- |
| gaussian_additive | Gaussian groups | $1.5A_G$ | 0, 6, 12 | 0 | random |
| gaussian_interaction | same Gaussian groups | $1.5I_G$ | 0, 6 | 0 | random |
| structured_additive | structured bounded | $A_S$ | 0, 1, 2 | 0.92 | random |
| structured_mixed | same structured bounded | $A_S+0.2I_S$ | 0, 1, 2 | 0.92 | random |
| structured_interaction | same structured bounded | $I_S$ | 0, 1 | 0.92 | random |
| structured_independent_prefix | same structured bounded | $A_S$ | 3, 4, 5 | 0 | source prefix fit / suffix tune |
| structured_persistent_prefix | same structured bounded | $A_S$ | 3, 4, 5 | 0.92 | source prefix fit / suffix tune |

The two prefix cases change only label-uniform persistence at a paired seed. Relative to the core structured additive case, they also change signal placement and allocation; that comparison does not isolate either change individually. Attention and residual models are not evaluated in the prefix cases.

## Gaussian Synthesis

Rows are independent. Draw nine independent standard-normal group factors $G_k$ and 54 independent standard-normal noises $E_j$ for every row. Let $\ell_j$ be a seeded permutation of 54 equally spaced values from 0.2 to 1.0; its permutation seed is the generation seed plus 31. For $j=0,\ldots,53$:

$$X_j=\sqrt{0.9\ell_j}\,G_{\lfloor j/6\rfloor}+\sqrt{1-0.9\ell_j}\,E_j.$$

Columns have unit population variance. Within-group pairwise correlations are $0.9\sqrt{\ell_j\ell_k}$, not uniformly 0.9; different groups are independent. Column 54 is 0.5. With $a=X_0$, $b=X_6$, $c=X_{12}$:

$$A_G=1.2\tanh(a)-0.8\tanh(b)+0.5(c^2-1),\qquad I_G=ab.$$

Use the case-specific logit in the table, with no intercept adjustment. Draw independent uniform values and set $Y=\mathbf{1}[U<\mathrm{sigmoid}(\eta)]$. Feature generation and uniform draws are paired between targets at the same seed. These results are newly fitted at 55 columns; no different-width comparison is used.

## Structured Synthesis

All feature sources are generated in source-row order. For each independent source, initialize $Z_0\sim N(0,1)$ and use $Z_t=0.8Z_{t-1}+\sqrt{1-0.8^2}\,\epsilon_t$, with independent standard-normal innovations. Draw 11 sources, then 54 independent idiosyncratic sources with the same persistence. Transform source zero to $F_0=(Z_0+0.5(Z_0^2-1))/\sqrt{1.5}$; other factors remain unchanged.

For active column $j\geq3$, construct the unnormalized value

$$V_j=0.45F_0+0.8F_{1+(j\bmod8)}+\left(0.2+0.15\frac{j\bmod5}{4}\right)F_{1+((j+1)\bmod8)}+0.1\cos(j)F_9+d_jE_j+w_jF_0F_{10},$$

where $d_j=0.22(0.35+1.3(j\bmod7)/6)$ and $w_j=0.35(0.5+0.5\sin(j))$. Divide by the square root of the sum of squared factor loadings, $d_j^2$ and $w_j^2$. Columns 0, 1 and 2 instead use their independent idiosyncratic source directly, without that loading mixture or normalization.

Apply the standard-normal CDF, cast to float32, raise column $j$ to the power $0.85+0.35(j\bmod5)/4$, then round to two decimals if $j\bmod11=10$ and five decimals otherwise. Set column 54 to 0.5. The CDF is a bounded mapping, not a claim that the nonlinear mixtures are normally distributed or that the resulting marginals are uniform. Normalization uses fixed source second moments, not fitted population statistics.

For the three signal columns of the case, define $a=2X_{s_0}-1$, $b=2X_{s_1}-1$, $c=2X_{s_2}-1$:

$$A_S=0.7\left(2a-1.5b+0.8(c^2-1/3)\right),\qquad I_S=3ab.$$

Add an intercept, found by 60 bisection steps on [-40,40], so the mean simulated target probability is 0.31. This is a generator definition using the simulated unlabeled population, not a learned preprocessing step. Sigmoid arguments are clipped to [-60,60].

Independently generate one stationary normal source $L_t$ with persistence $\rho_Y$ from the case table, using generation seed plus 10,000. Set $U_t=\Phi(L_t)$ and $Y_t=\mathbf{1}[U_t<p_t]$. Each row retains marginal success probability $p_t$; persistent uniforms add observation dependence. With $\rho_Y=0$, uniforms are independent, but features and therefore probabilities still persist across rows. Changing $\rho_Y$ leaves features and target probabilities identical.

## Allocation and Fitting

Use NumPy's default generator at split seed 42 to permute all row positions. The final 4,000 positions form the fixed test set. Random allocation uses the first 12,000 for fitting and the next 4,000 for tuning. Prefix allocation sorts only the first 16,000 positions, then assigns the earlier 12,000 to fitting and the later 4,000 to tuning. Test membership is unchanged. This is a mixed-period test, not a future-only forecasting evaluation.

Fit each column's mean and population standard deviation on fitting rows only; clamp scales below $10^{-6}$. The constant becomes zero. Compute gate scores only from fitting features: mean absolute off-diagonal correlation, held-out ridge reconstructability, or their mean. Reconstructability uses a seeded 70/30 internal split, internal-fit scaling, ridge 0.01 and $R^2$ clipped to [0,1]. No labels are used in these scores.

Every model uses AdamW, batch size 256, gradient-norm clipping at 5, weight decay 0.0001 and learning rates {0.001, 0.003}. Each trial allows 300 epochs, patience 30 and checkpoint improvement threshold $10^{-5}$. Checkpoints and learning rates are selected exclusively by tuning log loss. The test set does not select settings. Initialization and minibatch ordering are paired within a seed; generation and model initialization share the reported seed, while split seed remains 42. Reference fits use one CPU Torch thread.

## Models and Controls

The additive core has four SiLU units per feature, summed with a scalar intercept. A static gate multiplies each feature contribution by

$$g_j=\left(0.25+0.75e^{-s_j}\right)e^{0.5\tanh(\theta_j)}.$$

Learned corrections are penalized by $0.001\,\mathrm{mean}[(e^{0.5\tanh\theta}-1)^2]$. `learned_gate` uses zero scores; correlation, reconstructability and combined variants use their respective fitting-only scores. Shuffled controls permute the corresponding score vector using seed plus 23. Fixed controls hold corrections at one. These gates are static, not input-dependent interactions. Positive fixed gates preserve the additive function class but can change optimization and effective regularization.

| Model family | Trainable parameters at 55 columns | Definition |
| --- | ---: | --- |
| additive, fixed gates | 716 | additive core; fixed rescaling for controls |
| learned/correlation/reconstruction/combined/shuffled gates | 771 | additive core plus 55 learned corrections |
| additive_residual | 1,174 | additive contributions plus zero-initialized width-8 SiLU residual |
| redundancy_residual | 1,229 | combined gate plus the same residual |
| low_rank | 936 | additive core plus rank-4 off-diagonal quadratic interactions |
| mlp | 1,825 | width-32 SiLU network |
| attention | 3,153 | scalar and feature embeddings, dimension 16, two heads, one encoder layer, mean pooling, no dropout |

The 14 core variants are defined in [config.py](config.py). Prefix cases use `additive`, `recon_gate`, `shuffled_recon_gate`, `fixed_recon_gate`, `low_rank` and `mlp`. Matching feature width matches parameter counts **between generators for each model**, not between different model families. A fitting-prevalence constant predictor is saved as an additional baseline.

## Run

Use Python 3.12 with the packages in [requirements.txt](requirements.txt). From this directory:

```console
python -m pip install -r requirements.txt
python study.py --smoke --output-dir results/smoke
python study.py --threads 1 --output-dir results/rerun
python -m unittest discover -s tests -v
```

For a subset, use `--cases gaussian_additive gaussian_interaction`, for example. The default full run executes the selected suite only: 106 selected model fits from 212 hyperparameter trials. `--smoke` uses the same seven cases/model sets but tiny splits, one seed per case and two epochs; its output is labeled smoke and is not evidence for the analysis. The notebook defaults to full rerunning; disable `RERUN` to inspect the checked export instead. CPU execution can take substantial time. CUDA is available through `--device cuda:0`, but its numerical results and timing need not match CPU fits.

Each timestamped export contains predictions, labels, row IDs, fitting scales, redundancy scores, gate values, all learning curves, selected settings, parameter counts and aggregate CSVs. Existing exports are not overwritten. [study.py](study.py) handles selected-case execution and export; [datasets.py](datasets.py) implements only the two synthetic laws; [presentation.py](presentation.py) produces result tables and plots.

## Interpretation Boundaries

These are selected illustrative cases, not a preregistered benchmark or independent confirmation. Core results have one seed; prefix results have three paired seeds. Seed variation changes generated observations and initialization together. Differences are descriptive, not significance tests, and temporally dependent rows do not provide 20,000 independent observations. No universal architecture ranking, causal explanation from a ranking alone, or general advantage of feature redundancy is implied.
