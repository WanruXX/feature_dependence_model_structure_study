# Observed Results

All cases have 55 columns (54 varying, one constant), 20,000 rows and identical 12,000/4,000/4,000 fitting/tuning/test sizes. Lower test log loss is better. Full-precision scores, predictions, trial histories and validation selections are in [results.json](results/selected/20261024T025852680255Z/results.json). The [README](README.md) specifies the feature laws, target equations and model definitions.

## Core Cases

All 14 models use seed 42. The Gaussian and structured interaction targets introduce explicit cross-feature terms; their additive counterparts do not. The feature matrices and row allocations are paired across targets within each generator. Columns below abbreviate Gaussian additive/interaction and structured additive/mixed/interaction.

| Model | G additive | G interaction | S additive | S mixed | S interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| additive | 0.513530 | 0.695205 | 0.513155 | 0.522298 | 0.600404 |
| learned_gate | 0.511335 | 0.695189 | 0.513172 | 0.522159 | 0.601664 |
| corr_gate | 0.511218 | 0.695058 | 0.512945 | 0.521904 | 0.601530 |
| recon_gate | 0.512305 | 0.694352 | 0.512662 | 0.521723 | 0.601258 |
| redundancy_gate | 0.512911 | 0.694645 | 0.512778 | 0.521638 | 0.600746 |
| shuffled_gate | 0.512656 | 0.694699 | 0.512875 | 0.521753 | 0.600725 |
| fixed_redundancy_gate | 0.511969 | 0.694652 | 0.512680 | 0.521712 | 0.601128 |
| additive_residual | 0.512555 | 0.580656 | 0.513079 | 0.522366 | 0.527238 |
| redundancy_residual | 0.512578 | 0.577767 | 0.512526 | 0.519369 | 0.527355 |
| low_rank | 0.520188 | 0.579283 | 0.511880 | 0.519665 | 0.526083 |
| mlp | 0.528493 | 0.582970 | 0.508192 | 0.514736 | 0.526348 |
| attention | 0.510454 | 0.568981 | 0.517852 | 0.521190 | 0.528006 |
| shuffled_recon_gate | 0.513225 | 0.694431 | 0.512738 | 0.521685 | 0.600608 |
| fixed_recon_gate | 0.512232 | 0.694356 | 0.512642 | 0.521799 | 0.601036 |

**Gaussian additive.** Attention has the lowest observed loss, 0.510454, compared with additive at 0.513530 and MLP at 0.528493. The target is additive, so this ranking cannot be explained as a necessary benefit of interaction capacity. Correlation and generic learned gates also improve the additive score in this fit. The result shows that matching the target's algebraic class does not guarantee the best finite-sample, finite-budget fit.

**Gaussian interaction.** Attention scores 0.568981, low-rank 0.579283, MLP 0.582970 and additive 0.695205. Residual models also improve substantially, while static-gate models remain near 0.694-0.695. This is consistent with the known product target requiring cross-feature representation; a static rescaling does not supply that representation. It does not establish attention's superiority beyond this seed and budget.

**Structured additive and mixed.** MLP has the lowest observed losses, 0.508192 and 0.514736, respectively, versus additive at 0.513155 and 0.522298. Attention changes from 0.517852 on additive to 0.521190 on mixed and is not the best model on either. The pure additive target still permits an MLP advantage under the implemented training protocol; that advantage alone is not evidence of a hidden interaction in the additive target.

**Structured interaction.** Low-rank scores 0.526083 and MLP 0.526348, versus additive at 0.600404. Residual and attention models score 0.527238-0.528006. All interaction-capable variants improve substantially over the static additive variants. Low-rank's edge over MLP is only about 0.000265 and comes from one seed, so the evidence supports an interaction-capacity contrast more strongly than an ordering within that group.

Changing target structure while holding the generator's feature matrix fixed changes the architecture contrast. Comparing the two feature laws also changes the ordering among flexible models. These observations separate the relevance of input dependence from the relevance of the input-to-target function; correlated features alone do not determine a model winner.

## Prefix Cases

Both cases place the additive signal on structured columns 3, 4 and 5, retain persistent feature sources, and use source-prefix fitting/latest-development tuning with unchanged mixed-period test membership. They differ only in label-uniform persistence: 0 versus 0.92. Entries are means over paired seeds 73, 74 and 75. Only the following six models were fitted.

| Model | Independent uniforms | Persistent uniforms |
| --- | ---: | ---: |
| additive | 0.566566 | 0.561124 |
| recon_gate | 0.566818 | 0.560342 |
| shuffled_recon_gate | 0.567088 | 0.561037 |
| fixed_recon_gate | 0.567150 | 0.560381 |
| low_rank | 0.569290 | 0.563941 |
| mlp | 0.570270 | 0.562097 |

With independent uniforms, additive has the lowest loss in all three seeds among these six models. The reconstructability gate is worse by a mean 0.000251. With persistent uniforms, the reconstructability gate has the lowest mean and wins two of three seeds; MLP wins seed 73. The gate improves over additive in all three seeds, by a mean 0.000782, and improves over the shuffled control in all three, by a mean 0.000695.

The persistent case's fixed gate is almost tied with the learned reconstructability gate: their mean difference is only 0.000038679, with the learned version ahead in two of three seeds. Thus these cases provide descriptive evidence of condition-dependent shrinkage behavior, not a robust benefit from learned correction over fixed rescaling. They do not establish that the gate is better than attention, which was not evaluated here. The two cases have different sampled labels and test prevalences, so their absolute loss difference is not itself a measure of model improvement.

The prefix cases are not a one-factor comparison with the core structured additive case: signal placement, allocation and seed sets also differ. Conclusions about either placement or allocation individually cannot be drawn from that comparison.

## Budget and Limits

Both generators use identical feature width and per-model parameter counts. Additive and fixed gates have 716 parameters, learned gates 771, additive residual 1,174, gated residual 1,229, low-rank 936, MLP 1,825 and attention 3,153. Capacity is matched across generators, not across model families.

All 212 trials use the same two learning rates, checkpoint criterion and early-stopping budget. One trial reaches the 300-epoch cap: structured mixed, gated residual, learning rate 0.001, with best checkpoint at epoch 278. No other trial reaches the cap. This leaves a budget limitation for that trial; no extra optimization was allocated according to test rankings.

Core comparisons contain one seed, prefix comparisons three. These selected cases are illustrative rather than an independent confirmatory benchmark. No confidence interval or statistical-significance claim is made; row dependence also limits independence-based uncertainty calculations. Test rows span source periods rather than representing a strictly later forecasting period.

## Conclusions

- **Does feature dependence determine the predictive structure a model needs?** No. Gaussian additive and interaction targets share the same features and allocation, as do the three core structured targets. Nevertheless, their additive-versus-interaction-capable model contrasts differ substantially. Feature dependence describes the distribution of inputs; it does not by itself identify the input-to-target function.

- **When is interaction capacity useful, and can static gates replace it?** In both explicit product-target cases, interaction-capable models substantially outperform static additive variants: Gaussian attention scores 0.568981 versus additive at 0.695205; structured low-rank scores 0.526083 versus additive at 0.600404. Static-gate variants stay near their additive baselines. This supports the usefulness of interaction capacity for these targets, not the necessity of any particular interaction architecture.

- **Does an additive target guarantee the additive architecture fits best?** No. On Gaussian additive, attention scores 0.510454 versus additive at 0.513530. On structured additive, MLP scores 0.508192 versus additive at 0.513155. An additive target specifies the generating function, but does not guarantee that the implemented finite-width additive network wins under the chosen sample size, regularization and optimization budget. These rankings do not imply hidden target interactions.

- **Is attention, an MLP or another flexible model consistently preferable?** No universal ordering is supported. Attention has the lowest observed loss in the two Gaussian cases; MLP leads structured additive and mixed; low-rank leads structured interaction. The low-rank/MLP interaction gap is only about 0.000265. These comparisons involve different parameter counts and one core seed, so they support case-specific observations rather than a general architecture recommendation.

- **Do informative gates improve prediction, and is learned correction responsible?** The evidence is conditional. In the persistent-prefix case, the reconstructability gate beats additive and the shuffled control in all three seeds, with mean improvements of 0.000782 and 0.000695. In the independent-prefix case, additive wins all three seeds among the six evaluated models. The learned gate's persistent-case mean edge over the fixed gate is only 0.000038679, so a robust benefit from learned correction is not established. Static-gate gains are compatible with fitting or regularization effects, not added interaction expressiveness.

- **What do observation dependence and allocation contribute?** With feature values, target probabilities, signal placement and prefix allocation paired, changing label-uniform persistence changes the gate-versus-additive comparison. Thus the rowwise target probability function alone does not determine the observed finite-sample comparison. However, these cases do not isolate allocation's effect: comparing prefix cases with the core additive case also changes signal placement and seed sets. Mixed-period test results do not establish future-only forecasting performance.

- **Does redundancy itself help or hurt?** This remains unresolved in general. The selected suite compares models and gate controls, not otherwise matched inputs with and without redundant columns. Therefore it does not identify redundancy's causal effect or establish that stronger feature dependence favors a particular architecture. The supported finding is narrower: static weighting can change fitting outcomes under some selected conditions.

These answers are bounded by the selected cases, seed counts and training budget described above. They are descriptive findings, not statistical-significance claims or independent confirmation. The strongest observed contrast is the large benefit of interaction capacity on the two product targets; small differences among gates or flexible architectures warrant much more caution.