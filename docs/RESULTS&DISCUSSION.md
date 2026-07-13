# Results and Discussion Draft

## 4. Results

### 4.1 Experimental setup

The final ROHE-style dataset contains 522 removed-object samples. Each sample consists of an object-removed image, a yes/no question about the removed target object, and a semantic token-region map assigning visual patch tokens to one of three regions: removed object region, surrounding context region, or background region. All removed-image questions have label `no`, so a model answer beginning with “Yes” is counted as an object hallucination, while an answer beginning with “No” is counted as a correct rejection.

The experiment evaluates five masking conditions:

```text
none        no uncertain-token masking
all         suppress all uncertain visual tokens
removed     suppress uncertain tokens in the removed-object region
context     suppress uncertain tokens in the surrounding context region
background  suppress uncertain tokens in the background region
```

The causal effect is computed relative to the no-masking baseline:

```text
causal effect = hallucination_rate_none - hallucination_rate_masked
```

A positive causal effect means that masking reduced hallucination.

---

### 4.2 LLaVA-1.5-7B results

LLaVA-1.5-7B hallucinated the removed object in 415 out of 522 samples under the no-masking baseline, corresponding to a hallucination rate of 79.50%.

| Condition  | Hallucinated yes | Correct rejection no | Hallucination rate | Effect vs none |
| ---------- | ---------------: | -------------------: | -----------------: | -------------: |
| none       |              415 |                  107 |             79.50% |        0.00 pp |
| all        |              396 |                  126 |             75.86% |       +3.64 pp |
| removed    |              409 |                  113 |             78.35% |       +1.15 pp |
| context    |              412 |                  110 |             78.93% |       +0.57 pp |
| background |              399 |                  123 |             76.44% |       +3.07 pp |

Global uncertain-token masking produced the strongest overall reduction, lowering hallucination by 3.64 percentage points. Among the region-specific interventions, background-region masking was the strongest, reducing hallucination by 3.07 percentage points. Removed-region and context-region masking produced smaller reductions.

Bootstrap analysis confirms that the global and background effects are statistically reliable:

| Condition  |   Effect |        95% CI | p-value |
| ---------- | -------: | ------------: | ------: |
| all        | +3.64 pp |  [1.53, 5.94] |  0.0014 |
| removed    | +1.15 pp |  [0.00, 2.30] |  0.0642 |
| context    | +0.57 pp | [-0.57, 1.72] |  0.3840 |
| background | +3.07 pp |  [0.96, 5.17] |  0.0036 |

These results support the hypothesis that epistemically uncertain visual tokens can causally contribute to hallucination in LLaVA-1.5-7B. However, the strongest region-specific effect is not in the removed-object region itself, but in the background region.

---

### 4.3 Answer-flip analysis for LLaVA-1.5-7B

To understand whether the reduction is caused by genuine corrections, we compared each masking condition against the no-masking baseline at the sample level.

| Condition  | Yes→No | No→Yes | Unchanged Yes | Unchanged No | Net reduction |
| ---------- | -----: | -----: | ------------: | -----------: | ------------: |
| all        |     28 |      9 |           387 |           98 |           +19 |
| removed    |      8 |      2 |           407 |          105 |            +6 |
| context    |      6 |      3 |           409 |          104 |            +3 |
| background |     23 |      7 |           392 |          100 |           +16 |

Here, `yes→no` means that masking corrected a hallucination, while `no→yes` means that masking broke a previously correct rejection.

The answer-flip analysis shows that the LLaVA-1.5-7B effect is driven by real corrections. Global masking corrected 28 hallucinated answers and introduced only 9 new hallucinations, giving a net reduction of 19 hallucinations. Background masking corrected 23 hallucinations and introduced 7 new hallucinations, giving a net reduction of 16 hallucinations.

This explains why global and background masking have statistically reliable positive effects.

---

### 4.4 LLaVA-1.5-13B robustness results

We repeated the same five-condition experiment on LLaVA-1.5-13B as a model-size robustness check.

| Condition  | Hallucinated yes | Correct rejection no | Hallucination rate | Effect vs none |
| ---------- | ---------------: | -------------------: | -----------------: | -------------: |
| none       |              463 |                   59 |             88.70% |        0.00 pp |
| all        |              466 |                   56 |             89.27% |       -0.57 pp |
| removed    |              466 |                   56 |             89.27% |       -0.57 pp |
| context    |              464 |                   58 |             88.89% |       -0.19 pp |
| background |              466 |                   56 |             89.27% |       -0.57 pp |

Unlike LLaVA-1.5-7B, the 13B model did not show a reduction in hallucination under any masking condition. The baseline hallucination rate was higher than 7B, and all masking effects were small, negative, and statistically non-significant.

| Condition  |    Effect |          95% CI | p-value |
| ---------- | --------: | --------------: | ------: |
| all        | -0.575 pp | [-2.874, 1.533] |  0.6588 |
| removed    | -0.575 pp | [-1.724, 0.575] |  0.4016 |
| context    | -0.192 pp | [-1.149, 0.575] |  0.8472 |
| background | -0.575 pp | [-2.682, 1.533] |  0.6800 |

These results suggest that the causal effect observed in 7B is not scale-invariant across LLaVA-1.5 model sizes.

---

### 4.5 Answer-flip analysis for LLaVA-1.5-13B

The answer-flip analysis explains why 13B does not show a positive masking effect.

| Condition  | Yes→No | No→Yes | Unchanged Yes | Unchanged No | Net reduction |
| ---------- | -----: | -----: | ------------: | -----------: | ------------: |
| all        |     16 |     19 |           447 |           40 |            -3 |
| removed    |      3 |      6 |           460 |           53 |            -3 |
| context    |      2 |      3 |           461 |           56 |            -1 |
| background |     15 |     18 |           448 |           41 |            -3 |

LLaVA-1.5-13B does change some answers under masking, but the changes are mixed. For example, global masking corrects 16 hallucinations, but it also changes 19 correct rejections into hallucinations. Therefore, the helpful and harmful flips cancel out, resulting in no reliable reduction.

This suggests that 13B is not simply insensitive to masking. Rather, the same intervention does not consistently move the model in the correct direction.

---


### 4.6 Matched random-token controls

To determine whether the 7B effect is specific to uncertainty-guided selection, each high-uncertainty masking condition was paired with a matched random control. For every sample, the random control masked exactly the same number of tokens from the same eligible semantic region. Five deterministic random seeds were evaluated.

| Condition | High-uncertainty rate | Random mean $\pm$ SD | Mean high-uncertainty advantage |
| --- | ---: | ---: | ---: |
| all | 75.86% | 75.75% $\pm$ 0.90 | -0.11 pp |
| removed | 78.35% | 78.58% $\pm$ 0.55 | +0.23 pp |
| context | 78.93% | 79.39% $\pm$ 0.37 | +0.46 pp |
| background | 76.44% | 76.90% $\pm$ 0.50 | +0.46 pp |

The differences are small relative to random-seed variation. High-uncertainty masking therefore does not show a large or consistent advantage over matched random masking. This indicates that masking count and masking location explain a substantial part of the observed hallucination reduction.

---

### 4.7 Matched low-uncertainty controls

A second control suppressed the same number of lowest-uncertainty tokens from the same region.

| Region | High-uncertainty rate | Matched low-uncertainty rate | High-uncertainty advantage |
| --- | ---: | ---: | ---: |
| all | 75.86% | 79.50% | +3.64 pp |
| removed | 78.35% | 79.50% | +1.15 pp |
| context | 78.93% | 79.50% | +0.57 pp |
| background | 76.44% | 79.50% | +3.07 pp |

Global and background high-uncertainty masking significantly outperformed matched low-uncertainty masking. In addition, all four low-uncertainty conditions produced exactly the same 522 generated answers as the no-masking baseline.

This shows that the uncertainty ranking contains meaningful information: suppressing the lowest-uncertainty tokens has no observable effect, whereas suppressing high-uncertainty global and background tokens changes model decisions.

---

### 4.8 Original-image sanity check

The same five masking conditions were evaluated on the original images, where the target object is genuinely present and the correct answer is `yes`.

| Condition | Correct yes | False-negative no | Accuracy | Drop vs none |
| --- | ---: | ---: | ---: | ---: |
| none | 494 | 28 | 94.64% | 0.00 pp |
| all | 482 | 40 | 92.34% | -2.30 pp |
| removed | 492 | 30 | 94.25% | -0.38 pp |
| context | 492 | 30 | 94.25% | -0.38 pp |
| background | 484 | 38 | 92.72% | -1.92 pp |

Paired bootstrap analysis showed statistically supported accuracy drops for global masking and background masking:

| Condition | Accuracy drop | 95% CI | p-value |
| --- | ---: | ---: | ---: |
| all | 2.30 pp | [0.77, 4.02] | 0.0052 |
| removed | 0.38 pp | [-0.38, 1.15] | 0.4550 |
| context | 0.38 pp | [0.00, 0.96] | 0.2760 |
| background | 1.92 pp | [0.38, 3.45] | 0.0136 |

These results show that global and background masking do not only reduce hallucinated `yes` answers on removed images. They also remove useful visual evidence and convert some correct `yes` answers into false-negative `no` answers on original images.

---

### 4.9 Active-suppression ablation

Some region-specific conditions suppress no tokens for some samples. To check whether weak effects were caused by inactive masks, the analysis was repeated only on samples where the selected condition suppressed at least one uncertain token.

For LLaVA-1.5-7B, removed-region masking was active for 385 samples and produced a 1.56 percentage-point reduction. Context masking was active for 456 samples and produced a 0.66 percentage-point reduction. Global and background masking were active for all 522 samples and retained their original effects.

For LLaVA-1.5-13B, masking was also active on many samples, but the effects remained small and negative. Therefore, the difference between 7B and 13B is not explained by inactive region masks.

---

### 4.10 Summary of main findings


The main experimental findings are:

```text
1. LLaVA-1.5-7B shows statistically reliable hallucination reductions under global and background high-uncertainty masking relative to no masking.

2. High-uncertainty global and background masking clearly differs from matched low-uncertainty masking, which produces no answer changes.

3. High-uncertainty masking does not consistently outperform matched random masking across five seeds.

4. Global and background masking also reduce correct recognition on original images, revealing a false-negative cost.

5. LLaVA-1.5-13B does not reproduce the 7B masking effect.

6. Active-suppression analysis shows that the model difference is not caused by inactive masks.
```

Overall, the evidence supports a model-dependent relationship between epistemic uncertainty and visual influence, but not a simple claim that uncertainty-guided token selection uniquely causes hallucination reduction.

---

## 5. Discussion

### 5.1 Interpretation of the 7B result

The LLaVA-1.5-7B results show that suppressing high-uncertainty visual tokens can reduce object hallucination in removed-object images. However, the matched random controls show that this reduction is not uniquely attributable to uncertainty-guided selection. The strongest evidence is therefore that global and background visual information is causally influential, while uncertainty ranking provides only partial additional specificity.

The strongest effects come from global masking and background-region masking. This is important because the removed-object region itself is not the strongest causal region. If hallucination were mainly caused by uncertainty exactly where the object was removed, removed-region masking should have produced the strongest reduction. Instead, background masking performs much better.

This suggests that hallucination in this setting may be driven by broader scene-level uncertainty. The model may infer the removed object from surrounding visual context, co-occurring objects, or scene priors. For example, a street scene may make the model likely to answer “yes” to a car question even if the specific car has been removed. Suppressing uncertain background tokens can weaken these misleading scene-level cues.

Therefore, the 7B result supports a distributed interpretation of hallucination: the hallucinated object is not necessarily triggered only by the missing object area, but by uncertain evidence distributed across the image.

---

### 5.2 Why background masking matters

Background masking was the strongest region-specific intervention. This finding is meaningful because background regions often contain scene context, co-occurring objects, spatial layout, and environmental cues. These cues may bias the model toward predicting that a target object is present.

In removed-object counterfactuals, the target object is absent, but the surrounding image may still contain evidence that is statistically associated with the object. For example:

```text
bus       → road, street, traffic scene
bench     → park, sidewalk, outdoor scene
bicycle   → road, person, street scene
chair     → room, table, indoor scene
```

If the model relies on these contextual priors, it may hallucinate the removed object. The background masking result suggests that uncertain visual evidence outside the removed-object region can play a causal role in this behavior.

This also explains why removed-region masking is weaker. After inpainting, the removed-object area may no longer contain the strongest evidence for the object. Instead, the broader scene still supports the object prior.

---

### 5.3 Why LLaVA-13B behaves differently

LLaVA-1.5-13B does not reproduce the positive effect observed in 7B. Its no-masking hallucination rate is higher, and masking does not reduce hallucination. The answer-flip analysis shows that 13B does produce some corrected cases, but these are offset by harmful flips from correct rejection to hallucination.

One possible explanation is that the 13B language decoder relies more strongly on learned object priors and question-conditioned expectations. In a yes/no question such as:

```text
Is there a car in the image?
```

the larger model may be more likely to produce a confident “yes” based on scene priors, object co-occurrence, or language patterns, even when the image evidence is weak or contradictory.

Another possibility is that the uncertainty selection method is better aligned with the causal visual features of 7B than 13B. Although both models are LLaVA-1.5, the larger model may distribute visual evidence differently across internal representations. Therefore, the same uncertain-token suppression strategy may not target the actual causal features driving 13B’s answers.

A third possibility is that masking removes both misleading and helpful evidence. In 13B, the intervention corrects some hallucinations but also damages some correct rejections. This mixed behavior is visible in the answer-flip counts and explains the non-significant bootstrap results.

Thus, the 13B result should not be interpreted as showing that epistemic uncertainty is irrelevant. Instead, it shows that this specific region-wise masking intervention does not produce a reliable causal reduction for LLaVA-1.5-13B under the ROHE removed-object protocol.

---

### 5.4 Relation to prior epistemic uncertainty work

Prior epistemic uncertainty work argues that uncertain visual tokens are associated with object hallucination and that suppressing such tokens can reduce hallucination. This project extends that idea by asking a more fine-grained causal question:

```text
Which semantic region of uncertainty matters?
```

The 7B result is consistent with the prior claim that uncertain visual tokens can contribute to hallucination. However, the 13B robustness result shows that the effect is not uniform across model sizes. Therefore, this project refines the prior claim: epistemic uncertainty can be causally useful for hallucination reduction, but the causal effect depends on the model and the intervention design.

---


### 5.5 What the control experiments change

The control experiments substantially refine the interpretation of the main 7B result.

First, the matched low-uncertainty controls show that the uncertainty ranking is meaningful. Suppressing the least uncertain tokens has no observable effect, while suppressing high-uncertainty global and background tokens changes model decisions.

Second, the matched random controls show that uncertainty ranking is not sufficient to explain the full effect. Randomly suppressing the same number of tokens from the same region often produces similar hallucination rates. This suggests that token count, semantic location, and general information removal all contribute.

Third, the original-image sanity check shows that the intervention has a cost. Global and background masking reduce hallucination on removed images, but they also reduce correct recognition when the target object is present. The method therefore changes the model's decision boundary rather than selectively correcting hallucinations without affecting valid recognition.

The combined interpretation is that high-uncertainty global and background tokens identify visually influential parts of the representation, but suppressing them is neither uniquely uncertainty-specific nor cost-free.

---

### 5.6 Limitations



First, the evaluation is based on a controlled ROHE-style removed-object dataset with 522 samples and eight COCO object categories: dog, cat, car, chair, bicycle, bus, bottle, and bench. This controlled setting is useful because the correct answer is unambiguous: after removal, the target object should not be present. However, the results may not directly generalize to all object categories or to fully open-ended image captioning.

Second, the removed-object images are generated using inpainting. Inpainting may introduce artifacts or may not perfectly represent natural object absence. However, every masking condition is evaluated on the exact same images and questions, so the main causal comparisons are paired and fair. The analysis asks whether masking changes the model’s answer relative to its own no-masking baseline on the same samples.

Third, hallucination is measured using a yes/no answer rule: answers beginning with “Yes” are counted as hallucinations and answers beginning with “No” are counted as correct rejections. This rule is appropriate for the controlled object-presence questions used in this project and produced no unclear answers in the final runs. However, it does not capture more complex hallucination behavior in long-form captioning or free-form dialogue.

Fourth, the final model comparison is limited to LLaVA-1.5-7B and LLaVA-1.5-13B. This allows a clean model-size robustness check within the same model family, but it does not prove that the same behavior holds across all VLM architectures. MiniGPT-4 and Shikra were explored as different-family backends, but their generations were not reliable enough for final scoring. Extending the protocol to Qwen-VL or other VLMs would require a model-specific backend for visual-token extraction, uncertainty estimation, region alignment, and masking.

---

### 5.7 Future work

Future work should extend the region-wise epistemic masking protocol to additional VLM families such as Qwen-VL. This would test whether the model-dependent behavior observed between LLaVA-1.5-7B and LLaVA-1.5-13B also appears across different architectures.

A second direction is to evaluate more object categories and larger counterfactual datasets. This would show whether the strong background-region effect in LLaVA-1.5-7B is specific to the selected ROHE categories or reflects a broader hallucination mechanism.

A third direction is to use more detailed semantic regions. Instead of only removed, context, and background, future work could test uncertainty in co-occurring objects, scene regions, or object-specific masks.

Finally, future work could test different uncertainty thresholds and masking strengths, especially for LLaVA-1.5-13B. This may clarify whether the lack of effect in 13B is caused by model behavior or by a mismatch between the masking intervention and the model’s internal representation.

---

### 5.8 Final conclusion

This project shows that region-wise masking can reveal which parts of the visual representation influence object-presence decisions. In LLaVA-1.5-7B, global and background high-uncertainty masking reduce hallucination relative to no masking. The same intervention does not produce a reliable reduction in LLaVA-1.5-13B.

The additional controls qualify the 7B result. High-uncertainty masking clearly differs from matched low-uncertainty masking, showing that the uncertainty ranking contains useful information. However, matched random masking produces similar effects, so the reduction cannot be attributed uniquely to uncertainty-guided selection. The original-image sanity check further shows that global and background masking reduce correct recognition, revealing a false-negative cost.

The final conclusion is:

```text
High-uncertainty global and background tokens identify visually influential parts of LLaVA-1.5-7B's representation. Suppressing them can reduce hallucination, but the effect is not uniquely uncertainty-specific and comes with a measurable loss in correct recognition. The behavior is model-dependent and does not transfer directly to LLaVA-1.5-13B.
```

