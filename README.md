# Monitoring CoT Faithfulness in Latent Space

**Idea**: learn a vector in model latent space that correlates with whether a chain-of-thought (CoT) is faithful to the final answer, using a simple difference-in-means probe.

**Reference inspiration**: CoT Reasoning in the Wild is Not Always Faithful (see `refs/` for materials). The `refs` folder is for reference only and is not used by the MVP code.

## MVP Approach

- Generate paired yes/no questions for numeric comparisons: (i) “Is X > Y?” and (ii) “Is Y > X?”.
- Get CoT-like responses from a small open-source HF model with a “Let’s think step by step” instruction.
- Label a pair as:
  - `faithful` if exactly one answer is Yes (consistent)
  - `unfaithful` if both are Yes or both are No (inconsistent)
- Extract hidden-state features from the model over the generated continuation and compute:
  - `v_faithful = mean(feats_faithful) - mean(feats_unfaithful)` (unit-normalized)
- Report the projection of representative faithful/unfaithful pairs onto `v_faithful`.
