---
title: "9 Lessons I Learned while Doing RL Post-Training for LLMs"
date: 2025-07-08
draft: false
---

I recently had the chance to experiment with post-training techniques for large language models, a space that has become central to making LLMs useful and controllable in real-world applications. I used [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300) from the open-source [`open-r1`](https://github.com/huggingface/open-r1) repository, fine-tuning [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the coding subset of the [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) dataset.

The experience was very illuminating, especially around stability, reward design, and evaluation. Training a model with pure RL is different from unsupervised and supervised deep learning. It took some trial and error, so to save you some time, here are my top takeaways from training a code generation model with GRPO.

---

## 1. Inputs and rewards are all you need (GRPO is label-free).
Reinforcement learning from human (or heuristic) feedback is not just a continuation of pretraining or supervised fine-tuning (SFT). It is _policy optimization_ and the model learns from comparisons, not labels. The training loop must be stable despite reward sparsity, exploration tradeoffs, and longer feedback cycles. 

---

## 2. Instability can happen quickly and unexpectedly.
Some of the failures I saw such as sudden spikes in gradient norms, excessively long completions, or language shifts were symptoms of instability driven by reward noise. When reward signals were inconsistent or sparse, the model made large, erratic updates that could trigger irreversible collapse.

---

## 3. Reward design is crucial for success.
In GRPO, there is no ground truth as your model is judged purely by the reward function. If the function is not directly aligned with the end goal, the model can learn to generate answers that get a high reward but are not useful. When I trained a model with only a code format reward, I found that it would produce a code block, but the actual code would often fail to execute. Adding a correctness reward was key. Not only would this ensure the code executed, but also that it provided results in the desired format.

---

## 4. More generations + larger batch size = more learning signal.
In GRPO, [`num_generations`](https://github.com/huggingface/trl/blob/15ff54790b42297d2cf569fba6d7dd44c1c269e3/trl/trainer/grpo_config.py#L53) defines how many completions the model produces per prompt. I found that increasing the number of generations, combined with gradient accumulation, produced more stable learning and better reward comparisons. But there’s a tradeoff: too few generations lead to noisy learning, while too many slow down training significantly.

---

## 5. Use `num_iterations` to stabilize training and improve efficiency.
The [`num_iterations`](https://github.com/huggingface/trl/blob/15ff54790b42297d2cf569fba6d7dd44c1c269e3/trl/trainer/grpo_config.py#L135) hyperparameter determines how many times each set of generated completions is reused for policy updates. Increasing it helped smooth out training, reduced gradient noise, and made better use of GPU compute. More updates per sample means greater training stability and faster progress, without needing to generate more data.

---

## 6. Dr. GRPO can enhance learning.
[Dr. GRPO](https://arxiv.org/abs/2503.20783) improves on the vanilla GRPO approach by increasing token efficiency. It drops the length normalization term, which essentially prevents the model from generating progressively longer incorrect responses. It also drops the KL divergence penalty, which is often unnecessary when using a verifiable reward (unlike in RLHF, where distributional shift is a bigger concern). In my experiments, removing the KL term not only simplified the objective but also reduced memory and compute overhead, leading to faster and more stable training.

---

## 7. Logging completions is a must.
Quantitative rewards don’t tell the whole story. I caught many issues with training such as super long completions or language switching thanks to inspecting samples during training. High reward scores don’t always mean high-quality responses. Logging even a few completions per step is essential for catching issues early.

---

## 8. Regularization for diversity helps the model learn.
While GRPO is inherently more stable than online RL, it can still converge to repetitive or safe behaviors. Even with multiple completions per prompt, I noticed that the model often produced very similar outputs. That can limit learning. Increasing diversity within generation groups (e.g., with temperature or nucleus sampling) can help the model explore new solution strategies.

---

## 9. Start with a strong SFT baseline when you can.
Post-training a base model from scratch is challenging, especially for complex, multi-turn tasks like those in Mixture-of-Thoughts. A base model may struggle to generate useful completions early on, making it less likely to receive any reward signal at all. This can stall learning or lead to instability. In contrast, a supervised fine-tuned (SFT) model starts from more relevant responses, giving GRPO a much better foundation to refine and align behavior effectively.

---