# nanoGPTJAX

This project is inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat), with one major difference: here we build everything from scratch in **pure JAX (on both GPUs and TPUs)**, avoiding higher-level third-party model/training libraries. This is not meant to start another PyTorch vs. JAX debate—I use both on a daily basis, and both are good in their own right. There are a few reasons I keep using JAX:

1. I am not a fan of the OOP paradigm for deep learning. It is often nice to have, but not required.
2. Nothing comes close to distributed training in JAX. The mental model is simple and fits well with the philosophy of having control over every design aspect of a training run.
3. Reproducibility is a first-class citizen in JAX.
4. I like having fine-grained control over implementation and performance details without fighting framework abstractions.

It is recommended to read the [design philosophy](notes/design.md) to better understand how this works under the hood. <br>

## Architecture

We follow the standard Transformer architecture, with the following choices:

- Grouped Query Attention (GQA)
- No weight tying
- RoPE
- QK-Norm
- Logits soft-capping
- ReLU-squared activations in the MLP (will be replaced with SwiGLU soon)
- RMSNorm without learnable parameters
- (Optional) weight decay

The models here can be trained with both `AdamW` and `Muon` optimizers (via Optax). You can use any sharding strategy depending on the size of the model. We use the cached, tokenized **FineWeb10B** dataset as in [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).


## Tasks

- [x] Minimal abstraction for defining layers and models
- [x] Pretrain a GPT-2-like model on FineWeb 10B tokens
- [x] Inference
- [x] Cautious Weight Decay
- [x] Mid-training
- [x] Supervised fine-tuning on a dataset
- [ ] Reinforcement learning on a dataset
- [ ] Speculative Decoding
- [ ] Leaderboard to track run time for convergence wrt tricks
- [ ] Quantization
- [ ] MoE example

<br>

## Getting Started

1. Install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

2. Create a `venv`
```
uv venv --python 3.12
source .venv/bin/activate
```

3. Install dependencies
```
uv sync

# If you are running the code on GPUs, run this instead:
uv sync --all-extras
```

4. Prepare the dataset
```
# Download the dataset
python download_fineweb_tokens.py
```

5. Train the model
```
# Pass the data dir path in the config file located at `nanochat/config.py`
# Change the hparams in the file if you want.
python nanogpt/train.py
```

6. Run inference by providing the checkpoint path
```
# Change this in the config file
load_ckpt_path = /home/.../params  # absolute path only

# Run the inference code
python nanogpt/inference.py
```
<br>

## Results

After pretraining the model on first 30 shards of Fineweb10B tokens, here are some sample outputs from the mode:

```
prompts = [
        "<|endoftext|>Did you notice that this world",
        "<|endoftext|>Hello World! My dear",
        "<|endoftext|>Some say we are tired far",
        "<|endoftext|>Hear that?",
    ]
---

Completions:

<|endoftext|>Did you notice that this world is filled with so many details and characters that children deserve? Does it feel like money and art hasn't stopped up the world of fantasy that it is obsessed with? Perhaps a cat ate ice cream while I screamed till the last minute. Or the things

<|endoftext|>Hello World! My dear friend Wendy who lives in Helsinki and it has been a great experience to have met some amazing new people at such amazing places! Many of these photos have not been taken since I arrived in Helsinki. While my colleagues quite frequently comment on how wonderful the locations

<|endoftext|>Some say we are tired far too soon. Unfortunately, a few weeks ago, my friend Julie Dossakis had her wedding day from a newly beautiful downtown Hotel with her gorgeous smile at the thought of the occasion. She said that “I’m going to just dress

<|endoftext|>Hear that? I run across while I sit on a station wagon in The Village Voice and the car stays silent and beeps. What is that like? Find out on one of our monthly Community Radio shows.\nTalk about the power of truth! Podcast Series\n
```
<br>

## Benchmarking 

As of now without using any tricks, the training loss converges in around ~16 minutes on `4 X H100` machine. The dataloader is not optimal,
we have not included gradient accumulation, we have not used any tricks to improve the convergence. Still, 16 minutes is neither bad nor great.
I am sure we can do it in under 5-8 minutes soon without using many tricks. 🤞

We will soon add a table that will list down the runs with changes in code and performance improvements.

## Contributing

Contributions are welcome. Apart from bug fixes, the task list above is a good start for contributions.

- **Before you start:** Please open an issue to discuss significant changes (new features, refactors, training pipeline changes).
- **Branching:** Create a feature branch from `main` (e.g., `feat/<name>` or `fix/<name>`).
- **Testing:** If you add or change functionality, include minimal tests or a small reproducible script to validate the change.
- **Pull requests:** In your PR description, include (1) what changed, (2) why it changed, and (3) how to reproduce/verify. <br>

We use `ruff` for linting and formatting. Please run `ruff check nanogpt/*.py` and `ruff format nanogpt/*.py` before sending a PR. We will streamline this process via `pre-commit` soon.  

## References

This work would not have been possible without these existing resources:

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nanochat](https://github.com/karpathy/nanochat)
- [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples/tree/main)
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)
- [JAX Tutorials](https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-4-jax-and-devicearray)