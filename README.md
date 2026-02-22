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
- [x] Chunked Cross Entropy
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
temperature = 0.8
top_k = 100
max_new_tokens = 50
prompts = [
        "<|endoftext|>Did you notice that this world",
        "<|endoftext|>Hello World! My dear",
        "<|endoftext|>Some say we are tired far",
        "<|endoftext|>The capital of France",
    ]
---

Completions:

<|endoftext|>Did you notice that this world is filled with so many people and so many ideas?
Well, I’m going to tell you about a few of them that I personally love.
One is an article written for a few local blogs, one is for a national magazine

<|endoftext|>Hello World! My dear friend,
This is my first post in a while. I’m glad that I have found such amazing blog because this is really a place I would visit.
If you haven’t read my past post, you’ll

<|endoftext|>Some say we are tired far too easily. It’s true that we often feel tired and defeated when we’re not getting the right treatment.
But the good news is that you can get the right treatment and get the right life.
You don’

<|endoftext|>The capital of France, Paris, is the birthplace of many art treasures. The city has a rich history of art and architecture, and a beautiful
city park is a place to relax and enjoy the peace. The city also features a number of historical buildings and museums, which
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

We use `ruff` for linting and formatting. You can either manually run `ruff check nanogpt/*.py` and `ruff format nanogpt/*.py` before sending a PR or you can
install `pe-commit` and will do the job for you. To install pre-commit, use the following command:

```
uv tool install pre-commit --with pre-commit-uv
```

## References

This work would not have been possible without these existing resources:

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nanochat](https://github.com/karpathy/nanochat)
- [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples/tree/main)
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)
- [JAX Tutorials](https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-4-jax-and-devicearray)