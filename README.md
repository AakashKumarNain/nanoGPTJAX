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

**Note:** Only the dataloader was written with the help of code agents. It is not optimal, and I plan to replace it with my own implementation. <br>

## Tasks

- [x] Minimal abstraction for defining layers and models
- [x] Pretrain a GPT-2-like model on FineWeb 10B tokens
- [x] Inference
- [ ] Speculative Decoding
- [ ] Add tricks from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo to push the convergence
- [ ] Leaderboard to track run time for convergence wrt tricks
- [ ] Supervised fine-tuning on a dataset
- [ ] Reinforcement learning on a dataset
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

# 4. Prepare the dataset
```
# Download the dataset
python download_fineweb_tokens.py

# Preprocess the dataset

# Assuming fineweb10B is the directory containing the downloaded dataset, preprocess
# the dataset and build the token indices as follows:
python fineweb_dataloader.py --fineweb10b_path=../fineweb10B --save_train_idx_path=./train.npz --save_val_idx_path="./val.npz"
```

5. Train the model
```
# Pass the data dir, and indices path we created above in the config file. 
# Change the hparams in the file if you want.
# TODO: Streamline all params via config

python nanogpt/train.py
```

6. Run inference by providing the checkpoint path
```
# Change this in the config file
load_ckpt_path = /home/.../params  # absolute path only

# Run the inference code
python nanogpt/inference.py
```


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