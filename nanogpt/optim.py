import jax
import optax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey, SequenceKey, DictKey


def build_optimizer(
    params,
    *,
    d_model: int,
    other_peak_lr: float,
    other_min_lr: float,
    total_train_steps: int,
    warmup_steps: int = 30,
    b1: float = 0.9,
    b2: float = 0.95,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    weight_decay: float = 0.0,
    cautious_weight_decay: float = 0.01,
    use_muon=True,
):
    # nanochat's width scaling for AdamW groups: (d_model / 768) ** -0.5
    dmodel_lr_scale = (d_model / 768.0) ** -0.5

    # use width scaling for embed/lm_head; do not tie to other_peak_lr
    emb_lr = embedding_lr * dmodel_lr_scale
    unemb_lr = unembedding_lr * dmodel_lr_scale

    if use_muon:
        print("Using Muon Optimizer!")
        other_peak_lr = max(other_peak_lr, 2e-2)

    schedules = {
        "embed": optax.constant_schedule(emb_lr),
        "lm_head": optax.constant_schedule(unemb_lr),
        "other": optax.warmup_cosine_decay_schedule(
            init_value=other_min_lr,
            peak_value=other_peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_train_steps - warmup_steps),
            end_value=other_min_lr,
        ),
    }

    def _path_names(path):
        out = []
        for k in path:
            if isinstance(k, GetAttrKey):
                out.append(k.name)
            elif isinstance(k, SequenceKey):
                out.append(str(k.idx))
            elif isinstance(k, DictKey):
                out.append(str(k.key))
            else:
                out.append(str(k))
        return out

    def label_fn(path, leaf):
        # Top-level fields in GPT pytree: embed, blocks, lm_head
        names = _path_names(path)
        top = names[0] if names else ""
        if top == "embed":
            return "embed"
        if top == "lm_head":
            return "lm_head"
        return "other"

    def make_weight_dim_nums(p):
        def choose(x):
            s = getattr(x, "shape", None)
            if s is None:
                return None
            if len(s) == 2:
                return optax.contrib.MuonDimensionNumbers((0,), (1,))
            if len(s) == 3:
                # wo: (heads, head_dim, d_model)
                if s[-1] == d_model:
                    return optax.contrib.MuonDimensionNumbers((1,), (2,))
                # wq/wk/wv: batch=heads
                return optax.contrib.MuonDimensionNumbers((0,), (2,))
            return None

        return jax.tree_util.tree_map(choose, p)

    def weight_decay_mask_fn(p):
        def keep(x):
            s = getattr(x, "shape", None)
            return s is not None and len(s) >= 2

        return jax.tree_util.tree_map(keep, p)

    def cautious_decay(schedule, wd):
        def init_fn(params):
            return {"count": jnp.array(0, dtype=jnp.int32)}

        def update_fn(updates, state, params=None):
            step = state["count"]
            scale = schedule(step)
            if params is None:
                return updates, {"count": step + 1}

            def apply_updates(update, param):
                if param is None:
                    return update
                s = getattr(param, "shape", None)
                eligible_for_update = (s is not None) and (len(s) >= 2)
                if not eligible_for_update:
                    return update
                # Cautious: only decay when update and param are aligned (same sign)
                decay = jnp.where(
                    (update * param) >= 0,
                    param.astype(update.dtype),
                    jnp.zeros_like(update),
                )
                # Subtract to apply weight decay (moves parameters toward zero)
                return update - (scale * wd) * decay

            new_updates = jax.tree_util.tree_map(apply_updates, updates, params)
            return new_updates, {"count": step + 1}

        return optax.GradientTransformation(init_fn, update_fn)

    param_labels = jax.tree_util.tree_map_with_path(label_fn, params)
    muon_weight_dim_nums = make_weight_dim_nums(params)
    muon_wd_mask = weight_decay_mask_fn(params)

    def make_adamw(lr_schedule, weight_decay=0.0):
        return optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            eps=1e-10,  # for better stability like nanochat/modded-nanogpt
            weight_decay=weight_decay,
            mu_dtype=jnp.float32,
        )

    def make_muon(lr_schedule, weight_decay=0.0):
        return optax.contrib.muon(
            learning_rate=lr_schedule,
            ns_coeffs=(3.4445, -4.775, 2.0315),
            ns_steps=5,
            beta=b2,
            eps=1e-8,
            weight_decay=0.0,
            weight_decay_mask=muon_wd_mask,
            mu_dtype=jnp.float32,
            nesterov=True,
            adaptive=False,
            adam_b1=b1,
            adam_b2=b2,
            adam_eps_root=0.0,
            adam_weight_decay=weight_decay,
            muon_weight_dimension_numbers=muon_weight_dim_nums,
            consistent_rms=None,
        )

    tx = optax.multi_transform(
        {
            "embed": make_adamw(schedules["embed"]),
            "lm_head": make_adamw(schedules["lm_head"]),
            "other": (
                optax.chain(
                    make_muon(schedules["other"], weight_decay=weight_decay),
                    cautious_decay(schedules["other"], cautious_weight_decay),
                )
                if use_muon
                else optax.chain(
                    make_adamw(schedules["other"], weight_decay=weight_decay),
                    cautious_decay(schedules["other"], cautious_weight_decay),
                )
            ),
        },
        param_labels,
    )

    return tx