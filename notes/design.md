# 1. Is this another abstraction (hell)?

It is a well known fact that the JAX ecosystem is fragmented. There are different libraries and frameworks (based on JAX), each serving its own purpose. I’ve been involved in the development of some of these (Keras 3) and have contributed to others (Equinox, Orbax, etc.). Though Google has recently leaned more toward Flax, I’m not a big fan of the API (totally personal opinion; I wouldn’t have designed the API that way).

> [!NOTE]
> [Keras](https://keras.io/) and [Equinox](https://docs.kidger.site/equinox/) are good abstractions, and you should totally try them out.

Having said that, when working with JAX, I prefer keeping things minimal (in terms of abstraction) but having maximum control over the functionality. This abstraction provided in the current codebase is lightweight, yet highly scalable, efficient, easy to read.


# 2. Who is the target audience?

If you are someone who:
- enjoys minimal (low‑level) abstractions
- loves to write scalable code that can run on thousands of accelerators
- wants to write JAX for more than fun
- loves training LLMs and VLMs at scale

then you are at the right place. We will write every piece of the modern LLM stack with this abstraction from scratch.

# 3. The Mental Model

Though there are many valid design choices for building an abstraction, I do not like the idea of mixing OOP‑based design choices with purely functional paradigms. Broadly speaking, there is only a finite set of things we need to achieve while designing the abstraction. These include:

- We want to keep the “state” (model parameters, optimizer parameters, etc.) and the functionality separate. For example, we can use a class as a namespace for the parameters belonging to a layer, and then define a function that consumes that state, stays pure, and plays nicely with JAX transformations.
- We should be able to change the functionality as needed without changing how we store the state.
- It needs to be scalable and agnostic of the accelerator used.

> [!IMPORTANT]
> This design cleanly separates “what to create” (parameter specifications), “where to put it” (sharding), and “how to create it” (initialization). That separation is the core abstraction that lets you write simple layer definitions and then scale them across devices without changing layer code.

We only need two classes to achieve all of this, defined in the [`utils.py`](https://github.com/AakashKumarNain/nanoGPTJAX/blob/main/nanogpt/utils.py) file.

- **ParamSpec:** Defines the specification of JAX arrays we want to create. It stores the following abstract information about our arrays:
    - *shape*: shape of the array
    - *dtype*: data type of the array
    - *initializer*: initializer to be used for initializing the elements of the array
    - *logical_axes*: how the different logical axes of the array will be sharded
    ```python
    @dataclasses.dataclass(frozen=True)
    class ParamSpec:
        shape: Tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
        logical_axes: Axes = dataclasses.field(metadata=dict(static=True))
        dtype: jnp.dtype = dataclasses.field(default=jnp.float32)
        initializer: Callable | None = dataclasses.field(
            default=None, metadata=dict(static=True)
        )
    ```

- **ParamInitializer:** This is the base abstract class registered as a PyTree for our layers, and it’s responsible for materializing our arrays as per the defined `ParamSpec`. It contains three methods:
    - *param_specs*: defines the leaves of this PyTree. Each leaf is defined using `ParamSpec`.
    - *shardings*: defines the sharding of all the leaves (arrays) in our PyTree.
    - *_init_fn*: initializes the actual JAX arrays. It first calls `param_specs` to get the abstract layout of
      the parameters, then uses that information to generate and shard the concrete arrays.
    ```python
    class ParamInitializer:
    @classmethod
    def param_specs(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def shardings(cls, mesh, rules, *args, **kwargs):
        specs = cls.param_specs(*args, **kwargs)
        return jtu.tree_map(
            lambda spec: logical_to_sharding(spec.logical_axes, mesh, rules),
            specs,
            is_leaf=is_param_spec,
        )

    @classmethod
    def _init_fn(cls, key, mesh, rules, *args, **kwargs):
        specs = cls.param_specs(*args, **kwargs)
        shardings = jtu.tree_map(
            lambda spec: logical_to_sharding(spec.logical_axes, mesh, rules),
            specs,
            is_leaf=is_param_spec,
        )
        spec_leaves, spec_treedef = jtu.tree_flatten(specs, is_leaf=is_param_spec)
        shardings_leaves = jtu.tree_leaves(shardings, is_leaf=is_param_spec)
        initialized_leaves = _initialize_parameter_leaves(
            key, tuple(spec_leaves), tuple(shardings_leaves)
        )
        return jtu.tree_unflatten(spec_treedef, initialized_leaves)
    ```

That is all the abstraction we need to build everything from scratch. 😄


# 4. An Example

Let’s see this in action: how easy it is to build on top of this abstraction. How about a `Linear` layer? Here’s how you can define and build one. As stated earlier, we’re keeping the state and the consumption of that state separate, so we’ll also define a forward pass for this layer.
 

```python
@jax_pytree_struct
class Linear(ParamInitializer):
    in_features: int = dataclasses.field(metadata=dict(static=True))
    out_features: int = dataclasses.field(metadata=dict(static=True))
    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    use_bias: bool = dataclasses.field(default=False, metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.in_features, cfg.out_features),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer or linear_init(cfg.in_features, cfg.out_features),
        )
        if cfg.use_bias:
            bias = ParamSpec(
                shape=(cfg.out_features,),
                dtype=cfg.dtype,
                logical_axes=cfg.bias_logical_axes,
                initializer=cfg.bias_initializer or jax.nn.initializers.zeros,
            )
        else:
            bias = None
        return Linear(
            weight=weight,
            bias=bias,
            in_features=cfg.in_features,
            out_features=cfg.out_features,
            use_bias=cfg.use_bias,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return cls._init_fn(key, mesh, rules, cfg)


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 512
    use_bias: bool = False
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")

    def __post_init__(self):
        self.weight_initializer = ...

    
# Forward pass for linear layer
@jax.jit
def linear_forward(params, x):
    out = jnp.einsum("...d, dv-> ...v", x, params.weight)
    if params.bias is not None:
        return out + params.bias
    else:
        return out

# Get the params
params = Linear.init(...)
x = jnp.asarray(np.random.rand(32, 768)).astype(jnp.bfloat16)
out = linear_forward(params, x)

```

You can modify the forward pass to suit your workflow without ever touching the state. We can build all other layers in a similar way. 
