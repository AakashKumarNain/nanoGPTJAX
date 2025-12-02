import dataclasses
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.sharding import PartitionSpec as P, NamedSharding


AxisName = str | Tuple[str, ...] | None
Axes = Tuple[AxisName, ...]


def jax_pytree_struct(cls):
    """
    A decorator that registers a dataclass as a JAX PyTree, automatically
    detecting static fields from metadata.

    Fields marked with `dataclasses.field(metadata={'static': True})` are
    treated as meta_fields (non-trainable), and all other fields are
    treated as data_fields (trainable).
    """
    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)

    # 1. Get all fields that are part of the constructor (__init__)
    all_fields = tuple(f for f in dataclasses.fields(cls) if f.init)

    # 2. Partition the field names into meta and data
    meta_fields = tuple(f.name for f in all_fields if f.metadata.get("static", False))
    data_fields = tuple(
        f.name for f in all_fields if not f.metadata.get("static", False)
    )
    return jtu.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


def istype(x, cls):
    return (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)


def is_param_spec(x):
    return istype(x, ParamSpec)


def logical_to_physical(logical: Axes, rules) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    flat_axes = jtu.tree_leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {spec}"
        )
    return P(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions (i.e. the logical shape of an array)."""
    assert mesh is not None
    return NamedSharding(mesh, logical_to_physical(logical, rules))


def get_partition_spec_from_layers(tree):
    """Extract PartitionSpec tree from parameters' existing sharding information."""

    def extract_spec(x):
        if x is None:
            return None
        elif hasattr(x, "sharding") and hasattr(x.sharding, "spec"):
            return x.sharding.spec
        elif hasattr(x, "shape"):
            return P()
        else:
            return None

    return jtu.tree_map(extract_spec, tree, is_leaf=lambda x: x is None)


def layer_repr(obj, max_width: int = 80, _indent: int = 0) -> str:
    """Pretty repr for layers"""
    cls_name = obj.__class__.__name__
    indent_str = " " * _indent
    child_indent = " " * (_indent + 4)

    def arr_repr(x):
        if x is None:
            return "None"
        if isinstance(x, jax.Array):
            return f"{x.dtype.name}[{','.join(map(str, x.shape))}]"
        if hasattr(x, "__dict__"):  # nested custom object
            return layer_repr(x, max_width=max_width, _indent=_indent + 4)
        return repr(x)

    parts = [f"{k}={arr_repr(v)}" for k, v in obj.__dict__.items()]
    one_line = f"{cls_name}(" + ", ".join(parts) + ")"

    if len(one_line) <= max_width and "\n" not in one_line:
        return one_line
    else:
        inner = ",\n".join(f"{child_indent}{p}" for p in parts)
        return f"{cls_name}(\n{inner}\n{indent_str})"


def _initialize_parameter_leaves(key, specs, shardings):
    # specs and shardings are already flattened tuples
    keys = jax.random.split(key, len(specs))

    # Init one leaf at a time instead of a big jitted graph. The compile time would go crazy.
    # TODO: There may be some gotchas here, but I am not noticing anything weird for now. Keep an eye on it!
    def init_one(k, spec, sharding):
        return jax.jit(spec.initializer, out_shardings=sharding, static_argnums=(1, 2))(k, spec.shape, spec.dtype)
    
    return tuple(init_one(k, s, sh) for k, s, sh in zip(keys, specs, shardings))


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    shape: Tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    logical_axes: Axes = dataclasses.field(metadata=dict(static=True))
    dtype: jnp.dtype = dataclasses.field(default=jnp.float32)
    initializer: Callable | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )


class ParamInitializer:
    """A base class that provides a factory method (`init`) for initializing
    a PyTree of parameters based on their specifications.
    """

    @classmethod
    def param_specs(cls, *args, **kwargs):
        """
        Defines the specifications (ParamSpec) for all parameters in the PyTree.
        This method must be implemented by any subclass.
        """
        raise NotImplementedError

    @classmethod
    def shardings(cls, mesh, rules, *args, **kwargs):
        """Defines the shardings parameters in a PyTree."""

        # Get the PyTree of parameter specifications.
        specs = cls.param_specs(*args, **kwargs)
        return jtu.tree_map(
            lambda spec: logical_to_sharding(spec.logical_axes, mesh, rules),
            specs,
            is_leaf=is_param_spec,
        )

    @classmethod
    def _init_fn(cls, key, mesh, rules, *args, **kwargs):
        """
        Initializes the actual JAX arrays for all parameters.

        This method first calls `param_specs` to get the abstract layout of
        the parameters, then uses that information to generate and shard the
        concrete arrays.
        """

        # Get the PyTree of parameter specifications.
        specs = cls.param_specs(*args, **kwargs)

        # Create a parallel PyTree of sharding objects from the specs.
        shardings = jtu.tree_map(
            lambda spec: logical_to_sharding(spec.logical_axes, mesh, rules),
            specs,
            is_leaf=is_param_spec,
        )

        # Flatten both the spec and sharding PyTrees to get ordered lists of leaves.
        spec_leaves, spec_treedef = jtu.tree_flatten(specs, is_leaf=is_param_spec)
        shardings_leaves = jtu.tree_leaves(shardings, is_leaf=is_param_spec)

        # Call the external JIT-compiled function to initialize arrays.
        initialized_leaves = _initialize_parameter_leaves(
            key, tuple(spec_leaves), tuple(shardings_leaves)
        )

        # Reconstruct the original PyTree structure with the new initialized arrays.
        return jtu.tree_unflatten(spec_treedef, initialized_leaves)