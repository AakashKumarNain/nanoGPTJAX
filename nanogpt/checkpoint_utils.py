import jax
import grain
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_sharding_for_checkpoint(x, mesh):
    """Obtain shardings for the leaves of the pytree."""
    if hasattr(x, "ndim") and x.ndim == 0:
        return NamedSharding(mesh, P())
    if isinstance(x, jax.Array) and hasattr(x, "sharding"):
        from jax.sharding import SingleDeviceSharding

        # Ensure small optimizer leaves (e.g., Muon scalars/vectors) are replicated,
        # not left on a single device, to match param shardings during train_step.
        if isinstance(x.sharding, SingleDeviceSharding):
            return NamedSharding(mesh, P())
        return x.sharding
    else:
        return NamedSharding(mesh, P())


def load_checkpoint(mngr, step, model, optim_state, mesh, ds_iter=None):
    """Load the weights, the optimizer state, and (optional) data iterator state
    from a given checkpoint.

    Args:
        mngr: Checkpoint manager instance.
        step: The step from which the state has to be restored.
        model: Pytree  containing params.
        optim_state: Current optimizer state.
        mesh: Current mesh where the model and optimizer state is alive.
        ds_iter (optional): The data iterator whose state is to be restored from
            this checkpoint. Defaults to `None`.

    Returns:
        Tuple of (Restored weights, restored optim_state, restored ds_iter)
    """

    params_item, params_transforms = model, None
    optim_item, optim_transforms = optim_state, None
    params_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        model,
    )
    optim_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        optim_state,
    )
    if ds_iter is not None:
        restore_items = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item=params_item,
                transforms=params_transforms,
                restore_args=params_restore_args,
            ),
            optim_state=ocp.args.PyTreeRestore(
                item=optim_item,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
            ds=grain.checkpoint.CheckpointRestore(ds_iter),
        )
        restored = mngr.restore(step, args=restore_items)
        print(
            f"Restoring model, optim_state, and data_iter from step {step} is complete!"
        )
        return restored.params, restored.optim_state, restored.ds
    else:
        restore_items = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item=params_item,
                transforms=params_transforms,
                restore_args=params_restore_args,
            ),
            optim_state=ocp.args.PyTreeRestore(
                item=optim_item,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
        )
        restored = mngr.restore(step, args=restore_items)
        print(f"Restoring model and optim_state from step {step} is complete!")
        return restored.params, restored.optim_state


def load_weights_from_checkpoint(path, sharding):
    """Load weights of the model from a given checkpoint.

    Args:
        path: Path to the saved pytree checkpointed during training.
        sharding: The sharding info obtained from the current state of the pytree.

    Returns:
        PyTree with weights loaded from the given checkpoint.
    """

    print(f"Restoring checkpoint from: {path}")
    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item, transforms=transforms, restore_args=restore_args
            ),
        )


def load_optim_state_from_checkpoint(path, optim_state, mesh):
    """Load optimizer state from a given checkpoint.

    Args:
        path: Path to the saved pytree checkpointed during training.
        optim_state: The current optimizer state.
        mesh: Mesh of the current optimizer state

    Returns:
        PyTree with optim state loaded from the given checkpoint.
    """

    print(f"Restoring optimizer state from checkpoint: {path}")
    ioptim_item, optim_transforms = optim_state, None
    optim_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        optim_state,
    )
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=optim_state,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
        )
