"""
Implementation of BOP in optax
Reference: https://github.com/lukaszlew/poke/blob/main/adabop.py#L338-L360
"""
from optax._src import utils
from optax._src import base
from typing import Any, NamedTuple, Optional
import jax
import jax.numpy as jnp
import flax

class EmaState(NamedTuple):
  """Holds an aggregation of past updates."""
  ema: base.Params

def bop(
    tau: float = 1.0e-4,
    gamma: float = 10e-4,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Compute a trace of past updates.

  Note: `trace` and `ema` have very similar but distinct updates;
  `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
  Both are frequently found in the optimisation literature.

  Args:
    gamma: the decay rate for the ema of past updates.
    tau: the threshold for comparing the absolute value of ema.
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)


  def init_fn(params):
    return EmaState(
        ema=jax.tree_map(
            lambda m: jnp.zeros_like(m, dtype=accumulator_dtype), params))


  def update_fn(updates, state, params=None):
    """
    updates as an input argument is a pytree.
    Its values are the raw gradients.
    """
    assert params is not None, "BOP needs params in update_fn but found None"
    f = lambda g, m: (1 - gamma) * m + gamma * g
    new_ema = jax.tree_map(f, updates, state.ema)
    bop_update_fn = lambda m, w: jnp.where(jnp.abs(m)>tau, jnp.sign(-m)-w, 0)
    updates = jax.tree_map(bop_update_fn, new_ema, params)
    del params
    new_ema = utils.cast_tree(new_ema, accumulator_dtype)
    return updates, EmaState(ema=new_ema)


  return base.GradientTransformation(init_fn, update_fn)



"""
Implementation of adaptive BOP optimizer
Reference: https://github.com/lukaszlew/poke/blob/main/adabop.py#L362-L378
"""

class AdabopState(NamedTuple):
  """Holds an aggregation of past updates."""
  ema: base.Params
  ema_var: base.Params

def adabop(
    tau: float = 1e-4,
    gamma1: float = 10e-4,
    gamma2: float = 10e-4,
    std_prior: float = 0.0,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Compute a trace of past updates.

  Note: `trace` and `ema` have very similar but distinct updates;
  `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
  Both are frequently found in the optimisation literature.

  Args:
    gamma: the decay rate for the ema of past updates.
    tau: the threshold for comparing the absolute value of ema.
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)


  def init_fn(params):
    return AdabopState(
        ema=jax.tree_map(
            lambda m: jnp.zeros_like(m, dtype=accumulator_dtype), params),
        ema_var=jax.tree_map(
            lambda m: jnp.ones_like(m, dtype=accumulator_dtype) * std_prior, params),
            )


  def update_fn(updates, state, params=None):
    """
    updates as an input argument is a pytree.
    Its values are the raw gradients.
    """
    assert params is not None, "AdaBOP needs params in update_fn but found None"
    f_ema = lambda g, m: (1 - gamma1) * m + gamma1 * g
    f_ema_var = lambda g, m, v: (1 - gamma2) * v + gamma2 * jnp.square((g-m))
    new_ema = jax.tree_map(f_ema, updates, state.ema)
    new_ema_var = jax.tree_map(f_ema_var, updates, new_ema, state.ema_var)
    adabop_update_fn = lambda m, v, w: jnp.where(jnp.abs(m) > tau*jnp.sqrt(v), jnp.sign(-m)-w, 0)
    updates = jax.tree_map(adabop_update_fn, new_ema, new_ema_var, params)
    del params
    new_ema = utils.cast_tree(new_ema, accumulator_dtype)
    new_ema_var = utils.cast_tree(new_ema_var, accumulator_dtype)
    return updates, AdabopState(ema=new_ema, ema_var=new_ema_var)


  return base.GradientTransformation(init_fn, update_fn)
