# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CAD-DQN agent class."""

# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping, Text, Optional
import warnings
from absl import logging
import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import optax
import rlax

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

# Batch variant of cad_q_learning with fixed tau input across batch.


Array = chex.Array
Numeric = chex.Numeric








def categorical_l2_project(
    z_p: Array,
    probs: Array,
    z_q: Array
) -> Array:
  """Projects a categorical distribution (z_p, p) onto a different support z_q.

  The projection step minimizes an L2-metric over the cumulative distribution
  functions (CDFs) of the source and target distributions.

  Let kq be len(z_q) and kp be len(z_p). This projection works for any
  support z_q, in particular kq need not be equal to kp.

  See "A Distributional Perspective on RL" by Bellemare et al.
  (https://arxiv.org/abs/1707.06887).

  Args:
    z_p: support of distribution p.
    probs: probability values.
    z_q: support to project distribution (z_p, probs) onto.

  Returns:
    Projection of (z_p, p) onto support z_q under Cramer distance.
  """
  chex.assert_rank([z_p, probs, z_q], 1)
  chex.assert_type([z_p, probs, z_q], float)

  kp = z_p.shape[0]
  kq = z_q.shape[0]

  # Construct helper arrays from z_q.
  d_pos = jnp.roll(z_q, shift=-1)
  d_neg = jnp.roll(z_q, shift=1)

  # Clip z_p to be in new support range (vmin, vmax).
  z_p = jnp.clip(z_p, z_q[0], z_q[-1])[None, :]
  assert z_p.shape == (1, kp)

  # Get the distance between atom values in support.
  d_pos = (d_pos - z_q)[:, None]  # z_q[i+1] - z_q[i]
  d_neg = (z_q - d_neg)[:, None]  # z_q[i] - z_q[i-1]
  z_q = z_q[:, None]
  assert z_q.shape == (kq, 1)

  # Ensure that we do not divide by zero, in case of atoms of identical value.
  d_neg = jnp.where(d_neg > 0, 1. / d_neg, jnp.zeros_like(d_neg))
  d_pos = jnp.where(d_pos > 0, 1. / d_pos, jnp.zeros_like(d_pos))

  delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]
  d_sign = (delta_qp >= 0.).astype(probs.dtype)
  assert delta_qp.shape == (kq, kp)
  assert d_sign.shape == (kq, kp)

  # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
  delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
  probs = probs[None, :]
  assert delta_hat.shape == (kq, kp)
  assert probs.shape == (1, kp)

  return jnp.sum(jnp.clip(1. - delta_hat, 0., 1.) * probs, axis=-1)








def categorical_cross_entropy(
    labels: Array,
    logits: Array
) -> Array:
  """Computes the softmax cross entropy between sets of logits and labels.

  See "Deep Learning" by Goodfellow et al.
  (http://www.deeplearningbook.org/contents/prob.html). The computation is
  equivalent to:

                  sum_i (labels_i * log_softmax(logits_i))

  Args:
    labels: a valid probability distribution (non-negative, sum to 1).
    logits: unnormalized log probabilities.

  Returns:
    a scalar loss.
  """
  warnings.warn(
      "Rlax categorical_cross_entropy will be deprecated. "
      "Please use distrax.Categorical.cross_entropy instead.",
      PendingDeprecationWarning, stacklevel=2
  )
  return distrax.Categorical(probs=labels).cross_entropy(
      distrax.Categorical(logits=logits))











@jax.custom_gradient
def clip_gradient(x, gradient_min: float, gradient_max: float):
  """Identity but the gradient in the backward pass is clipped.

  See "Human-level control through deep reinforcement learning" by Mnih et al,
  (https://www.nature.com/articles/nature14236)

  Note `grad(0.5 * clip_gradient(x)**2)` is equivalent to `grad(huber_loss(x))`.

  Note: x cannot be properly annotated because pytype does not support recursive
  types; we would otherwise use the chex.ArrayTree pytype annotation here. Most
  often x will be a single array of arbitrary shape, but the implementation
  supports pytrees.

  Args:
    x: a pytree of arbitrary shape.
    gradient_min: min elementwise size of the gradient.
    gradient_max: max elementwise size of the gradient.

  Returns:
    a vector of same shape of `x`.
  """
  chex.assert_type(x, float)

  def _compute_gradient(g):
    return (tree_map(lambda g: jnp.clip(g, gradient_min, gradient_max),
                     g), 0., 0.)

  return x, _compute_gradient










def l2_loss(predictions: Array,
            targets: Optional[Array] = None) -> Array:
  """Caculates the L2 loss of predictions wrt targets.

  If targets are not provided this function acts as an L2-regularizer for preds.

  Note: the 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop, but not "The Elements of Statistical Learning" by Tibshirani.

  Args:
    predictions: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions.

  Returns:
    a vector of same shape of `predictions`.
  """
  if targets is None:
    targets = jnp.zeros_like(predictions)
  chex.assert_type([predictions, targets], float)
  return 0.5 * (predictions - targets)**2










def cad_q_learning(
    dist_q_tm1: Array,
    q_atoms_tm1: Array,
    q_logits_tm1: Array,
    a_tm1: Numeric,
    r_t: Numeric,
    discount_t: Numeric,
    dist_q_t_selector: Array,
    dist_q_t: Array,
    q_atoms_target_tm1: Array,
    q_logits_target_tm1: Array,
    grad_error_bound: Numeric,
    stop_target_gradients: bool = True,
) -> Numeric:
  """Implements Q-learning for avar-valued Q distributions.

  See "xxxx" by
  Achab et al. (https://arxiv.org/abs/xxxx).

  Args:
    dist_q_tm1: Q distribution at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    dist_q_t_selector: Q distribution at time t for selecting greedy action in
      target policy. This is separate from dist_q_t as in Double Q-Learning, but
      can be computed with the target network and a separate set of samples.
    dist_q_t: target Q distribution at time t.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    CAD-Q-learning temporal difference error.
  """
  chex.assert_rank([
      dist_q_tm1, q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t, q_atoms_target_tm1, q_logits_target_tm1
  ], [2, 1, 2, 0, 0, 0, 2, 2, 1, 2])
  chex.assert_type([
      dist_q_tm1, q_atoms_tm1, q_logits_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t, q_atoms_target_tm1, q_logits_target_tm1
  ], [float, float, float, int, float, float, float, float, float, float])

  # Only update the taken actions.
  dist_qa_tm1 = dist_q_tm1[:, a_tm1]
  #qa_logits_target_tm1 = q_logits_target_tm1[:, a_tm1]
  qa_logits_target_tm1 = q_logits_tm1[a_tm1]  # use online net as target

  # Select target action according to greedy policy w.r.t. dist_q_t_selector.
  q_t_selector = jnp.mean(dist_q_t_selector, axis=0)
  a_t = jnp.argmax(q_t_selector)
  dist_qa_t = dist_q_t[:, a_t]

  # ADDED BY MASTANE
  target_tm1 = r_t + discount_t * jnp.mean(dist_qa_t, keepdims=True)

  # Project using the Cramer distance and maybe stop gradient flow to targets.
  categ_target = categorical_l2_project(target_tm1, jnp.array([1.0]), q_atoms_tm1)
  categ_target = jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(categ_target),
                          categ_target)

  # Compute loss (i.e. temporal difference error).
  logit_qa_tm1 = q_logits_tm1[a_tm1]
  c_losses =  categorical_cross_entropy(
      labels=categ_target, logits=logit_qa_tm1)



  num_avars = dist_qa_tm1.shape[-1]
  # take argsort on atoms, then reorder atoms and probabilities
  probas = jax.nn.softmax(qa_logits_target_tm1)
  #atoms_target_tm1 = q_atoms_target_tm1
  atoms_target_tm1 = q_atoms_tm1  # use online net as target
  #sigma = jnp.argsort( atoms_target_tm1 )  # categorical support already sorted
  #atoms_target_tm1 = atoms_target_tm1[sigma]
  #probas = probas[sigma]
  # avar intervals
  i_window = jnp.arange( 1, num_avars + 1 ) / jnp.float32( num_avars )  # avar integration segments
  j_right = jnp.cumsum(probas)  # cumulative probabilities of the N+1 atoms
  j_left = j_right - probas
  i_window = jnp.expand_dims( i_window, axis=1 )
  j_right = jnp.expand_dims( j_right, axis=0 )
  j_left = jnp.expand_dims( j_left, axis=0 )
  # compute avars
  minij = jnp.minimum( i_window, j_right )
  maxij = jnp.maximum( i_window - 1.0/ jnp.float32( num_avars ) , j_left )
  lengths_inter = jnp.maximum( 0.0, minij - maxij )  # matrix of lengths of intersections of intervals [(i-1)/N, i/N] with [(j-1)/(N+1), j/(N+1)]
  dist_target = jnp.float32( num_avars ) * jnp.dot(lengths_inter, atoms_target_tm1)

  # Compute target, do not backpropagate into it.
  dist_target = jax.lax.select(stop_target_gradients,
                               jax.lax.stop_gradient(dist_target), dist_target)

  td_errors = dist_target - dist_qa_tm1
  td_errors = clip_gradient(td_errors, -grad_error_bound,
                                 grad_error_bound)
  a_losses = l2_loss(td_errors)
  a_losses = jnp.mean(a_losses, axis=-1)

  """
  losses = jnp.mean(losses, axis=-1)
  """


  return c_losses + a_losses

_batch_cad_q_learning = jax.vmap(cad_q_learning, in_axes=(0,None,0,0,0,0,0,0,None,0,None))

class CadDqn(parts.Agent):
  """Categorical Atomic Distributional Deep Q-Network agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      support: jnp.ndarray,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      grad_error_bound: float,
      rng_key: parts.PRNGKey,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._grad_error_bound = grad_error_bound
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}


    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      # Compute Q value distributions.
      _, online_key, target_key = jax.random.split(rng_key, 3)
      dist_q_tm1 = network.apply(online_params, online_key,
                                 transitions.s_tm1).q_dist
      #dist_q_target_t = network.apply(target_params, target_key,
      #                              transitions.s_t).q_dist
      dist_q_target_t = network.apply(online_params, online_key,  # use online net as target
                                      transitions.s_t).q_dist
      logits_q_tm1 = network.apply(online_params, online_key,
                                   transitions.s_tm1).q_logits
      logits_target_q_tm1 = network.apply(target_params, target_key,
                                        transitions.s_tm1).q_logits
      losses = _batch_cad_q_learning(
          dist_q_tm1,
          support,
          logits_q_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          dist_q_target_t,  # No double Q-learning here.
          dist_q_target_t,
          support,
          logits_target_q_tm1,
          self._grad_error_bound
      )
      chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = distrax.EpsilonGreedy(q_t,
                                  exploration_epsilon).sample(seed=policy_key)
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self._select_action(self._rng_key,
                                                  self._online_params, s_t,
                                                  self.exploration_epsilon)
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values())
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
