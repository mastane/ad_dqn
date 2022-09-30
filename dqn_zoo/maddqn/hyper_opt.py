"""
Lvl 7:
"""
from argparse import ArgumentParser
import collections
import itertools
import sys
import typing
import os
from datetime import datetime

# import neptune.new as neptune
# from absl import app
# from absl import flags
# from absl import logging


import chex
import dm_env
import haiku as hk
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import optax
from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.maddqn import agent


def generic_parser():
  parser = ArgumentParser()
  parser.add_argument('--environment_name', type=str, default='pong')
  parser.add_argument('--environment_height', type=int, default=84)
  parser.add_argument('--environment_width', type=int, default=84)
  parser.add_argument('--replay_capacity', type=int, default=int(1e6))
  parser.add_argument('--compress_state', type=bool, default=True)
  parser.add_argument('--min_replay_capacity_fraction', type=float, default=0.05)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--max_frames_per_episode', type=int, default=108000)  # 30 mins.
  parser.add_argument('--num_action_repeats', type=int, default=4)
  parser.add_argument('--num_stacked_frames', type=int, default=4)
  parser.add_argument('--exploration_epsilon_begin_value', type=float, default=1.)
  parser.add_argument('--exploration_epsilon_end_value', type=float, default=0.1)
  parser.add_argument('--exploration_epsilon_decay_frame_fraction', type=float, default=0.02)
  parser.add_argument('--eval_exploration_epsilon', type=float, default=0.05)
  parser.add_argument('--target_network_update_period', type=int, default=int(4e4))
  parser.add_argument('--grad_error_bound', type=float, default=1./32)
  parser.add_argument('--learning_rate', type=float, default=0.00005)
  parser.add_argument('--optimizer_epsilon', type=float, default=0.01/32**2)
  parser.add_argument('--additional_discount', type=float, default=0.99)
  parser.add_argument('--max_abs_reward', type=float, default=1.)
  parser.add_argument('--seed', type=int, default=1)  # GPU may introduce nondeterminism.
  parser.add_argument('--num_iterations', type=int, default=200)
  parser.add_argument('--num_train_frames', type=int, default=int(1e6))  # Per iteration.
  parser.add_argument('--num_eval_frames', type=int, default=int(5e5))  # Per iteration.
  parser.add_argument('--learn_period', type=int, default=16)
  parser.add_argument('--results_csv_path', type=str, default=None)

  parser.add_argument('--num_avars', type=int, default=51)
  parser.add_argument('--mixture_ratio', type=int, default=0.8)
  parser.add_argument('--jax_platform_name', default='gpu')  # Default to GPU.
  parser.add_argument('--jax_numpy_rank_promotion', default='raise')
  return parser


def parse_args():
  parser = generic_parser()
  parser.add_argument('--learning_rates', nargs='+', type=float, required=True)

  return parser.parse_args()


def main(args):
  """Trains MAD-DQN agent on Atari."""
  print(args)

  for lr in args.learning_rates:
    args.learning_rate = lr
    args.method = 'maddqn'

    if 'WANDB_PROJECT' in os.environ and 'WANDB_ENTITY' in os.environ:
      import wandb as wandb_run
      wandb_run.init(project=os.environ['WANDB_PROJECT'], entity=os.environ["WANDB_ENTITY"])
    else:
      wandb_run = None

    if wandb_run:
      wandb_run.config.update(args)
    benchmark_args(args, wandb_run=wandb_run)



def benchmark_args(args, wandb_run):
  print('MAD-DQN on Atari on ', jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(args.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64))

  if args.results_csv_path:
    writer = parts.CsvWriter(args.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        args.environment_name, seed=random_state.randint(1, 2**32))
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  print('Environment: ', args.environment_name)
  print('Action spec: ', env.action_spec())
  print('Observation spec: ', env.observation_spec())
  num_actions = env.action_spec().num_values
  num_avars = args.num_avars
  avars = jnp.arange(0, num_avars) / float(num_avars)
  network_fn = networks.ad_atari_network(num_actions, avars)
  network = hk.transform(network_fn)


  def preprocessor_builder():
    return processors.atari(
        additional_discount=args.additional_discount,
        max_abs_reward=args.max_abs_reward,
        resize_shape=(args.environment_height, args.environment_width),
        num_action_repeats=args.num_action_repeats,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=args.num_stacked_frames,
        grayscaling=True,
    )


  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)
  sample_network_input = sample_processed_timestep.observation
  chex.assert_shape(sample_network_input,
                    (args.environment_height, args.environment_width,
                     args.num_stacked_frames))

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(args.min_replay_capacity_fraction * args.replay_capacity *
                  args.num_action_repeats),
      decay_steps=int(args.exploration_epsilon_decay_frame_fraction *
                      args.num_iterations * args.num_train_frames),
      begin_value=args.exploration_epsilon_begin_value,
      end_value=args.exploration_epsilon_end_value)

  if args.compress_state:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t))

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t))
  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.TransitionReplay(args.replay_capacity, replay_structure,
                                       random_state, encoder, decoder)

  optimizer = optax.adam(
      learning_rate=args.learning_rate,
      #decay=0.95,
      eps=args.optimizer_epsilon,
      #centered=True,
  )

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent.MadDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      avars=avars,
      optimizer=optimizer,
      transition_accumulator=replay_lib.TransitionAccumulator(),
      replay=replay,
      batch_size=args.batch_size,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=args.min_replay_capacity_fraction,
      learn_period=args.learn_period,
      target_network_update_period=args.target_network_update_period,
      grad_error_bound=args.grad_error_bound,
      rng_key=train_rng_key,
      mixture_ratio=args.mixture_ratio,
  )
  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=args.eval_exploration_epsilon,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= args.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    print(datetime.now(), 'Training iteration ', state.iteration)
    train_seq = parts.run_loop(train_agent, env, args.max_frames_per_episode)
    num_train_frames = 0 if state.iteration == 0 else args.num_train_frames
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    train_trackers = parts.make_default_trackers(train_agent)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)


    print(datetime.now(), 'Evaluation iteration ', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, args.max_frames_per_episode)
    eval_seq_truncated = itertools.islice(eval_seq, args.num_eval_frames)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    # Logging and checkpointing.
    human_normalized_score = atari_data.get_human_normalized_score(
        args.environment_name, eval_stats['episode_return'])
    capped_human_normalized_score = np.amin([1., human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * args.num_train_frames, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_state_value', train_stats['state_value'], '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1. - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)

    if wandb_run:
      wandb_run.log({k: v for k, v, _ in log_output})
      # for n, v, f in log_output:
      #   neptune_run[n].log(v)

    print(datetime.now(), log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  if wandb_run:
    wandb_run.log({"eval/capped_normalized_final": capped_human_normalized_score})
    wandb_run.log({'train/episode_return': train_stats['episode_return']})
    wandb_run.finish()

  writer.close()


if __name__ == '__main__':
  args = parse_args()
  main(args)
