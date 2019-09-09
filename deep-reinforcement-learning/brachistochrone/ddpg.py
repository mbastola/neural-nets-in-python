"""
Manil Bastola
Modified with actor-critic implementations
Original at: https://spinningup.openai.com/en/latest/_modules/spinup/algos/ddpg/ddpg.html#ddpg"""

import numpy as np
import tensorflow as tf
import gym
import time
import matplotlib.pyplot as plt
from datetime import datetime
from ddpg_utils import *

### Implement the DDPG algorithm ###
def ddpg(
    env_fn,
    ac_kwargs=dict(),
    seed=0,
    save_folder=None,
    num_train_episodes=100,
    test_agent_every=15,
    replay_size=int(1e6),
    gamma=0.99, 
    decay=0.995,
    mu_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000, 
    action_noise=0.1,
    max_episode_length=1000):

  time_to_beat = 14.8
  tf.set_random_seed(seed)
  np.random.seed(seed)

  env  = env_fn()
  test_env = env_fn()
  # comment out this line if you don't want to record a video of the agent
  if save_folder is not None:
    test_env = gym.wrappers.Monitor(test_env, save_folder, force=True, video_callable=lambda episode_id: True)
    test_env.reset()
    #env = gym.wrappers.Monitor(env, save_folder, force=True ,video_callable=lambda episode_id: True)
    #env.max_path_length = max_episode_length
    #env.reset()

    
  # get size of state space and action space
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]

  # Maximum value of action
  # Assumes both low and high values are the same
  # Assumes all actions have the same bounds
  # May NOT be the case for all environments
  action_max = env.action_space.high[0]

  # Create Tensorflow placeholders (neural network inputs)
  X = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # state
  A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions)) # action
  X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # next state
  R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
  D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done

  # Main network outputs
  with tf.variable_scope('main'):
    mu = ActorNetwork(X, A, num_actions, action_max, **ac_kwargs)
    q = CriticNetwork(X, A,  **ac_kwargs)
    q_mu = CriticNetwork(X, mu,  **ac_kwargs)
    
  # Target networks
  with tf.variable_scope('target'):
    # max_a{ Q(s', a) } Where this is equal to Q(s', mu(s'))
    # used in the target calculation: r + gamma * max_a{ Q(s',a) }
    mu_targ = ActorNetwork(X, A, num_actions, action_max, **ac_kwargs)
    q_mu_targ = CriticNetwork(X2, mu_targ, **ac_kwargs)


  # Experience replay memory
  replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)

  # Target value for the Q-network loss
  # We use stop_gradient to tell Tensorflow not to differentiate
  # q_mu_targ wrt any params
  # i.e. consider q_mu_targ values constant
  q_target = tf.stop_gradient(R + gamma * (1 - D) * q_mu_targ)

  # DDPG losses
  mu_loss = -tf.reduce_mean(q_mu)
  q_loss = tf.reduce_mean((q - q_target)**2)

  # Train each network separately
  mu_optimizer = tf.train.AdamOptimizer(learning_rate=mu_lr)
  q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
  mu_train_op = mu_optimizer.minimize(mu_loss, var_list=get_vars('main/mu'))
  q_train_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

  # Use soft updates to update the target networks
  target_update = tf.group(
    [tf.assign(v_targ, decay*v_targ + (1 - decay)*v_main)
      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ]
  )

  # Copy main network params to target networks
  target_init = tf.group(
    [tf.assign(v_targ, v_main)
      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
    ]
  )

  # boilerplate (and copy to the target networks!)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(target_init)

  def get_action(s, noise_scale):
    a = sess.run(mu, feed_dict={X: s.reshape(1,-1)})[0]
    a += noise_scale * np.random.randn(num_actions)
    return np.clip(a, -action_max, action_max)

  test_returns = []
  test_times = []
  def test_agent(num_episodes=5):
    t0 = datetime.now()
    n_steps = 0
    for j in range(num_episodes):
      s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
      actionsList = np.zeros(max_episode_length, dtype = np.float32)
      while not (d or (episode_length == max_episode_length)):
        # Take deterministic actions at test time (noise_scale=0)
        #test_env.render()
        a = get_action(s, 0)
        actionsList[episode_length] = a
        s, r, d, _ = test_env.step(a)
        episode_return += r
        episode_length += 1
        n_steps += 1
      timeofCompletion = test_env.getTimeTaken()
      if ( timeofCompletion < time_to_beat) and (episode_return > 0):
        unq = int( time.time() * 1000.0 )
        acts = actionsList[0:episode_length]
        np.save(str(unq)+"_"+str(timeofCompletion)+".txt", acts)  
      print('test return:', episode_return, 'episode_length:', episode_length)
      test_returns.append(episode_return)
      test_times.append(timeofCompletion)
    # print("test steps per sec:", n_steps / (datetime.now() - t0).total_seconds())


  # Main loop: play episode and train
  returns = []
  times = []
  q_losses = []
  mu_losses = []
  num_steps = 0
  for i_episode in range(num_train_episodes):
    # reset env
    s, episode_return, episode_length, d = env.reset(), 0, 0, False
    actionsList = np.zeros(max_episode_length, dtype = np.float32)
    
    while not (d or (episode_length == max_episode_length)):
      # For the first `start_steps` steps, use randomly sampled actions
      # in order to encourage exploration.
      a = 0
      if num_steps > start_steps:
        a = get_action(s, action_noise)
      else:
        a = env.action_space.sample()
      #if i_episode%50 < 5:  
      #  env.render()
      actionsList[episode_length] = a
      # Keep track of the number of steps done
      num_steps += 1
      if num_steps == start_steps:
        print("USING AGENT ACTIONS NOW")

      # Step the env
      s2, r, d, _ = env.step(a)
      episode_return += r
      episode_length += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d_store = False if episode_length == max_episode_length else d

      # Store experience to replay buffer
      replay_buffer.store(s, a, r, s2, d_store)

      # Assign next state to be the current state on the next round
      s = s2

    # Perform the updates
    for _ in range(episode_length):
      batch = replay_buffer.sample_batch(batch_size)
      feed_dict = {
        X: batch['s'],
        X2: batch['s2'],
        A: batch['a'],
        R: batch['r'],
        D: batch['d']
      }

      # Q network update
      # Note: plot the Q loss if you want
      ql, _, _ = sess.run([q_loss, q, q_train_op], feed_dict)
      q_losses.append(ql)
      
      # Policy update
      # (And target networks update)
      # Note: plot the mu loss if you want
      mul, _, _ = sess.run([mu_loss, mu_train_op, target_update], feed_dict)
      mu_losses.append(mul)

    print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)
    returns.append(episode_return)
    timeofCompletion = env.getTimeTaken()
    times.append(timeofCompletion)
    if ( timeofCompletion < time_to_beat) and (episode_return > 0):
      #fout = None
      unq = int( time.time() * 1000.0 )
      #fout  = open("best/"+str(unq)+"_"+str(timeofCompletion)+".txt", "w")  
      acts = actionsList[0:episode_length]
      np.save("best/"+str(unq)+"_"+str(timeofCompletion)+".txt", acts)  
      #fout.write(str(acts))
      #fout.close()
      
    # Test the agent
    if i_episode > 0 and i_episode % test_agent_every == 0:
      test_agent(1)
      
  np.savez('ddpg_results.npz', train=returns, test=test_returns, q_losses=q_losses, mu_losses=mu_losses, times=times, test_times=test_times)
