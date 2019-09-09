import numpy as np
import tensorflow as tf

# simple feedforward neural nets
def ActorNetwork(s, a,num_actions,action_max,hidden_sizes=(300,),hidden_activation=tf.nn.relu,output_activation=tf.tanh):
  with tf.variable_scope('mu'):
    for h in hidden_sizes:
      s = tf.layers.dense(s, units=h, activation=hidden_activation)
    ann = tf.layers.dense(s, units=num_actions, activation=output_activation)
    mu = action_max * ann
  return mu

def CriticNetwork(s, a,  hidden_sizes=(300,),hidden_activation=tf.nn.relu,output_activation=tf.tanh ):
  with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
    input_ = tf.concat([s, a], axis=-1)
    for h in hidden_sizes:
      input_ = tf.layers.dense(input_, units=h, activation=hidden_activation)
    ann = tf.layers.dense(input_, units=1, activation=None)
    q = tf.squeeze(ann, axis=1)
  return q

# get all variables within a scope
def get_vars(scope):
  return [x for x in tf.global_variables() if scope in x.name]

### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y
