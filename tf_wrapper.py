from __future__ import print_function

import pickle

import numpy as np
import tensorflow.google as tf

from triplets import generate_triplets

class Wrapper(object):
  
    def __init__(self, config):
        self.config = config

    def build_tf_model(self, embed_init):
        embeddings = tf.get_variable(
            'embeddings',
            [self.num_examples, config.num_dims],
            initializer=tf.random_normal_initializer(stddev=embed_init or 0.0001))
        triplets = tf.constant(self.triplets)
        weights = tf.constant(self.weights)

        # Free up some memory
        del self.triplets
        del self.weights

        self.tf_t = tf.placeholder(tf.float32, name='t')

        y_ij = tf.gather(embeddings, triplets[:, 0]) - tf.gather(embeddings, triplets[:, 1])
        y_ik = tf.gather(embeddings, triplets[:, 0]) - tf.gather(embeddings, triplets[:, 2])

        d_ij = 1 + tf.reduce_sum(y_ij**2, axis=-1)
        d_ik = 1 + tf.reduce_sum(y_ik**2, axis=-1)

        self.embeddings = embeddings
        self.loss = tf.tensordot(weights, log_t(d_ij / d_ik, self.tf_t), axes=1)
        self.num_viol = tf.reduce_sum((tf.to_int32(tf.greater(d_ij, d_ik))))
        self.num_triplets = triplets.shape[0].value

    def embed(self, embed_init=None, return_seq=False):
        assert not return_seq
        self.build_tf_model(embed_init)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        lr = 1000.0
        eta = lr * self.num_examples / self.num_triplets
        tf_eta = tf.placeholder_with_default(eta, (), name='lr')

        # full-batch gradient descent
        if self.config.optimizer == 'gd':
            opt = tf.GradientDescentOptimizer(tf_eta)
        elif self.config.optimizer == 'gd-momentum':
            opt = tf.MomentumOptimizer(tf_eta, .9)
        elif self.config.optimizer == 'rmsprop':
            opt = tf.RMSPropOptimizer(tf_eta)
        elif self.config.optimizer == 'adam':
            opt = tf.AdamOptimizer(tf_eta)

        apply_grad = opt.minimize(self.loss)

        tol = 1e-7
        l = np.inf

        t = self.config.t
        if self.config.anneal_scheme != 1:
            tmin = self.config.t
            tmax = self.config.t_max

        num_iters = 1000
        projections = []
        for itr in range(self.config.num_iters):
            old_l = l

            if self.config.anneal_scheme == 1:
                # scale t linearly by fifths after first half of training
                if itr >= self.config.num_iters / 2.0:
                    if itr % int(num_iters / 10.0) == 0:
                        t += (tmax - tmin) / 5.0

            elif self.config.anneal_scheme == 2:
                # scale t linearly throughout training
                t += (tmax - tmin) / num_iters

            l, nv, _ = sess.run(
                [self.loss, self.num_viol, apply_grad],
                {self.tf_t: t, tf_eta: eta})

            viol = nv / self.num_triplets

            if 'gd' in self.config.optimizer:
                if l > old_l + tol:
                    eta *= 0.9
                else:
                    eta *= 1.01

            if not itr or (itr+1) % 50 == 0:
                print('[{}/{}] Loss: {:3.3f} Triplet Error: {:.2%}'. \
                    format(itr+1, num_iters, l, viol))
        return sess.run(self.embeddings)
    
    def load_triplets(self, path):
        with open(path, 'rb') as f:
            print('[*] Loading triplets from %s' % path)
            self.triplets, self.weights, self.num_examples = pickle.load(f)

    def generate_triplets(self, X, path=None):
        self.num_examples = X.shape[0]
        self.triplets, self.weights = generate_triplets(X, svd_dim=self.config.svd_dim, verbose=self.config.verbose)
        if path:
            print('[*] Saving generated triplets to %s' % path)
            with open(path, 'wb') as f:
                pickle.dump((self.triplets, self.weights, self.num_examples), f)

    def load_state(self, path):
        raise NotImplementedError
    
    def save_state(self, path):
        raise NotImplementedError


def log_t(x, t=2):
  return tf.cond(
      tf.less(tf.abs(t-1), 0.01),
      lambda: tf.log(x + 1),
      lambda: ((x + 1)**(1 - t) - 1) / (1 - t)
  )
