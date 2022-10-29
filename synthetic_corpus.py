# Copyright 2021 The DDSP Authors.
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

# Lint as: python3
"""Functions to generate self-supervised signal, EXPERIMENTAL."""
import random
import warnings

import ddsp
import numpy as np
import tensorflow.compat.v2 as tf

from scipy.interpolate import interp1d


def flip(p=0.5):
  return np.random.uniform() <= p


def uniform_int(minval=0, maxval=10):
  return np.random.random_integers(int(minval), int(maxval))


def uniform_float(minval=0.0, maxval=10.0):
  return np.random.uniform(float(minval), float(maxval))


def random_blend(length, env_start=1.0, env_end=0.0, exp_max=2.0):
  """Returns a linear mix between two values, with a random curve steepness."""
  exp = uniform_float(-exp_max, exp_max)
  v = np.linspace(1.0, 0.0, length) ** (2.0 ** exp)
  return env_start * v + env_end * (1.0 - v)


def random_harm_dist(n_harmonics=100, low_pass=True, rand_phase=0.0):
  """Create harmonic distribution out of sinusoidal components."""
  n_components = uniform_int(1, 20)
  smoothness = uniform_float(1.0, 10.0)
  coeffs = np.random.rand(n_components)
  freqs = np.random.rand(n_components) * n_harmonics / smoothness

  v = []
  for i in range(n_components):
    v_i = (coeffs[i] * np.cos(
        np.linspace(0.0, 2.0 * np.pi * freqs[i], n_harmonics) +
        uniform_float(0.0, np.pi * 2.0 * rand_phase)))
    v.append(v_i)

  if low_pass:
    v = [v_i * np.linspace(1.0, uniform_float(0.0, 0.5), n_harmonics) **
         uniform_float(0.5, 2.0) for v_i in v]
  harm_dist = np.sum(np.stack(v), axis=0)
  return harm_dist


def running_mean(x, N):
  return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def random_walk(length, smooth_win):
  y = np.cumsum(np.random.normal(loc=0., scale=1., size=(length,)))
  y = running_mean(y, smooth_win)
  return y


def random_walk_blend(length, env_start=1.0, env_end=0.0, exp_max=2.0):
  """Returns a linear mix between two values, with a random curve steepness."""
  def _rw():
    v = random_walk(length * 10, min(length, 10))
    max_idx = np.argmax(v)
    min_idx = np.argmin(v)
    if max_idx > min_idx:
      max_to_min_v = v[min_idx: max_idx][::-1]
      N = max_idx - min_idx
    else:
      max_to_min_v = v[max_idx: min_idx]
      N = min_idx - max_idx
    min_v = v[min_idx]
    max_v = v[max_idx]
    return max_to_min_v, N, max_v, min_v

  max_to_min_v, N, max_v, min_v = _rw()
  while N < 2:
    max_to_min_v, N, max_v, min_v = _rw()

  interp_func = interp1d(np.arange(N), max_to_min_v)
  ids = np.linspace(0, N - 1, length)
  max_to_min_v = interp_func(ids)

  blend_coef = (max_to_min_v - min_v) / (max_v - min_v)
  return env_start * blend_coef + env_end * (1 - blend_coef)


def generate_control(n_batch=1,
                      n_timesteps=125,
                      n_harmonics=100,
                      n_mags=65,
                      min_note_length=5,
                      max_note_length=25,
                      p_silent=0.1,
                      p_vibrato=0.8,
                      get_controls=True,
                      sample_rate=16000):
  harm_amp = np.zeros([n_batch, n_timesteps, 1])
  harm_dist = np.zeros([n_batch, n_timesteps, n_harmonics])
  f0_midi = np.zeros([n_batch, n_timesteps, 1])
  mags = np.zeros([n_batch, n_timesteps, n_mags])

  for b in range(n_batch):
    t_start = 0
    while t_start < n_timesteps:
      note_length = uniform_int(min_note_length, max_note_length)
      t_end = min(t_start + note_length, n_timesteps)
      note_length = t_end - t_start

      # Silent?
      silent = flip(p_silent)
      if silent:
        # Amplitudes.
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice -= 10.0

      else:
        # Amplitudes.
        amp_start = uniform_float(-1.0, 3.0)
        amp_end = uniform_float(-1.0, 3.0)
        if flip(0.5):
          amp_blend = random_walk_blend(note_length, amp_start, amp_end)
        else:
          amp_blend = random_blend(note_length, amp_start, amp_end)
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice += amp_blend[:, np.newaxis]

        # Add some noise.
        ha_slice += uniform_float(0.0, 0.1) * np.random.randn(*ha_slice.shape)

        # Harmonic Distribution.
        low_pass = flip(0.8)
        rand_phase = uniform_float(0.0, 0.4)
        harm_dist_start = random_harm_dist(n_harmonics,
                                           low_pass=low_pass,
                                           rand_phase=rand_phase)[np.newaxis, :]
        harm_dist_end = random_harm_dist(n_harmonics,
                                         low_pass=low_pass,
                                         rand_phase=rand_phase)[np.newaxis, :]
        if flip(0.5):
          blend = random_walk_blend(note_length, 1.0, 0.0)[:, np.newaxis]
        else:
          blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]
        harm_dist_blend = (harm_dist_start * blend +
                           harm_dist_end * (1.0 - blend))
        hd_slice = harm_dist[b, t_start:t_end, :]
        hd_slice += harm_dist_blend

        # Add some noise.
        hd_slice += uniform_float(0.0, 0.5) * np.random.randn(*hd_slice.shape)
        if flip(0.5):
          mask_idx = np.random.randint(7, 100)
          hd_slice[:, mask_idx:] = -np.inf

        # Fundamental Frequency.
        f0 = uniform_float(24.0, 84.0)
        if flip(p_vibrato):
          vib_start = uniform_float(0.0, 1.0)
          vib_end = uniform_float(0.0, 1.0)
          vib_periods = uniform_float(0.0, note_length * 2.0 / min_note_length)
          if flip(0.5):
            vib_blend = random_walk_blend(note_length, vib_start, vib_end)
          else:
            vib_blend = random_blend(note_length, vib_start, vib_end)
          if flip(0.5):
            vib = vib_blend * np.sin(
                np.linspace(0.0, 2.0 * np.pi * vib_periods, note_length))
            f0_note = f0 + vib
          else:
            f0_note = f0 + vib_blend
        else:
          f0_note = f0 * np.ones([note_length])

        f0_slice = f0_midi[b, t_start:t_end, :]
        f0_slice += f0_note[:, np.newaxis]

        # Add some noise.
        f0_slice += uniform_float(0.0, 0.1) * np.random.randn(*f0_slice.shape)

      # Filtered Noise.
      low_pass = flip(0.8)
      rand_phase = uniform_float(0.0, 0.4)
      mags_start = random_harm_dist(n_mags,
                                    low_pass=low_pass,
                                    rand_phase=rand_phase)[np.newaxis, :]
      mags_end = random_harm_dist(n_mags,
                                  low_pass=low_pass,
                                  rand_phase=rand_phase)[np.newaxis, :]
      if flip(0.5):
        blend = random_walk_blend(note_length, 1.0, 0.0)[:, np.newaxis]
      else:
        blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]

      mags_blend = mags_start * blend + mags_end * (1.0 - blend)

      mags_slice = mags[b, t_start:t_end, :]
      mags_slice += mags_blend

      # Add some noise.
      mags_slice += uniform_float(0.0, 0.2) * np.random.randn(*mags_slice.shape)

      # # Scale.
      mags_slice -= uniform_float(1.0, 10.0)

      t_start = t_end

  if get_controls:
    harm_amp = ddsp.core.exp_sigmoid(harm_amp)
    harm_amp /= uniform_float(1.0, [2.0, uniform_float(2.0, 10.0)][flip(0.2)])

  # Frequencies.
  f0_hz = ddsp.core.midi_to_hz(f0_midi)

  if get_controls:
    harm_dist = tf.nn.softmax(harm_dist)
    harm_dist = ddsp.core.remove_above_nyquist(f0_hz, harm_dist, sample_rate=sample_rate)
    harm_dist = ddsp.core.safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

  if get_controls:
    mags = ddsp.core.exp_sigmoid(mags)

  sin_amps, sin_freqs = ddsp.core.harmonic_to_sinusoidal(
      harm_amp, harm_dist, f0_hz, sample_rate=sample_rate)

  controls = {'harm_amp': harm_amp,
              'harm_dist': harm_dist,
              'f0_hz': f0_hz,
              'sin_amps': sin_amps,
              'sin_freqs': sin_freqs,
              'noise_magnitudes': mags}
  return controls
