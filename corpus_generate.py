import ddsp
import soundfile
import numpy as np
from synthetic_corpus import generate_control
from pathlib import Path
from bounded_pool_executor import BoundedProcessPoolExecutor


class Synthesizer(object):
    def __init__(self, n_samples, sample_rate):
        self.sinusoidal_synth = ddsp.synths.Sinusoidal(
            n_samples=n_samples,
            sample_rate=sample_rate,
            amp_scale_fn=None,
            freq_scale_fn=None,
            name='sinusoidal')
        self.filtered_noise_synth = ddsp.synths.FilteredNoise(
            n_samples=n_samples,
            window_size=0,
            scale_fn=None,
            name='filtered_noise')

        dag = [
            (self.sinusoidal_synth,
             ['amplitudes', 'frequencies']),
            (self.filtered_noise_synth,
             ['noise_magnitudes']),
            (ddsp.processors.Add(),
             [f'{self.filtered_noise_synth.name}/signal',
              f'{self.sinusoidal_synth.name}/signal']),
        ]
        """Convert synthetic controls into audio."""
        self.processor_group = ddsp.processors.ProcessorGroup(dag=dag)

    def generate_synthetic_audio(self, features):
        return self.processor_group({
            'amplitudes': features['sin_amps'],
            'frequencies': features['sin_freqs'],
            'noise_magnitudes': features['noise_magnitudes']
        })


def test():
    # frame shift 20 ms, 480 sample points.
    synth = Synthesizer(n_samples=96000, sample_rate=24000)
    features = generate_control(n_timesteps=200)
    audio = synth.generate_synthetic_audio(features)
    f0 = features['f0_hz'].numpy()[0, :, 0]
    amp = features['harm_amp'].numpy()[0, :, 0]
    soundfile.write('test_audio.wav', audio[0], 24000, 'PCM_16')
    print(audio.shape)


def generate(seg_id, seed):
    np.random.seed(seed)
    synth = Synthesizer(n_samples=192000, sample_rate=24000)
    save_dir = Path('/data/corpus/song_bbmulti/synthetic_24k')
    save_dir.mkdir(exist_ok=True)
    for i in range(100):
        save_prefix = save_dir / f'synthetic-{seg_id:0>6d}{i:0>3d}'
        features = generate_control(n_timesteps=400, max_note_length=40, sample_rate=24000, p_vibrato=1.)
        audio = synth.generate_synthetic_audio(features)[0]
        f0 = features['f0_hz'].numpy()[0, :, 0]
        amp = features['harm_amp'].numpy()[0, :, 0]
        soundfile.write(save_prefix.with_suffix('.wav'), audio, 24000, 'PCM_16')
        np.save(save_prefix.as_posix() + '-f0.npy', f0)
        np.save(save_prefix.as_posix() + '-amp.npy', amp)


if __name__ == '__main__':
    np.random.seed(123456)
    num_samples = 100
    executor = BoundedProcessPoolExecutor(max_workers=1)
    futures = []
    for i in range(num_samples // 100):
        seed = np.random.randint(0, 2**32)
        futures.append(executor.submit(generate, i, seed))
    [f.result() for f in futures]

