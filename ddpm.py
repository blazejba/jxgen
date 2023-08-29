import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def linear_diffusion_schedule(diffusion_times: int,
                              max_rate: float = 0.02,
                              min_rate: float = 0.0001):
    """Linear schedule for diffusion rate."""
    betas       = min_rate + jnp.array(diffusion_times) * (max_rate - min_rate)
    alphas      = 1 - betas
    alpha_bars  = tf.math.cumprod(alphas)

    signal_rates =     alpha_bars
    noise_rates  = 1 - alpha_bars
    return noise_rates, signal_rates
        

def load_dataset():
    ds, _ = tfds.load('oxford_flowers102', split='train', with_info=True)
    ds    = ds.map(lambda x: (tf.image.resize(x['image'], (64, 64)) / 255.0, x['label']))
    ds    = ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == '__main__':
    ds = load_dataset()

    T = 1000
    diffusion_times = [x/T for x in range(T)]
    linear_noise_rates, linear_signal_rates = linear_diffusion_schedule(
        diffusion_times
    )

    for batch in ds:
        images, labels = batch
        images, labels = jnp.array(images), jnp.array(labels)
        
        print(images.shape)
        print(labels.shape)
        exit()