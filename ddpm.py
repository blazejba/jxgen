import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


if __name__ == '__main__':
    # Download the dataset
    ds, ds_info = tfds.load('oxford_flowers102', split='train', with_info=True)

    # Prefetching and batching
    ds = ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    for batch in ds:
        images, labels = batch

        # Convert to JAX DeviceArray
        images = jnp.array(images)
        labels = jnp.array(labels)
        
        print(images.shape)
        exit()