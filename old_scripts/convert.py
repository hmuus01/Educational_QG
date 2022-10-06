import jax
from jax import numpy as jnp
from transformers import FlaxRobertaForMaskedLM, RobertaForMaskedLM, FlaxT5ForConditionalGeneration
import jax.numpy as jnp
import jax
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import numpy as np

def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)

flax_model = FlaxT5ForConditionalGeneration.from_pretrained("./test")
flax_model.params = to_f32(flax_model.params)
flax_model.save_pretrained("./test")

model = flax_model
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))

tx = optax.adam(learning_rate=0.0001)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)


CKPT_DIR = './test'
checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)
restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)

assert jax.tree_util.tree_all(jax.tree_multimap(lambda x, y: (x == y).all(), state.params, restored_state.params))