import tensorflow as tf
from tensorflow.keras import layers, initializers, models
from .random import split_key


class TransformerEncoder(layers.Layer):
	"""
	"""
	
	# type: (TransformerEncoder, int, int, int, int, float) -> None
	def __init__(self,
			key,
			input_dim,
			hidden_dim,
			n_heads=8,
			dropout=0.1,
			*args, 
			**kwargs
		):
		super(TransformerEncoder, self).__init__(*args, **kwargs)
		
		# split keys
		attn_key, ffn_key = split_key(key, n=2)
		attn_keys = split_key(attn_key, n=2)
		ffn_keys = split_key(ffn_key, n=3)
		
		# initialise attention portion
		self.attention = layers.MultiHeadAttention(
			n_heads, 
			input_dim//n_heads,
			kernel_initializer=initializers.GlorotUniform(seed=attn_keys[0]),
			bias_initializer='zeros'
		)
		self.drop0 = layers.Dropout(
			dropout,
			seed=attn_keys[1]
		)
		self.norm0 = layers.LayerNormalization(
			epsilon=1e-6,
			beta_initializer='zeros',
			gamma_initializer='ones'
		)
		
		# initialise feedforward portion
		self.dense0 = layers.Dense(
			hidden_dim, 
			activation='relu',
			kernel_initializer=initializers.GlorotUniform(seed=ffn_keys[0]),
			bias_initializer='zeros',
		)
		self.dense1 = layers.Dense(
			input_dim,
			activation=None,
			kernel_initializer=initializers.GlorotUniform(seed=ffn_keys[1]),
			bias_initializer='zeros',
		)
		self.drop1 = layers.Dropout(
			dropout,
			seed=ffn_keys[2]
		)
		self.norm1 = layers.LayerNormalization(
			epsilon=1e-6,
			beta_initializer='zeros',
			gamma_initializer='ones'
		)
	
	# type: (np.ndarray) -> np.ndarray
	def call(self, x):
		
		skip0 = x
		z = self.attention(x, x)
		z = self.drop0(z)
		z = self.norm0(z+skip0)
		
		skip1 = z
		z = self.dense0(z)
		z = self.dense1(z)
		z = self.drop1(z)
		z = self.norm1(z+skip1)
		
		return z


class PositionEncoding(layers.Layer):
	"""
	"""
	
	def __init__(self, key, sequence_length, *args, **kwargs):
		super(PositionEncoding, self).__init__(*args, **kwargs)
		self.init_key = key
		self.sequence_length = sequence_length

	def build(self, input_shape):
		self.bias = self.add_weight(
			'bias',
			shape=(self.sequence_length, input_shape[-1]),
			initializer=initializers.RandomUniform(seed=self.init_key),
			trainable=True
		)
	
	def call(self, x):
		return x + self.bias


# type: (int, int, int, int, int, float) -> tf.keras.models.Model
def get_denoising_transformer_encoder(
		key,
		token_dim,
		embed_dim,
		hidden_dim,
		n_blocks,
		n_heads=8,
		dropout=0.1,
		name='Denoising-Transformer-Encoder'
	):
	
	# split key
	noise_key, input_embed_key, encoder_key, output_embed_key = split_key(key, n=4)
	
	# init input with gaussian perturbation
	x = tf.keras.Input(shape=(None, token_dim))
	z = layers.GaussianNoise(0.3/4, seed=noise_key)(x) # P(-1 <= U(0,0.3) <= +1) = 0.9991419
	
	# init input embedding
	z = layers.Dense(
		embed_dim,
		activation=None,
		kernel_initializer=initializers.GlorotUniform(seed=input_embed_key),
		bias_initializer='zeros',
		name='input_embedding'
	)(z)
	# z = z + PositionEmbedding( ###! additive
		# 256,
		# initializer=initializers.GlorotUniform(seed=input_embed_key) ###! todo split key
	# )(z)
	z = PositionEncoding(
		input_embed_key,
		256
	)(z)
	
	# init encoder stack
	for sub_key in split_key(encoder_key, n=n_blocks):
		z = TransformerEncoder(
			sub_key,
			embed_dim,
			hidden_dim,
			n_heads=n_heads,
			dropout=dropout
		)(z)
	
	# init output embedding
	z = layers.Dense(
		token_dim,
		activation='tanh',
		kernel_initializer=initializers.GlorotUniform(seed=output_embed_key),
		bias_initializer='zeros',
		name='output_embedding'
	)(z)
	
	#finalise model
	model = models.Model(inputs=x, outputs=z)
	model.name = name
	
	return model
