import tensorflow as tf
from tensorflow.keras import layers, initializers, models
from .random import split_key


class TransformerEncoder(layers.Layer):
	""" Transformer Encoder Block
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
		
		# initialise attention
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
		
		# initialise feedforward
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
	
	# type: (TransformerEncoder, np.ndarray) -> np.ndarray
	def call(self, x):
		
		skip0 = x
		z = self.attention(x, x) ###! the MultiHeadAttention layer projects x to Q, K internally
		z = self.drop0(z)
		z = self.norm0(z+skip0)
		
		skip1 = z
		z = self.dense0(z)
		z = self.dense1(z)
		z = self.drop1(z)
		z = self.norm1(z+skip1)
		
		return z


class FixedLearnablePositionalEncoding(layers.Layer):
	""" Fixed size learnable positional encoding
	"""
	
	# type: (FixedLearnablePositionalEncoding, int, int) -> None
	def __init__(self, key, sequence_length, *args, **kwargs):
		super(FixedLearnablePositionalEncoding, self).__init__(*args, **kwargs)
		self.init_key = key
		self.sequence_length = sequence_length
	
	# type: (FixedLearnablePositionalEncoding, Tuple[int]) -> None
	def build(self, input_shape):
		self.bias = self.add_weight(
			shape=(self.sequence_length, input_shape[-1]),
			initializer=initializers.RandomUniform(seed=self.init_key),
			trainable=True,
			name='bias'
		)
	
	# type: (PositionEncode, np.ndarray) -> np.ndarray
	def call(self, x):
		return x + self.bias


class PermanentGaussianNoise(layers.GaussianNoise):
	""" Gaussian Noise layer active at both training and inference time
	"""
	
	# type: (PermanentGaussianNoise, np.ndarray, None) -> np.ndarray
	def call(self, x, training=None):
		return super().call(x, training=True)


def get_denoising_transformer_encoder(
		key,
		n_tokens,
		token_dim,
		embed_dim,
		hidden_dim,
		n_blocks,
		n_heads=8,
		dropout=0.1,
		noise_sd=0.3,
		name='Denoising-Transformer-Encoder'
	):
	""" Instantiate Denoising Transformer Encoder
	# type: (int, int, int, int, int, int, int, float, float, str) -> tf.keras.models.Model
	
	- Sequence length is fixed by position encoding layer
	- Uses functional interface internally
	"""
	
	# split key
	noise_key, embed_key, encoder_key = split_key(key, n=3)
	embed_keys = split_key(embed_key, n=3)
	encoder_keys = split_key(encoder_key, n=n_blocks)
	
	# init input with gaussian perturbation
	x = tf.keras.Input(shape=(n_tokens, token_dim))
	z = PermanentGaussianNoise(noise_sd, seed=noise_key)(x) # P(-1 <= U(0,0.3) <= +1) = 0.9991419
	
	# init input embedding
	input_embedding = layers.Dense(
		embed_dim,
		activation=None,
		kernel_initializer=initializers.GlorotUniform(seed=embed_keys[0]),
		bias_initializer='zeros',
		name='input_embedding'
	)
	z = input_embedding(z)
	###! print([w.shape for w in input_embedding.get_weights()])
	###! position encoding is not made redundant by embedding biases 
	###! because embedding biases are applied token-wise
	z = FixedLearnablePositionalEncoding(
		embed_keys[1],
		n_tokens
	)(z)
	
	# init encoder stack
	for sub_key in encoder_keys:
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
		kernel_initializer=initializers.GlorotUniform(seed=embed_keys[2]),
		bias_initializer='zeros',
		name='output_embedding'
	)(z)
	
	#finalise model
	model = models.Model(inputs=x, outputs=z, name=name)
	model.init_key = key
	
	return model


def convert_dte_to_classifier(
		denoising_model,
		n_classes,
		key=None,
		name='Transformer-Encoder-Classifier-from-DTE-backbone'
	):
	""" Instantiate Transformer Encoder Classifier from Denoising Transformer Encoder backbone
	# type: (tf.keras.models.Model, int, int, str) -> tf.keras.models.Model
	
	If no key is specified uses the cached DTE init key
	"""
	
	# determine key
	if key is None:
		key = denoising_model.init_key
	
	# get input
	x = denoising_model.input
	
	# skip over noise layer to input embedding
	z = denoising_model.layers[2](x)
	
	# skip output embedding
	for layer in denoising_model.layers[3:-1]:
		z = layer(z)
	
	# init classification head
	yh = layers.Dense(
		n_classes,
		activation='softmax',
		kernel_initializer=initializers.GlorotUniform(seed=key),
		bias_initializer='zeros',
		name='classification_head'
	)(z)
	yh = layers.Lambda(
		lambda x : x[:, 0, :],
		name='classification_token'
	)(yh)
	
	# finalise model
	model = models.Model(inputs=x, outputs=yh, name=name)
	
	return model




### Model subclass DTE
###! build related bug: attempts to trigger build() unsuccessful

# class DenoisingTransformerEncoder(models.Model):
	# """
	# """
	
	# # type: () ->
	# def __init__(self,
			# key,
			# n_tokens,
			# token_dim,
			# embed_dim,
			# hidden_dim,
			# n_blocks,
			# n_heads=8,
			# dropout=0.1,
			# noise_sd=0.3,
			# name='Denoising-Transformer-Encoder',
			# **kwargs
		# ):
		# super(DenoisingTransformerEncoder, self).__init__(name=name, **kwargs)
		
		# # split keys
		# noise_key, embed_key, encoder_key = split_key(key, n=3)
		# embed_keys = split_key(embed_key, n=3)
		# encoder_keys = split_key(encoder_key, n=n_blocks)
		
		# # store params
		# self.init_key = key
		# self.n_tokens = n_tokens
		# self.token_dim = token_dim
		# self.embed_dim = embed_dim
		# self.hidden_dim = hidden_dim
		# self.n_blocks = n_blocks
		# self.n_heads = n_heads
		# self.dropout = dropout
		# self.noise_sd = noise_sd
		
		# # initialise layers
		# self.gaussian_noise = layers.GaussianNoise(noise_sd, seed=noise_key)
		
		# self.input_embedding = layers.Dense(
			# embed_dim,
			# activation=None,
			# kernel_initializer=initializers.GlorotUniform(seed=embed_keys[0]),
			# bias_initializer='zeros',
			# name='input_embedding'
		# )
		
		# self.position_encode = PositionEncode(embed_keys[1], n_tokens)
		
		# self.transformer_encoders = [
			# TransformerEncoder(
				# sub_key,
				# embed_dim,
				# hidden_dim,
				# n_heads=n_heads,
				# dropout=dropout
			# )
			# for sub_key in encoder_keys
		# ]
		
		# self.output_embedding = layers.Dense(
			# token_dim,
			# activation='tanh',
			# kernel_initializer=initializers.GlorotUniform(seed=embed_keys[2]),
			# bias_initializer='zeros',
			# name='output_embedding'
		# )
	
	# def call(self, x):
		# z = self.gaussian_noise(x)
		# z = self.input_embedding(z)
		# z = self.position_encode(z)
		# for encoder in self.transformer_encoders:
			# z = encoder(z)
		# z = self.output_embedding(z)
		# return z
