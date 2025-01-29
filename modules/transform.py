import string
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'spam'
FEATURE_KEY = 'text'


def transformed_name(key):
  """ Transform feature key

  Args:
    key(str): the key has to be transformed

  Returns:
    str: transformed key
  """

  return f"{key}_xf"


def preprocessing_fn(inputs):
  """Preprocess input features into transformed features

    Args:
      inputs (dict): map from feature keys to raw features

    Returns:
      dict: map from features keys to transformed features
  """

  outputs = {}

  outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
  outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

  return outputs
