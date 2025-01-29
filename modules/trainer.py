import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from transform import (FEATURE_KEY, LABEL_KEY, transformed_name)

def gzip_reader_fn(filenames):
  """ Loads compressed data

  Args:
    filenames(str): Path to the data directory

  Return:
    TfRecord: Compressed data

  """
  return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64)->tf.data.Dataset:
  """ Creates a batched dataset from TFRecord file

  Args:
    file_pattern(str): Path pattern to the TFRecord file
    tf_transform_output(tft.TFTransformOutput): Output from TFX transformation
    batch_size(int): Batch size for the dataset

  Returns:
    tf.data.Dataset: Dataset that has been batched and ready for training
  """

  transform_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy()
  )

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transform_feature_spec,
      reader=gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=transformed_name(LABEL_KEY)
  )

  return dataset

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

embedding_dim=16

def model_builder() :
  """ Build keras model

  Returns:
    tf.keras.Model: Keras Model that has been compiled.
  """
  inputs = tf.keras.Input(shape=(1,), name = transformed_name(FEATURE_KEY), dtype = tf.string)
  reshaped_narrative = tf.reshape(inputs, [-1])
  x = vectorize_layer(reshaped_narrative)
  x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, name = "embedding")(x)
  x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
  x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
  x = tf.keras.layers.Dense(64, activation = "relu")(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation = "relu")(x)
  outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)

  model = tf.keras.Model(inputs = inputs, outputs = outputs)

  model.compile(
      loss = 'binary_crossentropy',
      optimizer = tf.keras.optimizers.Adam(0.01),
      metrics = [tf.keras.metrics.BinaryAccuracy()]
  )

  model.summary()
  return model

def get_serve_tf_examples_fn(model, tf_transform_output):
  """ Return a function that parses a serialized tf.Example and applies TFT

  Args:
    model(tf.keras.Model): The model to be used for serving.
    tf_transform_output(tft.TFTransformOutput): Output of the TFX Transformation

  Returns
    serve_tf_examples_fn(function): Function that processes the serialized input tf.Example and returns the model output
  """

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """ Parse and transform serialized tf.Example data, then return model predictions

    Args:
      serialized_tf_examples(tf.Tensor): A batch of serialized tf.Example data

    Returns:
      predictions(tf.Tensor): The model's predictions for the input data
    """
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(LABEL_KEY)

    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    predictions = model(transformed_features)

    return predictions

  return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
  """ Train the model based on given args

  Args:
    fn_args(FnArgs): Arguments provided by TFX, including:
      - train_files: List of training data files.
      - eval_files: List of evaluation data files.
      - transform_output: Output from the TFX transformation component.
      - train_steps: Number of training steps per epoch.
      - eval_steps: Number of evaluation steps per epoch.
      - serving_model_dir: Directory to save the trained model.
      - hyperparameters: Best hyperparameters from the tuner.
  """

  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = log_dir, update_freq='batch'
  )

  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
  val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
  vectorize_layer.adapt(
      [j][0].numpy()[0] for j in [
          i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)
      ]
  )

  model = model_builder()

  model.fit(
      x=train_set,
      validation_data=val_set,
      callbacks=[tensorboard_callback, early_stop, model_checkpoint],
      steps_per_epoch=200,
      validation_steps=200,
      epochs=10
  )

  signatures = {
      'serving_default':
      get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
          tf.TensorSpec(
              shape=[None],
              dtype=tf.string,
              name='examples'
          )
      )
  }

  print("Model will be saved to:", fn_args.serving_model_dir)
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
