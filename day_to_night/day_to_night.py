"""day_to_night dataset."""

import tensorflow_datasets as tfds
import os
import glob
import natsort
import tensorflow as tf


# TODO(day_to_night): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(day_to_night): BibTeX citation
_CITATION = """
"""


class DayToNight(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for day_to_night dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/tensorflow_datasets/'
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cornell_grasp): Specifies the tfds.core.DatasetInfo object
    
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'rgb': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.Tensor(shape=[], dtype=tf.float32)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('input', "depth", "box"),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cornell_grasp): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'dayToNight.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(rgb_path=extracted_path/'rgb', label_path=extracted_path/'label')
    }

  def _generate_examples(self, rgb_path, label_path):
    rgb = os.path.join(rgb_path, '*.png')
    
    
    rgb_files = glob.glob(rgb)
    rgb_files = natsort.natsorted(rgb_files,reverse=True)
    
    label_files = glob.glob(os.path.join(label_path, '*.txt'))
    label_files = natsort.natsorted(label_files,reverse=True)
    
    for i in range(len(rgb_files)):
      with open(label_files[i], "r") as f:
        scalar_value = f.read()
      yield i, {
          'rgb': rgb_files[i],
          'label' : scalar_value
      }
    
    