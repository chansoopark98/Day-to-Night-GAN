import tensorflow_io as tfio
import tensorflow_datasets as tfds
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

class Dataset:
    def __init__(self, data_dir, image_size, batch_size, dataset='DayToNight'):
        """from i
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
            dataset: 데이터셋 종류 ('DayToNight')
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset

        self.train_data, self.number_train = self._load_train_datasets()
        self.valid_data, self.number_valid = self._load_valid_datasets()


    def _load_valid_datasets(self):
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:1%]')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid


    def _load_train_datasets(self):
        # train -> 30000
        # train[:1%] -> 300
        # train[1%:] -> 29700
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[1%:]')

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)

        return train_data, number_train

    @tf.function
    def prepare_train_ds(self, sample):
        img = sample['rgb']
        label = sample['label']

        # data augmentation
        if tf.random.uniform([], minval=0, maxval=1) > 0.5:
            img = tf.image.flip_left_right(img)
            
        
        scale = tf.random.uniform([], 0.5, 1.5)
        
        nh = self.image_size[0] * scale
        nw = self.image_size[1] * scale

        img = tf.image.resize(img, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.image.resize_with_crop_or_pad(img, self.image_size[0], self.image_size[1])
        
        img = (img / 127.5) -1.
        
        return (img, label)

    
    @tf.function
    def prepare_valid_ds(self, sample):
        img = sample['rgb']
        label = sample['label']

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        
        img = (img / 127.5) -1.

        return (img, label)

    @tf.function
    def prepare_test_ds(self, sample):
        img = sample['rgb']
        label = sample['label']

        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)

        img /= 255.
        return (img, label)


    def generate_patch_labels(self, batch_size: int, disc_patch, random_augment: bool):
        fake_y_dis = tf.zeros((batch_size,) + disc_patch)
        real_y_dis = tf.ones((batch_size,) + disc_patch)

        if random_augment:
            if tf.random.uniform([]) < 0.05:
                real_factor = tf.random.uniform([], minval=0.8, maxval=1.)
                real_y_dis *= real_factor

        return fake_y_dis, real_y_dis


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.prepare_train_ds, num_parallel_calls=AUTO)
        # train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.prepare_valid_ds, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data

    
    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.prepare_test_ds, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data