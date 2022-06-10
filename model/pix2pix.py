import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D)
from model.Unet import Unet
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error, MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy, mean_squared_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.metrics import mean_absolute_error

class Pix2Pix():
    def __init__(self,
                 args,
                 image_size: tuple,
                 gen_input_channel: int,
                 gen_output_channel: int,
                 dis_input_channel: int):

        # Set model prefix name
        self.prefix = args.model_prefix
        self.mixed_precision = args.mixed_precision


        # Input shape
        self.image_size = (image_size[0], image_size[1])
        self.gen_input_channel = gen_input_channel
        self.gen_output_channel = gen_output_channel
        self.dis_input_channel = dis_input_channel

        opt = Adam(lr=args.lr, beta_1=0.5)

        # Set mixed precision
        if self.mixed_precision:
            self.policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
            mixed_precision.set_policy(self.policy)
            opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        # Calculate output shape of D (PatchGAN)
        patch = int(self.image_size[0] / 2**5)
        self.disc_patch = (patch, patch, 1)

        # Build discriminator
        self.d_model = self.build_discriminator(
            image_size=self.image_size, input_channel=self.dis_input_channel)
        self.d_model.compile(loss=self.discriminator_loss,
                             optimizer=opt, metrics=['accuracy'])

        # Build generator
        self.gen_model = self.build_generator(
            image_size=self.image_size, input_channel=self.gen_input_channel, output_channel=self.gen_output_channel)

        self.d_model.trainable = False

        input_src_image = Input(shape=(
            self.image_size[0], self.image_size[1], self.gen_input_channel))  # Input = L channel input
        gen_out = self.gen_model(input_src_image)  # return ab channel
    

        # connect the source input and generator output to the discriminator input
        dis_out = self.d_model(gen_out)  # L, ab channel

        # src image as input, generated image and real/fake classification as output
        self.gan_model = Model(
            input_src_image, [dis_out, gen_out], name='gan_model')

        self.gan_model.compile(loss=[self.discriminator_loss], metrics=[
                               'accuracy'], optimizer=opt)
        self.gan_model.summary()


    def generator_loss(self, y_true, y_pred):
        """_summary_

        Args:
            y_true (Tensor, float32 (B,H,W,C)): gt
            y_pred (Tensor, float32 ((B,H,W,C)): dl model prediction
        """
        mae_loss = mean_absolute_error(y_true=y_true, y_pred=y_pred)

        return mae_loss


    def discriminator_loss(self, y_true, y_pred):
        # Calculate bce Loss
        dis_loss = tf.reduce_mean(binary_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=True))

        return dis_loss


    def build_generator(self, image_size: tuple, input_channel: int, output_channel: int):
        unet = Unet(image_size=image_size,
                    input_channel=input_channel, output_channel=output_channel)
        model = unet.build_generator()

        return model

    def build_discriminator(self, image_size: tuple, input_channel: int):
        kernel_weights_init = RandomNormal(stddev=0.02)
        src_image_shape = (image_size[0], image_size[1], input_channel)
        input_src_image = Input(shape=src_image_shape)

        # C32
        d = Conv2D(32, (4, 4), strides=(2, 2), padding='same',
                   use_bias=True, kernel_initializer=kernel_weights_init)(input_src_image)
        d = LeakyReLU(alpha=0.2)(d)

        # C64
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                   use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C128
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                   use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C256
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same',
                   use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C512
        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
                   use_bias=False, kernel_initializer=kernel_weights_init)(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = LeakyReLU(alpha=0.2)(d)

        # for patchGAN
        output = Conv2D(1, (4, 4), strides=(1, 1), padding='same',
                        use_bias=True, kernel_initializer=kernel_weights_init)(d)

        # define model
        model = Model(input_src_image, output, name='discriminator_model')
        return model


    def calc_metric(self, y_true, y_pred, method='mae'):
        if method == 'mae':
            metric = mean_absolute_error(y_true, y_pred)
        metric = tf.reduce_mean(metric)
        return metric


    
    def save_weights(self, save_path: str, epoch: int, save_gen=True, save_dis=True, save_gan=True):
        if save_gen:
            self.gen_model.save_weights(save_path + '_gen_'+ str(epoch) + '.h5', overwrite=True)
        if save_dis:
            self.d_model.save_weights(save_path + '_dis_'+ str(epoch) + '.h5', overwrite=True)
        if save_gan:
            self.gan_model.save_weights(save_path + '_gan_'+ str(epoch) + '.h5', overwrite=True)


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train()
