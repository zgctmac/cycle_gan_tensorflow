#coding:utf-8
import tensorflow as tf
import layers
is_training = True
batch_size = 1
img_width = 256
img_height = 256
img_layers = 3
class cyclegan(object):
    def _input_setup(self):
        self.dataset_dir_A = "./input/horse2zebra/trainA/*.jpg"
        filenames_A = tf.train.match_filenames_once(self.dataset_dir_A)
        self.dataset_dir_B = "./input/horse2zebra/trainB/*.jpg"
        filenames_B = tf.train.match_filenames_once(self.dataset_dir_B)
        filenames_queue_A = tf.train.string_input_producer(filenames_A)
        filenames_queue_B = tf.train.string_input_producer(filenames_B)
        image_reader = tf.WholeFileReader()
        _,image_file_A = image_reader.read(filenames_queue_A)
        _,image_file_B = image_reader.read(filenames_queue_B)
        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)

    def _build_model(self):
        self.input_A = tf.placeholder(tf.float32,shape=[batch_size,img_height,img_width,img_layers],name='input_A')
        self.input_B = tf.placeholder(tf.float32,shape=[batch_size,img_height,img_width,img_layers],name='input_B')
        self.fake_pool_A = tf.placeholder(tf.float32,shape=[None,img_height,img_width,img_layers],name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32,shape=[None,img_height,img_width,img_layers],name="fale_pool_B")
        with tf.variable_scope("model") as scope:
            self.fake_B = layers.generator(self.input_A,name="g_A")
            self.fake_A = layers.generator(self.input_B,name="g_B")
            self.recA = layers.discriminator(self.input_A,name="D_A")
            self.recB = layers.discriminator(self.input_B,name="D_B")
            scope.reuse_variables()
            self.fake_recA = layers.discriminator(self.fake_A,name="fake_D_A")
            self.fake_recB = layers.discriminator(self.fake_B,name="fake_D_B")
            self.cycA = layers.generator(self.fake_B,name="g_B")
            self.cycB = layers.generator(self.fake_A,name="g_A")
            scope.reuse_variables()
            self.fake_pool_rec_A = layers.discriminator(self.fake_pool_A,name="d_A")
            self.fake_pool_rec_B = layers.discriminator(self.fake_pool_B,name="d_B")

    def _loss(self):
        pass
    def train(self):
        pass
    def test(self):
        pass

def main():
    model = cyclegan()
    if is_training:
        model.train()
    else:
        model.test()
if __name__=="__main__":
    tf.app.run()



