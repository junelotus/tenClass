import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
# %matplotlib inline
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import tensorflow_addons as tfa
import numpy as np

# 图此读取&解码函数
class DeBlur:
            path_to_deblur = ''
            dataset =  None
            buffer_size = 1
            img = ''
            def __init__(self,buffer_size,esc,ebc,eb,gen):
                self.esc_encoder = tf.keras.models.load_model(esc)#'first/esc_encoder_6.h5')
                self.ebc_encoder = tf.keras.models.load_model(ebc)#'first/ebc_encoder_6.h5')
                self.eb_encoder = tf.keras.models.load_model(eb)#'first/eb_encoder_6.h5')
                self.gen_s = tf.keras.models.load_model(gen)
                self.buffer_size = buffer_size
            
            # @staticmethod
            def load_img(self):
                img = tf.image.resize(DeBlur.img,(256,256)) # resize 会使得在后续显示的时候不是none none
                img = tf.cast(img, tf.float32)/127.5 - 1 #img/127.5-1
                return img

            def set_buffer_size(self,buffer_size):
                DeBlur.buffer_size = buffer_size

            def set_path_or_image_to_deblur(self,path_to_deblur,img):
                DeBlur.path_to_deblur = path_to_deblur
                DeBlur.img = img

            def set_dateset(self,images=None):
                train_b = tf.data.Dataset.from_tensor_slices((DeBlur.img))
                train_b = train_b.map(DeBlur.load_img,num_parallel_calls = AUTOTUNE).batch(DeBlur.buffer_size)#.cache().batch(DeBlur.buffer_size)#.shuffle(DeBlur.buffer_size).
                DeBlur.dataset = tf.data.Dataset.zip((train_b))

            def downsample(self,filters,size, apply_batchnorm=True):# 卷积核个数和大小
                result = keras.Sequential()#顺序模型的创建
                result.add(keras.layers.Conv2D(filters,size,strides = 2,padding='same',use_bias=False)
                        )
                if apply_batchnorm:
                    result.add(tfa.layers.InstanceNormalization())
                    result.add(keras.layers.LeakyReLU())
                return result



            def upsample(self,filters,size,drop=False):# 卷积核个数和大小,上采样过程中 为了过拟合 添加drop
                result = keras.Sequential()#顺序模型的创建
                result.add(keras.layers.Conv2DTranspose(filters,size,strides = 2,padding='same',
                                                    use_bias=False))
                result.add(tfa.layers.InstanceNormalization())
                if drop:
                    result.add(keras.layers.Dropout(0.5))   
                result.add(keras.layers.ReLU())
                return result

            def show_image(self,model,test_input,aa):
                prediction = test_input#model(test_input,training=False)
                plt.figure(figsize =(15,15))
                display_list = [aa[0],prediction[0]]
                titles = ['input','predicted']
                for i in range(2):
                    plt.subplot(1,2,i+1)
                    plt.title(titles[i])
                    plt.imshow(display_list[i]*0.5+0.5)
                    plt.axis('off')
                plt.show()

            #因为是一个循环的调用 密集的运算，可以定义为一个图运算
            @tf.function
            def train_step(self,image_b):
                ebc = self.ebc_encoder(image_b,training=False)# 模糊女人neromh
                eb = self.eb_encoder(image_b,training=False)# 女人的模糊
                gen_s_input = tf.keras.layers.concatenate([ebc, eb]) # 模糊女人+女人模糊 -》gen_s
                fake_s = self.gen_s(gen_s_input, training=False) #清晰女人  
                return gen_s_input,fake_s 

            def to_get_result(self):  
                    iterator = DeBlur.dataset.__iter__()
                    b = iterator.next()
                    gen_s_input,fake_s = self.train_step(b)
                    # self.show_image(self.gen_s,fake_s,b)
                    return fake_s