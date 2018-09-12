# -*- coding: utf-8 -*-
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, MaxPooling2D
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
import scipy.io as sio

#from skimage.color import rgb2gray
from skimage import color, exposure, transform
from keras.datasets import mnist
#from keras.utils import np_utils
from keras import backend as K
from keras.utils.np_utils import to_categorical

import scipy.misc
from PIL import Image
from scipy.misc import imresize
from keras.models import load_model
from numpy import random

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=32, img_cols=32, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel  = channel
        self.D_F  = None   # discriminator
        self.G_F  = None   # generator
        self.AM_F = None  # adversarial model
        self.DM_F = None  # discriminator model

        self.D_G  = None   # discriminator
        self.G_G  = None   # generator
        self.AM_G = None  # adversarial model
        self.DM_G = None  # discriminator model
	self.C_1  = None  # cyclegan


    def discriminator_F(self):
        if self.D_F:
            return self.D_F
        self.D_F = Sequential()


        self.D_F.add(Conv2D(64, (5,5), padding='same', strides=(2,2), 
                            input_shape=(32, 32, 1)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))

        self.D_F.add(Conv2D(32, (3,3), padding='same', strides=(2,2)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))

        self.D_F.add(Conv2D(16, (3,3), padding='same', strides=(2,2)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))
       
        self.D_F.add(Conv2D(8, (3,3), padding='same', strides=(2,2)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))

        self.D_F.add(Conv2D(4, (3,3), padding='same', strides=(2,2)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))

        self.D_F.add(Conv2D(2, (3,3), padding='same', strides=(2,2)))
        self.D_F.add(BatchNormalization())
        self.D_F.add(LeakyReLU(alpha=0.2))

        self.D_F.add(Conv2D(1, (3,3), padding='same'))
        self.D_F.add(BatchNormalization())

        self.D_F.add(Flatten())
        self.D_F.add(Dense(1, activation='sigmoid'))

	print('DISCRIMINATOR F')
        return self.D_F


    def discriminator_G(self):
        if self.D_G:
            return self.D_G
        self.D_G = Sequential()

        self.D_G.add(Conv2D(64, (5,5), padding='same', strides=(2,2), 
                            input_shape=(32, 32, 1)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(32, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(16, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(8, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(4, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(2, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())
        self.D_G.add(LeakyReLU(alpha=0.2))

        self.D_G.add(Conv2D(1, (3,3), padding='same', strides=(2,2)))
        self.D_G.add(BatchNormalization())

        self.D_G.add(Flatten())
        self.D_G.add(Dense(11, activation='sigmoid'))

        return self.D_G


    def generator_F(self):
        if self.G_F:
            return self.G_F
        self.G_F = Sequential()

        self.G_F.add(Conv2D(128, (5, 5), padding='same',input_shape=(32, 32, 1)))
        self.G_F.add(BatchNormalization())
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(64, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(32, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(8, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(4, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(2, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))
        self.G_F.add(Conv2D(1, (3,3), padding='same'))
        self.G_F.add(Activation('tanh'))

	print('GENERATOR F')
        return self.G_F


    def generator_G(self):
        if self.G_G:
            return self.G_G
        self.G_G = Sequential()

        self.G_G.add(Conv2D(128, (5, 5), padding='same',input_shape=(32, 32, 1)))
        self.G_G.add(BatchNormalization())
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(64, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(32, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(16, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(8, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(4, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(2, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))
        self.G_G.add(Conv2D(1, (3,3), padding='same'))
        self.G_G.add(Activation('tanh'))

	print('GENERATOR G')
        return self.G_G


    def discriminator_model_F(self):
        if self.DM_F:
            return self.DM_F
        self.DM_F = Sequential()
        self.DM_F.add(self.discriminator_F())
        self.DM_F.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

        return self.DM_F

    def discriminator_model_G(self):
        if self.DM_G:
            return self.DM_G
	sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.DM_G = Sequential()
        self.DM_G.add(self.discriminator_G())
        self.DM_G.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
        return self.DM_G


    # CHANGE THE LOSS HERE
    def adversarial_model_F(self):
        if self.AM_F:
            return self.AM_F
        self.AM_F = Sequential()
        self.AM_F.add(self.generator_F())
        self.AM_F.add(self.discriminator_F())
        self.AM_F.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
        return self.AM_F

    def adversarial_model_G(self):
        if self.AM_G:
            return self.AM_G
        self.AM_G = Sequential()
        self.AM_G.add(self.generator_G())
        self.AM_G.add(self.discriminator_G())
        self.AM_G.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
        return self.AM_G

    def cyclegan_model_1(self):
	if self.C_1:
            return self.C_1
        self.C_1 = Sequential()
        self.C_1.add(self.generator_G())
        self.C_1.add(self.generator_F())
	self.C_1.compile(loss='mean_absolute_error', optimizer=SGD(), metrics=['accuracy'])
        return self.C_1
	

class MNIST_DCGAN(object):


    def __init__(self):
        self.img_rows_1 = 32
        self.img_cols_1 = 32
        self.channel_1 = 1

	# MNIST
	(x_train_mnist, self.y_train_mnist), (x_test_mnist, self.y_test_mnist) = mnist.load_data()
	# MNIST TRAINING
	d=[]
	for i in range(x_train_mnist.shape[0]):
	    mnist_img = x_train_mnist[i]
	    mnist_img = np.reshape(mnist_img, (28,28))
	    img = imresize(mnist_img, (32, 32))
	    img = img/255.0
	    d.append(img)

	x_train_mnist = np.asarray(d)
	x_train_mnist /= 255
	self.x_train_mnist = x_train_mnist.reshape(60000, 32,32,1)
	self.y_train_mnist = np.expand_dims(self.y_train_mnist,axis=1)
	# MNIST TESTING
	d=[]
	for i in range(x_test_mnist.shape[0]):
	    mnist_img = x_test_mnist[i]
	    mnist_img = np.reshape(mnist_img, (28,28))
	    img = imresize(mnist_img, (32, 32))
	    img = img/255.0
	    d.append(img)

	x_test_mnist = np.asarray(d)
	x_test_mnist /= 255
	self.x_test_mnist = x_test_mnist
	self.x_test_mnist = x_test_mnist.reshape(10000, 32,32,1)
	self.y_test_mnist = np.expand_dims(self.y_test_mnist,axis=1)

	# SVHN TRAINING
	train = sio.loadmat("train_32x32.mat")
        self.y_train_svhn = train['y']
        train_examples = train['X'].shape[3]
	d = []
	data = train['X']
        for i in range(train_examples):
	    rgb = data[:, :, :, i]
	    gray = color.rgb2gray(rgb)
	    d.append(gray)

	x_train_svhn = np.asarray(d)


	x_train_svhn = x_train_svhn.astype("float32")
	self.x_train_svhn = x_train_svhn.reshape(73257, 32,32,1)
	
	# SVHN TESTING
	test = sio.loadmat("test_32x32.mat")
        self.y_test_svhn = test['y']
        test_examples = test['X'].shape[3]
	d = []
	data = test['X']
        for i in range(test_examples):
	    rgb = data[:, :, :, i]

	    gray = color.rgb2gray(rgb)

	    d.append(gray)

	x_test_svhn = np.asarray(d)
	x_test_svhn = x_test_svhn.astype("float32")
	self.x_test_svhn = x_test_svhn
	self.x_test_svhn = x_test_svhn.reshape(26032, 32,32,1)

	# Labels 10 to 0
	self.y_train_svhn[self.y_train_svhn == 10] = 0
	self.y_test_svhn[self.y_test_svhn == 10]   = 0

#############################################################################################

        self.DCGAN = DCGAN()
        self.discriminator_F = self.DCGAN.discriminator_model_F()
        self.adversarial_F   = self.DCGAN.adversarial_model_F()
        self.generator_F     = self.DCGAN.generator_F()

        self.discriminator_G = self.DCGAN.discriminator_model_G()
        self.adversarial_G   = self.DCGAN.adversarial_model_G()
        self.generator_G     = self.DCGAN.generator_G()

	self.cyclegan_1      = self.DCGAN.cyclegan_model_1()


    def train(self, train_steps=2000, batch_size=256, save_interval=0, printing=True):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):

	    rndm_mnist = np.random.randint(0, self.x_train_mnist.shape[0], size=batch_size)
	    rndm_svhn  = np.random.randint(0, self.x_train_svhn.shape[0], size=batch_size)

	    mnist_imgs = self.x_train_mnist[rndm_mnist]
	    svhn_imgs  = self.x_train_svhn[rndm_svhn]

            #DIRECTION 1###############################################################
	    print('D', i)

	    d_loss_3 = self.cyclegan_1.train_on_batch(svhn_imgs, svhn_imgs)

	    #GENERATOR G--------------------------------------
            fake_mnist_imgs = self.generator_G.predict(svhn_imgs)
	    # DISCRIMINATOR G---------------------------------------
            all_mnist_imgs = np.concatenate((mnist_imgs, fake_mnist_imgs))
            y_1 = np.ones([2*batch_size, 1])
            y_1[0:batch_size, :] = self.y_train_mnist[rndm_mnist]
            y_1[batch_size:, :] = 10
	    cat_labels = to_categorical(y_1, num_classes=11)
            d_loss_1 = self.discriminator_G.train_on_batch(all_mnist_imgs, cat_labels)
	    # ADVERSARIAL G----------------------------------------------------
            y_1 = np.ones([batch_size, 1])*(10)
	    cat_labels = to_categorical(y_1, num_classes=11)
            a_loss_1 = self.adversarial_G.train_on_batch(svhn_imgs, cat_labels)

            #DIRECTION F######################################################
	    #print('DIRECTION 2', i)
	    # GENERATOR F-------------------------------------
            fake_svhn_imgs = self.generator_F.predict(mnist_imgs)
	    # DISCRIMINATOR F-----------------------------------------
            all_svhn_imgs = np.concatenate((svhn_imgs, fake_svhn_imgs))
            y_2 = np.ones([2*batch_size, 1])
            y_2[batch_size:, :] = 0
            d_loss_2 = self.discriminator_F.train_on_batch(all_svhn_imgs, y_2)
	    # ADVERSARIAL_F--------------------------------------------
            y_2 = np.ones([batch_size, 1])*0
            a_loss_2 = self.adversarial_F.train_on_batch(mnist_imgs, y_2)

	    #LOGGING
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss_1[0], d_loss_1[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss_1[0], a_loss_1[1])
            log_mesg_2 = "%d: [D loss: %f, acc: %f]" % (i, d_loss_2[0], d_loss_2[1])
            log_mesg_2 = "%s  [A loss: %f, acc: %f]" % (log_mesg_2, a_loss_2[0], a_loss_2[1])
	    log_mesg_3 = "%d: [D loss: %f, acc: %f]" % (i, d_loss_3[0], d_loss_3[1])

            print(log_mesg)
	    print(log_mesg_2)
	    print(log_mesg_3)
            if save_interval>0:
                if (i+1)%save_interval==0:
		    if printing:
                        self.plot_images(save2file=True, samples=4,\
                            noise=noise_input, step=(i+1))
		    else:
                    	print('SAVING MODELS', i)
                        self.generator_F.save_weights('genFmodel_weights_82.h5')
                        self.generator_G.save_weights('genGmodel_weights_82.h5')

    def train_other(self, train_steps=2000, batch_size=256, save_interval=0):

	a_model = Sequential()

        a_model.add(Conv2D(32, (5,5), padding='same', 
                            input_shape=(32, 32, 1),
                            activation='relu'))
        a_model.add(MaxPooling2D(pool_size=(2, 2)))
        a_model.add(Conv2D(15, (3, 3), activation='relu'))
        a_model.add(MaxPooling2D(pool_size=(2, 2)))
        a_model.add(Dropout(0.2))

        a_model.add(Flatten())
        a_model.add(Dense(128, activation='relu'))
        a_model.add(Dense(50, activation='relu'))
        a_model.add(Dense(10, activation='softmax'))

        a_model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

	#--------------------------------------------------------------------------------------------------

	# Train the classifier model
	for i in range(train_steps):
	    print('t', i)

	    # Get batch
	    rndm_svhn = np.random.randint(0, self.x_train_svhn.shape[0], size=batch_size)
	    svhn_data = self.x_train_svhn[rndm_svhn]

	    # Project svhn images to mnist
	    fake_mnist_imgs = self.generator_G.predict(svhn_data)

	    # Put labels in correct form
	    svhn_lbls = self.y_train_svhn[rndm_svhn]
	    svhn_lbls = np.squeeze(svhn_lbls,axis=1)
	    print(svhn_lbls.shape, svhn_lbls)
	    cat_svhn_lbls = to_categorical(svhn_lbls, 10)

	    # Train final classifier
	    a_model.train_on_batch(fake_mnist_imgs, cat_svhn_lbls)

        a_model.save_weights('a_model_weights_82.h5')
	print('end train_other')

	correct = 0
	print(self.x_test_svhn.shape)
	correct = 0
	TOTAL = 26032
	start_count = 0
	next_count = 2000
	for i in range(13):

	    # Get the svhn data
	    svhn_data = self.x_test_svhn[start_count:next_count,:,:,:]
	    print('Testing', i, svhn_data.shape)

	    # Project to mnist
	    fake_mnist_imgs = self.generator_G.predict(svhn_data)

	    # Predict label
	    predicted_lbls = a_model.predict(fake_mnist_imgs)

	    # Update correct count
	    predicted_lbls = np.argmax(predicted_lbls, axis=1)

	    svhn_test_lbls = self.y_test_svhn[start_count:next_count,0]

	    correct = correct + np.sum(predicted_lbls == svhn_test_lbls)

	    start_count = next_count
	    next_count = next_count + 2000

	svhn_data = self.x_test_svhn[26000:26032,:,:,:]
	fake_mnist_imgs = self.generator_G.predict(svhn_data)
	predicted_lbls = a_model.predict(fake_mnist_imgs)
	predicted_lbls = np.argmax(predicted_lbls, axis=1)
	svhn_test_lbls = self.y_test_svhn[26000:26032,0]
	correct = correct + np.sum(predicted_lbls == svhn_test_lbls)

	print(correct*1.0/TOTAL)
	exit()

    def plot_images(self, save2file=False, fake=True, samples=1, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
	    print('PLOTTING 111')

            filename = "mnist_%d.png" % step
	    rndm = np.random.randint(0, self.x_train_svhn.shape[0], size=samples)
	    print('rndm', rndm, self.x_train_svhn.shape)
	    sample_images = self.x_train_svhn[rndm, :, :, :]
            images = self.generator_G.predict(sample_images)

	    model = load_model('genF.h5')
	    images2 = self.generator_G.predict(images)
        else:
	    print('PLOTTING 222')
            i = np.random.randint(0, self.x_train_svhn.shape[0], samples)
            images = self.x_train_svhn[i, :, :, :]

        print('IMG', images.shape)
        plt.figure(figsize=(3,samples))
        counter = 1
        for i in range(samples):
	    print('blerta',i, counter)

	    plt.subplot(samples, 3, counter)
	    smpl = sample_images[i, :, :, :]
	    smpl = np.reshape(smpl, [32,32])
	    plt.imshow(smpl, cmap='gray')
            plt.axis('off')

	    counter = counter + 1

            plt.subplot(samples, 3, counter)
            image = images[i, :, :, :]
            image = np.reshape(image, [32,32])
            plt.imshow(image, cmap='gray')
            plt.axis('off')

	    counter = counter + 1

            plt.subplot(samples, 3, counter)
            image = images2[i, :, :, :]
            image = np.reshape(image, [32,32])
            plt.imshow(image, cmap='gray')
            plt.axis('off')

	    counter = counter + 1

        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()



if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()

    mnist_dcgan.train(train_steps=200000, batch_size=64, save_interval=200000, printing=False)
    mnist_dcgan.train_other(train_steps=100000, batch_size=64, save_interval=1)

    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
mnist_dcgan.plot_images(fake=False, save2file=True)
