#import all necessary modules
from numpy import zeros, ones
from numpy.random import randn, randint
from keras.datasets.cifar10 import load_data
from keras.optimizer_v2.adam import Adam
from keras.models import Sequential 
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from matplotlib import pyplot
import keras
# define the standalone discriminator model
def create_discriminator(current_epoch_shape=(32,32,3)):
    discriminator = Sequential()
    # normal
    discriminator.add(Conv2D(64, (3,3), padding='same', input_shape=current_epoch_shape))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # downsample
    discriminator.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    # classifier
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return discriminator
 
# define the standalone generator model
def create_generator(latent_dim):
    generator = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    generator.add(Dense(n_nodes, input_dim=latent_dim))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # output layer
    generator.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return generator
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # combine the two together
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt)
    return gan
 
# load and prepare cifar10 training images
def get_real_samples():
    # load cifar10 dataset
    (images, _), (_, _) = load_data()
    # convert from intigers to floats
    images = images.astype('float32')
    # scale from [0,255] to [-1,1]
    scaled_images = (images - 127.5) / 127.5
    return scaled_images
 
# select real samples
def get_dataset_instances(dataset, n_samples):
    # choose random instances
    counter = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    instances = dataset[counter]
    # generate 'real' class labels of one
    real_label = ones((n_samples, 1))
    return instances, real_label
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    generator_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    generator_input = generator_input.reshape(n_samples, latent_dim)
    return generator_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    inputs = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    generated_images = generator.predict(inputs)
    # create 'fake' class labels of 0
    fake_label = zeros((n_samples, 1))
    return generated_images, fake_label
 
# create and save a plot of generated images
def save_plot(examples, epoch, plot_dims=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(plot_dims * plot_dims):
        # create subplot
        pyplot.subplot(plot_dims, plot_dims, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    pyplot.savefig("/content/drive/MyDrive/generated_plot_e%03d.png" % (epoch+1))
    pyplot.close()
 
# evaluate the discriminator, save generator model, save weights
def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=150):
    # prepare real instances
    instances, real_label = get_dataset_instances(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(instances, real_label, verbose=0)
    # generate fake examples
    generated_images, fake_label = generate_fake_samples(generator, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(generated_images, fake_label, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save weights
    generator.save_weights(
       '/content/drive/MyDrive/params_generator_epoch_%03d.hdf5' % (epoch+1), True)
    discriminator.save_weights(
        '/content/drive/MyDrive/params_discriminator_epoch_%03d.hdf5' % (epoch+1), True)
    # save the generator model to file
    generator.save('/content/drive/MyDrive/generator_model_%03d.h5' % (epoch+1))
 
# train the gan
def train(generator, discriminator, gan, dataset, latent_dim, epochs_completed, summarization, batchloss, n_epochs, n_batch, save_freq):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # required if starting at already progressed model
    if epochs_completed > 0:
        n_epochs = n_epochs - epochs_completed
    # one epoch is signified by 390 batches being complete
    for i in range(n_epochs):
        # required if starting at already progressed model
        if epochs_completed > 0:
            current_epoch = i + epochs_completed
        else:
            current_epoch = i
        # every batch the gan get trained
        for j in range(bat_per_epo):
            # get randomly selected real instances
            instances, real_label = get_dataset_instances(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = discriminator.train_on_batch(instances, real_label)
            # generate fake examples
            generated_images, fake_label = generate_fake_samples(generator, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = discriminator.train_on_batch(generated_images, fake_label)
            # prepare points in latent space as input for the generator
            generator_input = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            target_labels = ones((n_batch, 1))
            # update the generator via the discriminator's error in the whole GAN
            g_loss = gan.train_on_batch(generator_input, target_labels)
            # summarize loss on this batch if requirments are met
            if (j+1) % batchloss == 0:
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (current_epoch, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # save plot if requirments are met
        if (current_epoch+1) % save_freq == 0:
            save_plot(generated_images, current_epoch, n=7)
        # evaluate the model performance, save model and weights if requirments are met
        if (current_epoch+1) % summarization == 0:
            summarize_performance(current_epoch, generator, discriminator, dataset, latent_dim)
 
# control panel
latent_dim = 100
summarization = 5
batchloss = 39
n_epochs = 500
n_batch = 128
save_freq = 0

#resume training
load_weights = False
epochs_completed = 0

# create the discriminator, generator and combines
discriminator = create_discriminator()
generator = create_generator(latent_dim)
gan = define_gan(generator, discriminator)

# load image data
dataset = get_real_samples()
#load weights if resuming training
if load_weights:
        generator.load_weights('/content/drive/MyDrive/params_generator_epoch_%03d.hdf5' % epochs_completed)
        discriminator.load_weights('/content/drive/MyDrive/params_discriminator_epoch_%03d.hdf5' % epochs_completed)
# train model
train(generator, discriminator, gan, dataset, latent_dim, epochs_completed, summarization, batchloss, n_epochs, n_batch, save_freq)