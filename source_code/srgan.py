# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input

class SRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):
        # initialize the input layer
        inputs = Input((None, None, 3))
        xIn = Rescaling(scale=(1.0 / 255.0), offset=0.0)(inputs)

        # pass the input through CONV => PReLU block
        xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = PReLU(shared_axes=[1, 2])(xIn)

        # construct the "residual in residual" block
        x = Conv2D(featureMaps, 3, padding="same")(xIn)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        xSkip = Add()([xIn, x])

        # process the feature maps with global average pooling
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leakyAlpha)(x)

        # final FC layer with sigmoid activation function
        x = Dense(1, activation="sigmoid")(x)

        # create the discriminator model
        discriminator = Model(inputs, x)
        
        # return the discriminator
        return discriminator