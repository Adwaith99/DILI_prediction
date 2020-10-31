from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class Resnet:
    @staticmethod
    def residual_module(data, K, stride, activation, kernel_size, chanDim=-1, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9, i=-1, j=-1):
        # the shortcut branch of a residual module is initialized as the input
        shortcut = data

        # First block of resudual module
        bn1 = BatchNormalization(
            epsilon=bnEps, momentum=bnMom, name='bn1_'+str(i)+'_'+str(j))(data)
        act1 = Activation(
            activation, name='act1_'+str(i)+'_'+str(j))(bn1)
        conv1 = Conv1D(K, kernel_size, use_bias=False, kernel_regularizer=l2(
            reg), padding='same', name='conv1_'+str(i)+'_'+str(j))(act1)

        # 2nd block of residual module
        bn2 = BatchNormalization(
            epsilon=bnEps, momentum=bnMom, name='bn2_'+str(i)+'_'+str(j))(conv1)
        act2 = Activation(
            activation, name='act2_'+str(i)+'_'+str(j))(bn2)
        conv2 = Conv1D(K, kernel_size, use_bias=False, kernel_regularizer=l2(
            reg), padding='same', name='conv2_'+str(i)+'_'+str(j))(act2)

        # In the beginning of each stage of the resnet you will have adjust the shortcut dimensions
        if red:
            shortcut = Conv1D(K, kernel_size, use_bias=False, kernel_regularizer=l2(
                reg), padding='same', name='conv_srt_'+str(i)+'_'+str(j))(act1)

        # add the second residual moduleand the shortcut
        x = add([conv2, shortcut],
                name='add_'+str(i)+'_'+str(j))

        return x

    @staticmethod
    def build(input_shape, classes, stages, filters, activation, fc_layers, kernel_size=9, stride=1, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # create input and apply BN
        inputs = Input(shape=input_shape)
        x = BatchNormalization(
            epsilon=bnEps, momentum=bnMom, name='bn_input')(inputs)
        x = Reshape((2325, 1), name='Reshape')(x)

        # Create the starting conv layer
        x = Conv1D(filters[0], kernel_size, use_bias=False, padding='same',
                   kernel_regularizer=l2(reg), name='conv_first_layer')(x)
        x = BatchNormalization(
            epsilon=bnEps, momentum=bnMom, name='bn_first_layer')(x)
        x = Activation(
            activation, name='act_first_layer')(x)

        # loop over the stages
        for i in range(0, len(stages)):
            # first layer in each stage should reduce the dimensions of input
            if i == 0 and filters[0] == filters[1]:
                x = Resnet.residual_module(
                    x, filters[i+1], stride, activation, kernel_size, bnEps=bnEps, bnMom=bnMom, i=i+1, j=1)
            else:
                x = Resnet.residual_module(
                    x, filters[i+1], stride, activation, kernel_size, red=True, bnEps=bnEps, bnMom=bnMom, i=i+1, j=1)

            for j in range(0, stages[i] - 1):
                # apply a renset module
                x = Resnet.residual_module(
                    x, filters[i+1], stride, activation, kernel_size, bnEps=bnEps, bnMom=bnMom, i=i+1, j=j+2)

        x = BatchNormalization(
            epsilon=bnEps, momentum=bnMom, name='bn_last_layer')(x)
        x = Activation(activation)(x)

        # Final dense layer applied with softmax activation
        x = Flatten(name='flatten')(x)
        x = Dense(fc_layers, kernel_regularizer=l2(
            reg), name='out_dense_layer')(x)
        x = Dense(classes, activation='sigmoid',
                  name='last_layer', kernel_regularizer=l2(reg))(x)

        # create the model
        model = Model(inputs, x, name='resnet')

        # return the model
        return model

#The model can be created by calling Resnet.build() with the required parameters
       
