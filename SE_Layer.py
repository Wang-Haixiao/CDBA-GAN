from keras.layers import Layer,Dense,GlobalAveragePooling2D,Multiply,\
    Conv2D,Reshape,GlobalMaxPooling2D,Add,Activation


class SeNetBlock(Layer):

    def __init__(self, reduction = 4, **kwargs):
        self.reduction = reduction
        super(SeNetBlock, self).__init__(**kwargs)

    # def senet(self, input):
    def call(self, x):
        channels = x.shape.as_list()[-1]
        avg_x = GlobalAveragePooling2D()(x)
        avg_x = Reshape((1, 1, channels))(avg_x)
        avg_x = Conv2D(int(channels) // self.reduction, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                       activation='relu')(avg_x)
        avg_x = Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), padding='valid')(avg_x)

        max_x = GlobalMaxPooling2D()(x)
        max_x = Reshape((1, 1, channels))(max_x)
        max_x = Conv2D(int(channels) // self.reduction, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                       activation='relu')(max_x)
        max_x = Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), padding='valid')(max_x)

        cbam_feature = Add()([avg_x, max_x])

        cbam_feature = Activation('sigmoid')(cbam_feature)

        x = Multiply()([x, cbam_feature])

        return x