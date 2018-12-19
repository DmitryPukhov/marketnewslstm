from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LSTM, Embedding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

class ModelFactory:
    """
    Generate different models. Actually only one of them is used in the kernel,
    this factory is for experiments when debugging.
    """
    # LSTM look back window size
    look_back=90
    # In windows size look back each look_back_step days
    look_back_step=10

    def lstm_128(input_size):
        model = Sequential()
        # Add an input layer market + news
        #input_size = len(market_prepro.feature_cols) + len(news_prepro.feature_cols)
        # input_shape=(timesteps, input features)
        model.add(LSTM(units=128, return_sequences=True, input_shape=(None,input_size)))
        model.add(LSTM(units=64, return_sequences=True ))
        model.add(LSTM(units=32, return_sequences=False))

        # Add an output layer
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return(model)

    def train(model, toy, join_generator, val_generator):
        weights_file='best_weights.h5'

        # We'll stop training if no improvement after some epochs
        earlystopper = EarlyStopping(patience=5, verbose=1)

        # Low, avg and high scor training will be saved here
        # Save the best model during the traning
        checkpointer = ModelCheckpoint(weights_file
                                       #,monitor='val_acc'
                                       ,verbose=1
                                       ,save_best_only=True
                                       ,save_weights_only=True)

        #reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.001)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.001)

        # Set fit parameters
        # Rule of thumb: steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
        #                validation_steps = TotalvalidationSamples / ValidationBatchSize
        if toy:
            batch_size=1000
            validation_batch_size=1000
            steps_per_epoch=5
            validation_steps=2
            epochs=2
            look_back=10
            look_back_step=2
        else:
            batch_size=1000
            validation_batch_size=1000
            steps_per_epoch=20
            validation_steps=5
            epochs=20
            look_back=90
            look_back=10

        print(f'Toy:{toy}, epochs:{epochs}, steps per epoch: {steps_per_epoch}, validation steps:{validation_steps}')
        print(f'Batch_size:{batch_size}, validation batch size:{validation_batch_size}')

        # Fit
        training = model.fit_generator(join_generator.flow_lstm(batch_size=batch_size
                                                                , is_train=True
                                                                , look_back=look_back
                                                                , look_back_step=look_back_step)
                                       , epochs=epochs
                                       , validation_data=val_generator.flow_lstm(batch_size=validation_batch_size
                                                                                 , is_train=False
                                                                                 , look_back=look_back
                                                                                 , look_back_step=look_back_step)
                                       , steps_per_epoch=steps_per_epoch
                                       , validation_steps=validation_steps
                                       , callbacks=[earlystopper, checkpointer, reduce_lr])
        # Load best weights saved
        model.load_weights(weights_file)
        return training

# model = ModelFactory.lstm_128()
# model.summary()

# model = ModelFactory.lstm_128()
# model.summary()