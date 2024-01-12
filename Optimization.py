from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Step 7: Model Optimization

# a. Hyperparameter Tuning
def hyperparameter_tuning(train_generator, val_generator, input_shape=(48, 48, 3), num_classes=7):
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks (optional but recommended)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=20,  # Adjust the number of epochs based on your dataset
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping],
        batch_size=32
    )

    return model, history

# Example usage
# Assuming you have data generators for training and validation images (train_generator, val_generator)

tuned_model, tuning_history = hyperparameter_tuning(train_generator, val_generator)

# b. Regularization Techniques
def apply_regularization(train_generator, val_generator, input_shape=(48, 48, 3), num_classes=7):
    # Define the model with dropout regularization
    model_with_dropout = Sequential()
    model_with_dropout.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model_with_dropout.add(MaxPooling2D((2, 2)))
    model_with_dropout.add(Conv2D(64, (3, 3), activation='relu'))
    model_with_dropout.add(MaxPooling2D((2, 2)))
    model_with_dropout.add(Conv2D(128, (3, 3), activation='relu'))
    model_with_dropout.add(MaxPooling2D((2, 2)))
    model_with_dropout.add(Flatten())
    model_with_dropout.add(Dense(128, activation='relu'))
    model_with_dropout.add(Dropout(0.5))  # Dropout layer for regularization
    model_with_dropout.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model_with_dropout.compile(optimizer=Adam(learning_rate=0.001),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

    # Train the model with dropout
    history_with_dropout = model_with_dropout.fit(
        train_generator,
        epochs=20,  # Adjust the number of epochs based on your dataset
        validation_data=val_generator,
        batch_size=32
    )

    return model_with_dropout, history_with_dropout

# Example usage
# Assuming you have data generators for training and validation images (train_generator, val_generator)

dropout_model, dropout_history = apply_regularization(train_generator, val_generator)
