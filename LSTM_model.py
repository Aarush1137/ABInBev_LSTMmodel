import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
file_path = r'E:\abinbev\hoegarden.csv'
sales_data = pd.read_csv(file_path)

# Extract sales data and city names
sales = sales_data.filter(like='hoe_')  # Only select columns starting with 'hoe_'
cities = sales_data['City'].values

# Normalize the data
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)

# Split the data into training and testing sets
train_size = int(len(sales_scaled) * 0.8)
train_data, test_data = sales_scaled[:train_size], sales_scaled[train_size:]

def create_sequences(data, time_steps=4):
    """
    Create sequences for LSTM model with multiple features
    """
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:(i + time_steps)]
        sequences.append(seq)
    return np.array(sequences)

# Define parameters
time_steps = 4
n_features = sales.shape[1]  # Number of time periods in your data

# Create sequences for training and testing
X_train = create_sequences(train_data, time_steps)
X_test = create_sequences(test_data, time_steps)

# Define target variables (next time step values)
y_train = train_data[time_steps:]
y_test = test_data[time_steps:]

# Build the LSTM model
model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, 
         input_shape=(time_steps, n_features)),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_features)  # Output layer matches the number of features
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Save the model in .keras format
model.save('lstm_hoe.keras')

# Evaluate the model
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTraining Loss: {train_loss:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")