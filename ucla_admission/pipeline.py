from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
import tensorflow as tf

def create_keras_model(input_shape):
    """Defines the Keras neural network model structure."""
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Input(shape=input_shape),
        
        
        tf.keras.layers.Dense(128, activation='relu'),
        
        
        tf.keras.layers.Dense(64, activation='relu'),
        
        
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


admission_pipeline = Pipeline([
   
    ('scaler', StandardScaler()),
    
   
    ('regressor', KerasRegressor(
        model=create_keras_model,
        input_shape=(7,),      
        epochs=100,            
        batch_size=32,         
        verbose=1,
        random_state=35            
    ))
])
