import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def simulate_sensor_data(n_samples=2000):
    """
    Simulates accelerometer data for normal driving and accident cases.
    Normal driving: values centered near gravity on Z axis, small noise.
    Accident: large spikes on accelerometer axes.
    """
    data = []
    labels = []
    for _ in range(n_samples):
        if np.random.rand() > 0.1:  # 90% normal driving
            accel_x = np.random.normal(0, 0.3)
            accel_y = np.random.normal(0, 0.3)
            accel_z = np.random.normal(9.8, 0.5)  # gravity approx.
            label = 0
        else:  # 10% accident
            accel_x = np.random.uniform(-20, 20)
            accel_y = np.random.uniform(-20, 20)
            accel_z = np.random.uniform(-20, 20)
            label = 1
        data.append([accel_x, accel_y, accel_z])
        labels.append(label)
    df = pd.DataFrame(data, columns=['accel_x', 'accel_y', 'accel_z'])
    df['label'] = labels
    return df

def train_model(df):
    """
    Train a Random Forest model on accelerometer data.
    """
    X = df[['accel_x', 'accel_y', 'accel_z']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    return clf

def detect_accident(clf, sensor_data):
    """
    Predict if the sensor data corresponds to an accident.
    sensor_data: list or array [accel_x, accel_y, accel_z]
    Returns True if accident detected, else False.
    """
    pred = clf.predict([sensor_data])[0]
    return pred == 1

if __name__ == "__main__":
    print("Simulating sensor data...")
    df = simulate_sensor_data()

    print("Training model...")
    model = train_model(df)

    # Example real-time sensor input (simulate)
    test_data = [
        [0.1, -0.2, 9.7],        # Normal driving data
        [15, -18, 20],           # Accident data
        [0, 0.1, 9.8],           # Normal driving
        [-19, 16, -20]           # Accident
    ]

    for i, sample in enumerate(test_data, 1):
        result = detect_accident(model, sample)
        print(f"Test sample {i}: {sample} --> Accident Detected? {'Yes' if result else 'No'}")
