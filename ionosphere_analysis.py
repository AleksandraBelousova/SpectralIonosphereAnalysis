import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import xgboost as xgb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ionosphere_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def load_data(file_path):
    try:
        converters = {
            'column_a': lambda x: 1 if x == 'true' else 0,
            'column_b': lambda x: 1 if x == 'true' else 0
        }
        df = pd.read_csv(file_path, header=0, converters=converters)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].map({'g': 1, 'b': 0}).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Data successfully loaded and normalized.")
        return X_scaled, y
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
    
def extract_spectral_features(X):
    n_pulses = 17
    X_complex = X[:, ::2] + 1j * X[:, 1::2]  
    X_fft = np.fft.fft(X_complex, axis=1) 
    X_spectral = np.abs(X_fft)
    logger.info("Spectral features successfully extracted.")
    return X_spectral

def train_and_evaluate(X, y, model_name="Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring='accuracy')
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"{model_name} - Cross-validation accuracy: {scores.mean():.3f} (±{scores.std():.3f})")
    logger.info(f"{model_name} - Test Precision: {precision:.3f}, Test Recall: {recall:.3f}, Test F1-score: {f1:.3f}")
    return ensemble

def plot_spectral_analysis(X_spectral, y):
    good_mask = y == 1
    mean_spectrum_good = X_spectral[good_mask].mean(axis=0)
    mean_spectrum_bad = X_spectral[~good_mask].mean(axis=0)
    std_spectrum_good = X_spectral[good_mask].std(axis=0) / np.sqrt(good_mask.sum())
    std_spectrum_bad = X_spectral[~good_mask].std(axis=0) / np.sqrt((~good_mask).sum())
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_spectrum_good, label='Good returns', color='blue')
    plt.fill_between(range(len(mean_spectrum_good)), 
                     mean_spectrum_good - 1.96 * std_spectrum_good, 
                     mean_spectrum_good + 1.96 * std_spectrum_good, 
                     color='blue', alpha=0.2)
    plt.plot(mean_spectrum_bad, label='Bad returns', color='red')
    plt.fill_between(range(len(mean_spectrum_bad)), 
                     mean_spectrum_bad - 1.96 * std_spectrum_bad, 
                     mean_spectrum_bad + 1.96 * std_spectrum_bad, 
                     color='red', alpha=0.2)
    plt.xlabel('Frequency index')
    plt.ylabel('Amplitude')
    plt.title('Average Spectra with 95% Confidence Intervals')
    plt.legend()
    plt.grid()
    plt.savefig('spectral_analysis_final.png', dpi=300)
    plt.close()
    logger.info("Graph saved as 'spectral_analysis_final.png'")

def main():
    file_path = "ionosphere_data.csv"
    logger.info("Start of program execution.")
    
    X, y = load_data(file_path)
    if X is None or y is None:
        logger.error("Program terminated due to data loading error.")
        return
    
    X_spectral = extract_spectral_features(X)
    
    X_combined = np.hstack((X, X_spectral))
    logger.info("Combined features created.")
    
    logger.info("Training the model on the original data:")
    train_and_evaluate(X, y, "Initial indications")
    
    logger.info("Training the model on spectral features:")
    train_and_evaluate(X_spectral, y, "Spectral features")
    logger.info("Training the model on combined features:")
    
    train_and_evaluate(X_combined, y, "Combined features")
    plot_spectral_analysis(X_spectral, y)
    logger.info("Program successfully completed.")
if __name__ == "__main__":
    main()