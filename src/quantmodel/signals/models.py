# analysis/signals/models.py

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime
from quantmodel.utils.logger import get_logger
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
logger = get_logger(__name__)


class ModelSelector:
    """Handles model selection and optimization"""

    AVAILABLE_MODELS = {
        'random_forest': {
            'class': RandomForestClassifier,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 5,                 # Added limit
                'min_samples_split': 20,        # Increased
                'min_samples_leaf': 10,         # Increased
                'max_features': 'sqrt',         # Added
                'class_weight': 'balanced',
                'bootstrap': True,              # Added
                'random_state': 42,
                'oob_score': True              # Added
            }
        },
        'gradient_boost': {
            'class': GradientBoostingClassifier,
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.05,          # Reduced for better generalization
                'max_depth': 3,
                'min_samples_split': 20,        # Increased
                'min_samples_leaf': 10,         # Increased
                'subsample': 0.8,               # Added
                'random_state': 42,
                'validation_fraction': 0.1      # Added
            }
        },
        'logistic': {
            'class': LogisticRegression,
            'default_params': {
                'C': 0.1,                       # Reduced for more regularization
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs',              # Added
                'multi_class': 'ovr',           # Added
                'penalty': 'l2'                 # Added
            }
        },
        'svm': {
            'class': SVC,
            'default_params': {
                'C': 0.1,                       # Reduced for more regularization
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True,            # Added for predict_proba
                'gamma': 'scale',               # Added
                'cache_size': 1000              # Added
            }
        }
    }

    @classmethod
    def get_model(cls, model_type: str, custom_params: Optional[Dict] = None):
        """Get model instance with specified parameters"""
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(
                f"Model type {model_type} not supported. Available models: {list(cls.AVAILABLE_MODELS.keys())}")

        model_config = cls.AVAILABLE_MODELS[model_type]
        model_class = model_config['class']
        params = model_config['default_params'].copy()

        if custom_params:
            params.update(custom_params)

        return model_class(**params)

class SignalPredictor:
    def __init__(self, config):
        self.config = config

        # Initialize base model with conservative parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=42,
            oob_score=True
        )

        # Initialize cross-validation and scaling
        self.cv = TimeSeriesSplit(n_splits=5)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.model_metrics = {}

    def prepare_features(self, features: pd.DataFrame, training: bool = False) -> np.ndarray:
        """Scale and prepare features for model"""
        self.feature_names = features.columns

        if training or not self.is_fitted:
            X = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            X = self.scaler.transform(features)

        return X

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with confidence scores"""
        try:
            if not self.is_fitted:
                raise ValueError("Model is not fitted yet. Call train() first.")

            X = self.prepare_features(features, training=False)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)

            return predictions, confidence_scores

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def train(self, features: pd.DataFrame, targets: np.ndarray) -> Dict:
        """Train the model with cross-validation using balanced metrics"""
        try:
            logger.info(f"Training model with {len(features)} samples")

            # Prepare features
            X = self.prepare_features(features, training=True)

            # Log feature statistics after scaling
            logger.info("Scaled feature statistics:")
            for i, col in enumerate(features.columns):
                logger.info(f"{col}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")

            # Perform cross-validation with balanced accuracy
            cv_scores = cross_val_score(
                self.model,
                X,
                targets,
                cv=self.cv,
                scoring='balanced_accuracy'  # Changed to balanced accuracy
            )

            # Train final model
            self.model.fit(X, targets)

            # Store training metrics
            self.model_metrics = {
                'training_size': len(features),
                'cv_scores_mean': float(np.mean(cv_scores)),
                'cv_scores_std': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist(),
                'oob_score': float(self.model.oob_score_),
                'class_distribution': dict(zip(*np.unique(targets, return_counts=True))),
                'feature_names': list(features.columns),
                'feature_importances': dict(zip(
                    features.columns,
                    self.model.feature_importances_
                ))
            }

            # Log training results
            logger.info(f"Balanced accuracy (CV): {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.3f}")

            return self.model_metrics

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
