import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
YES_NO_SERVICES = [
    "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
TENURE_LABELS = [
    "0-12 months", "13-24 months", "25-48 months", "Over 48 months",
]


class CustomerPreprocessor(TransformerMixin, BaseEstimator):
    """Apply fixed cleaning rules and optional customer features.

    This step learns no medians, category vocabularies, or target statistics.
    Those learned operations belong in later pipeline steps and are fitted
    separately inside each cross-validation fold.

    Every input column must be present. Missing cell values are allowed and
    are left for the pipeline's imputers. API validation should separately
    reject invalid types, ranges, categories, and contradictory service data.
    """

    def __init__(self, add_features=True):
        self.add_features = add_features

    @staticmethod
    def _validate_columns(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CustomerPreprocessor requires a pandas DataFrame.")
        if X.columns.duplicated().any():
            raise ValueError("Customer input contains duplicate column names.")
        missing = [column for column in INPUT_FEATURES if column not in X]
        if missing:
            raise ValueError("Missing customer input columns: " + ", ".join(missing))

    def fit(self, X, y=None):
        self._validate_columns(X)
        # Store the input schema, not any learned values or target information.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._validate_columns(X)

        # Selecting named inputs excludes identifiers, targets, and unrelated
        # columns even if a caller supplies them. Never mutate the input table.
        data = X.loc[:, INPUT_FEATURES].copy()

        for column in NUMERICAL_FEATURES:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # Normalize whitespace and represent missing categories as np.nan.
        # Keeping an object dtype also works with sklearn imputers across
        # pandas versions that use different default string dtypes.
        for column in CATEGORICAL_FEATURES:
            if column == "SeniorCitizen":
                values = pd.to_numeric(data[column], errors="coerce")
            else:
                values = data[column].astype("string").str.strip()
                values = values.mask(values.eq(""), pd.NA)
            data[column] = values.astype(object).where(values.notna(), np.nan)

        if self.add_features:
            data["TenureGroup"] = pd.cut(
                data["tenure"],
                bins=[0, 12, 24, 48, float("inf")],
                labels=TENURE_LABELS,
                include_lowest=True,
            ).astype(object)

            service_flags = pd.DataFrame(index=data.index)
            for column in YES_NO_SERVICES:
                known_values = ["Yes", "No", "No internet service"]
                if column == "PhoneService":
                    known_values = ["Yes", "No"]
                flag = data[column].eq("Yes").astype(float)
                service_flags[column] = flag.where(
                    data[column].isin(known_values), np.nan
                )

            internet_flag = data["InternetService"].isin(
                ["DSL", "Fiber optic"]
            ).astype(float)
            service_flags["InternetService"] = internet_flag.where(
                data["InternetService"].isin(["No", "DSL", "Fiber optic"]),
                np.nan,
            )

            # An unknown or missing service is not automatically "No".
            # A missing count is handled by the numerical imputer instead.
            data["ServiceCount"] = service_flags.sum(
                axis=1, min_count=len(service_flags.columns)
            )

        return data
