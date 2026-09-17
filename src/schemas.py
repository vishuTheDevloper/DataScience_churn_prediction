from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Allowed answers for common service fields.
YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """Validate the 19 original customer inputs before prediction."""

    model_config = ConfigDict(
        extra="forbid",       # Reject unexpected fields.
        strict=True,          # Require correct data types.
        allow_inf_nan=False   # Reject infinity and NaN.
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo

    tenure: int = Field(ge=0)

    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)

    # The field is required, but JSON null is allowed for unknown charges.
    # The saved pipeline will handle that missing value.
    TotalCharges: float | None = Field(ge=0)

    @model_validator(mode="after")
    def check_service_consistency(self) -> Self:
        """Reject contradictory phone and internet service details."""

        if self.PhoneService == "No":
            if self.MultipleLines != "No phone service":
                raise ValueError(
                    "When PhoneService is 'No', MultipleLines must "
                    "be 'No phone service'."
                )
        elif self.MultipleLines == "No phone service":
            raise ValueError(
                "When PhoneService is 'Yes', MultipleLines must "
                "be 'Yes' or 'No'."
            )

        internet_addons = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        for field_name in internet_addons:
            value = getattr(self, field_name)

            if self.InternetService == "No":
                if value != "No internet service":
                    raise ValueError(
                        f"When InternetService is 'No', {field_name} "
                        "must be 'No internet service'."
                    )
            elif value == "No internet service":
                raise ValueError(
                    f"When InternetService is 'DSL' or 'Fiber optic', "
                    f"{field_name} must be 'Yes' or 'No'."
                )

        return self


class PredictionResponse(BaseModel):
    """Define the response returned by the prediction API."""

    model_config = ConfigDict(allow_inf_nan=False)

    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)