from pydantic import BaseModel, Field, field_validator
from typing import Optional


class IPOApplication(BaseModel):
    ipo_name: str = Field(..., description="Name of the IPO", json_schema_extra={"example": "ABC Infra"})
    issue_size: float = Field(..., gt=0, le=27858.8, description="Issue size in crores", json_schema_extra={"example": 500})
    subscription_qib: float = Field(
        ..., gt=0, le=331.6, description="QIB subscription multiple", json_schema_extra={"example": 42.42}
    )
    subscription_hni: float = Field(
        ..., gt=0, le=958.07, description="HNI subscription multiple", json_schema_extra={"example": 7.13}
    )
    subscription_rii: float = Field(
        ..., gt=0, le=1000, description="RII subscription multiple", json_schema_extra={"example": 2.84}
    )
    issue_price: float = Field(..., gt=0, le=10000, description="Issue price in INR (informational, not used by model)", json_schema_extra={"example": 220})
    listing_date: str = Field(
        ..., description="Listing date (YYYY-MM-DD)", json_schema_extra={"example": "2022-08-26"}
    )

    @field_validator("listing_date")
    @classmethod
    def validate_listing_date(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("listing_date must be in YYYY-MM-DD format")
        return v


class FeatureContribution(BaseModel):
    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Actual value provided")
    shap_value: float = Field(..., description="SHAP contribution value")
    impact: str = Field(
        ..., description="increases_profitability or decreases_profitability"
    )


class IPOResearchAssessment(BaseModel):
    ipo_name: str = Field(..., description="IPO name")
    prediction: str = Field(..., description="Profitable or Not Profitable")
    profitability_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of profit"
    )
    features_used: list[str] = Field(
        ..., description="Feature names used by the model"
    )
    shap_explanations: list[FeatureContribution] = Field(
        ..., description="Per-feature SHAP explanations"
    )
    research_report: str = Field("", description="LLM-generated research report")
    report_generated: bool = Field(
        False, description="Whether LLM report was generated"
    )
    input_warnings: list[str] = Field(
        default_factory=list, description="Warnings about input validity (e.g., out-of-distribution values)"
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
