"""Human-readable metadata and value formatters for model features.

Internal representations (feature codes like ``HNI_pct``, raw subscription
multiples, raw SHAP values) are translated into domain language here so that
both the API responses and the LLM prompts stay self-explanatory.
"""


def format_subscription(value: float) -> str:
    """Format a subscription multiple as ``x`` times."""
    return f"{value:.2f}x"


def format_percent(value: float) -> str:
    """Format a ratio as a percentage."""
    return f"{value * 100:.1f}%"


def format_issue_size(value: float) -> str:
    """Format an issue size in crores."""
    return f"{value:,.0f} crores"


def describe_magnitude(shap_value: float) -> str:
    """Convert a SHAP value into a plain-language strength description.

    SHAP values are in log-odds units, so we bucket by absolute value
    and combine with direction.
    """
    abs_v = abs(shap_value)
    if abs_v >= 0.30:
        strength = "Very strongly"
    elif abs_v >= 0.15:
        strength = "Strongly"
    elif abs_v >= 0.05:
        strength = "Moderately"
    else:
        strength = "Slightly"
    direction = "increases profit" if shap_value >= 0 else "decreases profit"
    return f"{strength} {direction}"


def humanize_value(feature_name: str, value: float) -> str:
    """Format a raw feature value into a human-readable string.

    Args:
        feature_name: The model feature name (e.g. ``HNI_pct``).
        value: The raw value the applicant provided.

    Returns:
        A plain-language interpretation of the value.
    """
    meta = FEATURE_META.get(feature_name)
    if meta is not None and "formatter" in meta:
        return meta["formatter"](value)
    return str(value)


FEATURE_META = {
    "Total_Sub": {
        "label": "Total Subscription",
        "formatter": format_subscription,
    },
    "QIB": {
        "label": "QIB Subscription",
        "formatter": format_subscription,
    },
    "HNI": {
        "label": "HNI Subscription",
        "formatter": format_subscription,
    },
    "HNI_pct": {
        "label": "HNI Share of Subscription",
        "formatter": format_percent,
    },
    "RII_pct": {
        "label": "RII Share of Subscription",
        "formatter": format_percent,
    },
    "Issue_Size_crores": {
        "label": "Issue Size",
        "formatter": format_issue_size,
    },
}
