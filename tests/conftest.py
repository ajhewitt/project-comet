import warnings

# Healpy uses its own subclass, so catch it broadly by name
warnings.filterwarnings(
    "ignore",
    message=r'"verbose" was deprecated',
    category=Warning,
    module=r"^healpy(\.|$)",
)
