"""Utilities for handling Google Colab-specific issues."""

import sys
import warnings


def is_google_colab():
    """Check if code is running in Google Colab environment."""
    return 'google.colab' in sys.modules


def check_colab_backports_issue():
    """
    Check for the known Colab backports warning issue.
    
    Google Colab pre-imports scikit-learn, which uses joblib.backports.
    When installing packages, pip detects "backports" in module names and
    shows a warning popup, even though joblib.backports is not a real
    backports package.
    
    Returns:
        bool: True if the issue is detected
    """
    if not is_google_colab():
        return False
    
    # Check if joblib.backports is already loaded
    return 'joblib.backports' in sys.modules


def warn_about_colab_issue():
    """Issue a warning about the known Colab installation issue."""
    if check_colab_backports_issue():
        warnings.warn(
            "Note: You may see a Google Colab warning about 'backports' when installing datawrangler. "
            "This is a known issue caused by Colab pre-importing scikit-learn. "
            "The warning can be safely ignored - datawrangler will work correctly after installation. "
            "No runtime restart is required.",
            UserWarning,
            stacklevel=2
        )