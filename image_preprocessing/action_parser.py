def parse_gemini_response_to_actions(response_text: str) -> dict:
    """Convert Gemini's response (natural language) to structured edit actions."""

    actions = {}

    text = response_text.lower()

    # Contrast
    if "increase contrast" in text:
        actions["increase_contrast"] = True
    if "reduce contrast" in text or "lower contrast" in text:
        actions["reduce_contrast"] = "slightly"

    # Brightness
    if "increase brightness" in text or "boost brightness" in text:
        actions["boost_brightness"] = "moderate"
    if "reduce brightness" in text or "darken" in text:
        actions["decrease_brightness"] = "slightly"

    # Saturation
    if "desaturate" in text or "reduce saturation" in text:
        actions["desaturate"] = "moderate"
    if "increase saturation" in text or "vibrant" in text:
        actions["increase_saturation"] = "high"

    # Color Grade
    if "teal-orange" in text or "cinematic" in text:
        actions["color_grade"] = "teal_orange"

    # Vignette
    if "add vignette" in text or "moody" in text:
        actions["add_vignette"] = True

    # Filter
    if "sepia" in text:
        actions["add_filter"] = "sepia"

    # White balance
    if "fix white balance" in text or "balance white" in text:
        actions["balance_white"] = True

    # Sharpen
    if "sharpen" in text:
        actions["sharpen"] = True

    return actions
