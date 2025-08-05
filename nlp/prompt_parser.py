import re

PROMPT_MAPPINGS = {
    "cinematic" : {
        "increase_contrast": True,
        "color_grade" : "teal_orange"
    },

    "moody" : {
        "decrease_brightness": "slightly",
        "add_vignette": True,
        "desaturate": "moderate"
    },

    "vibrant": {
        "increase_saturation": "high",
        "boost_brightness": "moderate"
    },
    "retro": {
        "add_filter": "sepia",
        "reduce_contrast": "slightly"
    },
    "professional": {
        "balance_white": True,
        "sharpen": True
    }
}

def parse_user_prompt(prompt: str):
    prompt = prompt.lower()
    parsed_result = {}

    for keyword, actions in PROMPT_MAPPINGS.items():
        if re.search(rf"\b{keyword}\b", prompt):
            parsed_result.update(actions)

    return parsed_result

if __name__ == "__main__":
    user_input = "Make this photo more cinematic and moody"
    actions = parse_user_prompt(user_input)
    print("Actions to apply:", actions)