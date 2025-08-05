import google.generativeai as genai

genai.configure(api_key="AIzaSyBRgkK_KfGc0_NVxRQJGH64yYs3iBapJ-I")

def ask_gemini_edit_instructions(prompt_text, image_metadata):
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Construct system prompt
    full_prompt = f"""
You are a professional photo editing assistant. Your task is to:

1. Analyze the given image metadata to describe the current visual style and quality of the image.also rate the image out of 10 according to it's current state and at what point it can be taken. 
2. Understand the user's desired edit style or mood.
3. Suggest a list of editing instructions (step-by-step) to transform the image from its current state to the desired style. Include edits such as changes in contrast, brightness, saturation, hue, sharpness, color tones, shadow/highlight adjustments, white balance, etc.

Be specific in your response. Do NOT generate code or perform any image manipulation—only return professional editing instructions.

---

📊 Image Metadata:
{image_metadata}

🗣️ User Prompt:
"{prompt_text}"

---

Respond in this format and properly guide like an Image editinf specialist:

🖼️ Current Style:
<brief summary of the image look and feel>

🎯 Desired Style:
<what the user wants>

🛠️ Editing Recommendations for technical users and also for basic users:
- Adjust brightness/contrast/etc...
- Add tone curve / cinematic grading
- Apply warmer/cooler tones
- Enhance shadows/highlights
- Change dominant color balance...
"""


    response = model.generate_content(full_prompt)
    return response.text



def match_tones(prompt, sample_image_metadata, for_edit_metadata):
    model = genai.GenerativeModel('gemini-2.0-flash')

    full_prompt = f"""
    You are a visual style and image aesthetics expert. Your task is to analyze and compare the metadata of two images — a sample image and a user-provided image — to determine how to match the user's image color tones, lighting, and overall style with the sample image.

Here is the metadata for both images:
user_prompt: {prompt}

**Sample Image Metadata: {sample_image_metadata}**

**User Image Metadata: {for_edit_metadata}**

**Instructions:**
1. Suggest detailed color tone adjustments to make the user image match the sample image — including brightness, hue, saturation, contrast, sharpness and dominant color balance.
2. If needed, recommend stylistic changes (e.g., cinematic tone, vintage filter).
3. Suggest how to align skin tones using realistic, non-artificial techniques.
4. If the scenes or lighting differ, mention how to adapt them.
5. Give final creative recommendations to match the visual vibe of the sample image.

Respond in a clean, structured, and actionable format.

"""
    
    comp_response = model.generate_content(full_prompt)
    return comp_response.text
