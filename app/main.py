import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import face_recognition
from image_preprocessing.action_parser import parse_gemini_response_to_actions
from image_preprocessing.gemini_interface import ask_gemini_edit_instructions, match_tones
from nlp.prompt_parser import parse_user_prompt
from image_preprocessing.editor import apply_edits
from image_preprocessing.metadata_extractor import metadata

choice = int(input("Select 1/2:\n1. Image Critique\n2. Edit with sample image"))
if choice == 1:              
    prompt = input("Enter the prompt: ")
    path1 = input("Enter the image path: ")
    raw_instructions = ask_gemini_edit_instructions(prompt, metadata(path1))
    print(raw_instructions)
    # actions = parse_gemini_response_to_actions(raw_instructions)

elif choice == 2:
    prompt = input("Enter the prompt: ")
    path1 = input("Enter the image path: ")
    path2 = input("Enter the sample image path: ")
    instructions = match_tones(prompt, metadata(path1), metadata(path2))
    print(instructions)
else:
    print("Invalid Input")


# ai_photo_enhacer_cv_nd_nlp/assets/face.png