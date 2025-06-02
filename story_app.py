import streamlit as st
from PIL import Image # To display images
from dotenv import load_dotenv
import os
import json5
import time
from google import genai
from google.genai import types
import typing_extensions as typing

#Load environment variables for api key
load_dotenv()


# Schema for the visual storytelling task structured generation
# The Gemini/Gemma model output format will follow this
class StoryPart(typing.TypedDict):
    image: typing.Annotated[int, "Should be the number in order according to the number of images in the input."]
    story_part: typing.Annotated[str, "Should be the generated story for the corresponding image in order."]

class Story(typing.TypedDict):
    story: list[StoryPart]
##########################################

#Images for one-shot examples in english
image_eng_path1 = "one_shot_examples/english/00.png"
image_eng_path2 = "one_shot_examples/english/01.png"
image_eng_path3 = "one_shot_examples/english/02.png"
image_eng_path4 = "one_shot_examples/english/03.png"
image_eng_path5 = "one_shot_examples/english/04.png"

image_eng1 = Image.open(image_eng_path1)
image_eng2 = Image.open(image_eng_path2)
image_eng3 = Image.open(image_eng_path3)
image_eng4 = Image.open(image_eng_path4)
image_eng5 = Image.open(image_eng_path5)

#Images for one-shot examples in spanish
image_spa_path1 = "one_shot_examples/spanish/00.jpg"
image_spa_path2 = "one_shot_examples/spanish/01.png"
image_spa_path3 = "one_shot_examples/spanish/02.jpg"
image_spa_path4 = "one_shot_examples/spanish/03.png"
image_spa_path5 = "one_shot_examples/spanish/04.jpg"

image_spa1 = Image.open(image_spa_path1)
image_spa2 = Image.open(image_spa_path2)
image_spa3 = Image.open(image_spa_path3)
image_spa4 = Image.open(image_spa_path4)
image_spa5 = Image.open(image_spa_path5)

class GenAIAgent:
    """
    A utility class for interacting with Google's GenAI models,
    handling synchronous generation.
    """

    # Configuration constants
    DEFAULT_FLASH_MODEL_ID = "gemini-2.0-flash"
    DEFAULT_CONCURRENT_INPUT_TOKENS_LIMIT = 8192
    #System Prompt for the model
    ENGLISH_SYSTEM_PROMPT = \
    '''
    You are an expert storyteller. You are not overly verbose in your stories, and you keep them very interesting.
    Your stories need to be in third person, from an external point of view.
    Craft an engaging story based strictly on the following figures presented in order.
    Follow the schema as defined for your output:
    {{
    "story": [
        {
        "image": <int>,
        "story_part": "<str>"
        },
        ...
    ]
    }}
    '''
    SPANISH_SYSTEM_PROMPT = \
    '''
    Eres un narrador experto en español. No eres excesivamente verboso en tus historias y las mantienes muy interesantes.
    Tus historias necesitan ser creadas en tercera persona, desde un punto de vista externo.
    Crea una historia atractiva basada estrictamente en las siguientes imágenes presentadas en orden.
    Sigue el esquema definido para tu respuesta:
    {{
    "story": [
        {
        "image": <int>,
        "story_part": "<str>"
        },
        ...
    ]
    }}
    '''

    # Standard safety configurations
    API_SAFETY_SETTINGS = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]

    #User prompt used for the model with each english language task
    LLM_VIST_GEN_USER_ENGLISH_PROMPT = \
    f'''
    You are an expert storyteller. You are not overly verbose in your stories, and you keep them very interesting.
    Your stories need to be in third person, from an external point of view.
    Craft an engaging story based strictly on the following figures presented in order.
    Follow the schema as defined for your output:
    {{
    "story": [
        {{
        "image": <int>,
        "story_part": "<str>"
        }},
        ...
    ]
    }}

    First I will show you an example of how to do it:
    INPUT:
    Image 1:
    {image_eng1}
    Image 2:
    {image_eng2}
    Image 3:
    {image_eng3}
    Image 4:
    {image_eng4}
    Image 5:
    {image_eng5}
    OUTPUT:
    {{
    "story": [
        {{
        "image": 1,
        "story_part": "I am big. I am bigger than a stream. I have lot of water. My water is used for drinking. They say I always run. I never walk. My water moves from here to there. They say I flow."
        }},
        {{
        "image": 2,
        "story_part": "I have a mouth. But I never eat. I can't even talk. You hear the sound when I flow. I move soil and small rocks. I help move logs of wood. I do so much work but I don't have hands."
        }},
        {{
        "image": 3,
        "story_part": "I have a bed. But I don't sleep. I am always awake. I am always moving. I make soil. I make valleys. My water is sweet."
        }},
        {{
        "image": 4,
        "story_part": "Let me tell you again. I have no legs but I can run. I can't eat or talk but I have a mouth. I have a long bed but I never sleep. I have a bank but there is no money in it. Did you guess who am I?"
        }},
        {{
        "image": 5,
        "story_part": "A river! I am a river. Do you know some names I have?"
        }}
    ]
    }}

    Now generate the story for the user's input.

    ----INPUT STARTS HERE----
    '''

    #User prompt used for the model with each english language task
    LLM_VIST_GEN_USER_SPANISH_PROMPT = \
    f'''
    Eres un narrador experto en español. No eres excesivamente verboso en tus historias y las mantienes muy interesantes.
    Tus historias necesitan ser creadas en tercera persona, desde un punto de vista externo.
    Crea una historia atractiva basada estrictamente en las siguientes imágenes presentadas en orden.
    Sigue el esquema definido para tu respuesta:
    {{
    "story": [
        {{
        "image": <int>,
        "story_part": "<str>"
        }},
        ...
    ]
    }}
    Primero te enseñaré un ejemplo sobre como hacerlo:
    INPUT:
    Imagen 1:
    {image_spa1}
    Imagen 2:
    {image_spa2}
    Imagen 3:
    {image_spa3}
    Imagen 4:
    {image_spa4}
    Imagen 5:
    {image_spa5}
    OUTPUT:
    {{
    "story": [
        {{
        "image": 1,
        "story_part": "Ya pasaron varios meses. He crecido mucho últimamente. Creo que ya no quepo aquí en donde me encuentro.  Mi mami ya sabe cuando naceré. Creo que será ¡hoy! es ¡hoy!, es ¡hoy!Por fin conoceré a mamá y a papá. ¿Cómo serán? ¿Me querrán como yo los quiero?¿Será que ellos van a jugar conmigo? Tengo tantas preguntas. Ya estoy feliz pensando en estar con ellos siempre."
        }},
        {{
        "image": 2,
        "story_part": "Mi papá estaba angustiado en la sala de espera. No le daban noticias de mi nacimiento. El médico llega y le cuenta que soy un hermoso bebé.\nLe pide que se coloque una bata para entrar a conocerme. Al verme, papá llora de felicidad.\nYo sé que mi mamá es una guerrera. Ella siempre luchará por mí.  Me va a proteger toda la vida. Me va a alimentar y resguardar de todo peligro. Ya quiero estar en casa y conocer todo lo que me tienen preparado."
        }},
        {{
        "image": 3,
        "story_part": "Las mamás son muy dulces. Siempre saben qué es lo mejor para ti. Cuando estás en un problema, siempre te acompañan. Mamá siempre se preocupa cuando estás lejos. Le pide a Dios que te proteja.\n\n Papá siempre está con ella apoyándola. Juntos quieren verte crecer y que te conviertas en una persona de bien. Quieren que estudies y tengas un buen futuro."
        }},
        {{
        "image": 4,
        "story_part": "Papá y mamá son personas increíbles. Son como ángeles que nos cuidan sin importar la edad que tengamos. Si somos bebés, niños, adolescentes o personas adultas, siempre están para apoyarnos.\nCuando pase el tiempo y ellos envejezcan, debo cuidar de ellos, así como ellos cuidaron de mí.  Me enseñaron a caminar, a amarrarme las agujetas de los zapatos, a aprender, a ser feliz y a vivir en paz.\nMis papás son lo mejor que he tenido.  Siempre los llevo y llevaré en mi corazón. ¡Somos una hermosa familia!"
        }},
        {{
        "image": 5,
        "story_part": "¡Mi familia!"
        }}
    ]
    }}

    Ahora genera la historia para el input del usuario.
    IMPORTANTE: El output debe estar en idioma español.

    ----EL INPUT EMPIEZA AQUí----
    '''
    
    def __init__(
        self,
        specified_model_id: str = None,
        select_system_prompt: str = None,
    ):
        """
        Initializes the GenAIAgent with model and system prompt settings.

        Args:
            specified_model_id (str, optional): The ID of the model to use.
                                              Defaults to DEFAULT_FLASH_MODEL_ID.
            custom_system_prompt (str, optional): A custom system instruction.
                                                Defaults to DEFAULT_SYSTEM_PROMPT.
            api_key (str, optional): Your Google API key. If None, it will attempt
                                     to use the key passed during client initialization
                                     (e.g., via genai.configure or environment variable).
        """
        self.chosen_model_id = specified_model_id or self.DEFAULT_FLASH_MODEL_ID
        if select_system_prompt:
            if select_system_prompt == "Spanish":
                self.active_system_prompt = self.SPANISH_SYSTEM_PROMPT
                self.active_user_prompt = self.LLM_VIST_GEN_USER_SPANISH_PROMPT
            else:
                self.active_system_prompt = self.ENGLISH_SYSTEM_PROMPT
                self.active_user_prompt = self.LLM_VIST_GEN_USER_ENGLISH_PROMPT
        else:
            self.active_system_prompt = self.ENGLISH_SYSTEM_PROMPT
            self.active_user_prompt = self.LLM_VIST_GEN_USER_ENGLISH_PROMPT

        
        api_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        self.genai_client = genai.Client(api_key=api_key) # Client for generation

        self.output_token_cap = 8192 # Maximum tokens for model response

    def generate_single_response(
        self,
        input_content_list: list,
        response_schema_definition,
        include_metrics_log: bool = False
    ):
        """
        Generates a single content response from the model.
        """
        request_start_time = time.time()
        if(self.chosen_model_id=="gemini-2.0-flash"):
            content_generation_config = types.GenerateContentConfig(
                temperature=1,
                system_instruction=self.active_system_prompt,
                max_output_tokens=self.output_token_cap,
                response_modalities=["TEXT"],
                response_mime_type = "application/json",
                response_schema = response_schema_definition,
                safety_settings=self.API_SAFETY_SETTINGS
            )
        #Gemma 3 with the API does not support structured output or System Prompt
        else:
            content_generation_config = types.GenerateContentConfig(
                temperature=1,
                max_output_tokens=self.output_token_cap,
                response_modalities=["TEXT"],
                safety_settings=self.API_SAFETY_SETTINGS
            )

        api_response = self.genai_client.models.generate_content(
            model=self.chosen_model_id,
            contents=input_content_list,
            config=content_generation_config,
        )

        generated_text = api_response.text
        if include_metrics_log:
            performance_log = {
                "model_used": self.chosen_model_id,
                "output_tokens": api_response.usage_metadata.candidates_token_count,
                "input_tokens": api_response.usage_metadata.prompt_token_count,
                "elapsed_time_sec": round(time.time() - request_start_time, 2),
            }
            return generated_text, performance_log
        return generated_text



# Page Configuration
st.set_page_config(
    page_title="Visual Storyteller AI",
    page_icon="📖",
    layout="wide"
)

# Initialize Session State
# For managing file uploader reset
if 'upload_key_counter' not in st.session_state:
    st.session_state.upload_key_counter = 0
# For storing generated story parts
if 'generated_story_parts' not in st.session_state:
    st.session_state.generated_story_parts = None
# For storing PIL Image objects for final display alongside story parts
if 'images_for_story_display' not in st.session_state:
    st.session_state.images_for_story_display = []

# Placeholder for your LLM Logic
def generate_story_with_llm(image_data_list, llm_model_name, language):
    """
    Function to generate a story using an LLM.
    Returns a list of story segments, one for each image.

    Args:
        image_data_list (list): A list of image data (e.g., bytes).
        llm_model_name (str): The name of the selected LLM.
        language (str): The selected language ("English" or "Spanish").

    Returns:
        list[str]: A list of story segments, or an empty list/error indication.
    """
    num_images = len(image_data_list)
    story_segments = []
    model_name = ""
    if llm_model_name == "Gemma 3 (27B)":
        model_name = "gemma-3-27b-it"
    else:
        model_name = "gemini-2.0-flash"

    # Instance of the agent for story generation
    ai_handler = GenAIAgent(specified_model_id=model_name, select_system_prompt=language)
    # Story generation logic
    if num_images > 0:
        user_content = [ai_handler.active_user_prompt]
        for i, image in enumerate(image_data_list):
            if language == "Spanish":
                user_content.append(f"Imagen {i+1}:")
            else:
                user_content.append(f"Image {i+1}:")
            user_content.append(image)
    
    schema = Story
    gen_story, log = ai_handler.generate_single_response(input_content_list=user_content, response_schema_definition=schema, include_metrics_log = True)
    print(gen_story)
    text = gen_story.replace("```json","")
    text = text.replace("```","")
    final_story = json5.loads(text)
    print("Generation successful!")
    print(log)
    for part in final_story["story"]:
        story_segments.append(part["story_part"])
    return story_segments

# App Interface
st.title("🖼️ Visual Storyteller AI 📖")
st.markdown("Upload 2 to 5 images, choose your AI model and language, and let's craft a story!")

# Image Upload Section
st.header("1. Upload Your Images")
uploaded_files = st.file_uploader(
    "Select 2 to 5 images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"image_uploader_{st.session_state.upload_key_counter}" # Key for reset capability
)

num_uploaded_files = 0
image_data_for_llm = [] # To store image data for LLM

if uploaded_files:
    num_uploaded_files = len(uploaded_files)
    if num_uploaded_files > 5:
        st.warning("You can upload a maximum of 5 images. Displaying the first 5.")
        uploaded_files = uploaded_files[:5]
        num_uploaded_files = 5
    elif num_uploaded_files < 2 and num_uploaded_files > 0: # Allow 1 for preview, but not for generation
        st.info("Please upload at least 2 images to generate a story. You currently have 1.")
    elif num_uploaded_files == 0: # User cleared all files
        st.info("Upload 2 to 5 images to begin.")

# Process and display uploaded images for preview
if uploaded_files and num_uploaded_files > 0:
    st.subheader("Your Uploaded Images (Preview):")
    cols = st.columns(min(num_uploaded_files, 5))
    current_batch_pil_images = [] # For storing PIL images for display
    current_batch_byte_data = []  # For storing byte data for LLM

    for i, uploaded_file_obj in enumerate(uploaded_files):
        with cols[i]:
            try:
                pil_image = Image.open(uploaded_file_obj)
                st.image(pil_image, caption=f"Image {i+1}: {uploaded_file_obj.name}", width=150)
                current_batch_pil_images.append(pil_image)

                uploaded_file_obj.seek(0) # Reset pointer for reading bytes
                current_batch_byte_data.append(uploaded_file_obj.read())
            except Exception as e:
                st.error(f"Error loading image {uploaded_file_obj.name}: {e}")
                # Remove problematic file from lists if necessary or handle error
    
    # Store the processed images for use in story display and for LLM
    st.session_state.images_for_story_display = current_batch_pil_images
    image_data_for_llm = current_batch_pil_images
else:
    # Clear if no files are uploaded or if they were all removed
    st.session_state.images_for_story_display = []
    image_data_for_llm = []


# LLM and Language Selection
st.header("2. Configure Story Generation")
col1, col2 = st.columns(2)
with col1:
    selected_llm = st.selectbox(
        "Choose your LLM:",
        ("Gemma 3 (27B)", "Gemini 2.0 Flash"),
        key="llm_select"
    )
with col2:
    selected_language = st.radio(
        "Select story language:",
        ("English", "Spanish"),
        horizontal=True,
        key="language_select"
    )

# Generation Button
st.header("3. Generate Your Story")

# Enable button only if conditions are met (2-5 images)
can_generate = (num_uploaded_files >= 2 and num_uploaded_files <= 5 and image_data_for_llm)

if st.button("✨ Start Generating Story", disabled=not can_generate, type="primary"):
    if not can_generate:
        if num_uploaded_files < 2 :
            st.error("Please upload at least 2 images.")
        elif num_uploaded_files > 5:
             st.error("Please ensure you have between 2 and 5 images uploaded.")
        # This case should be covered, but as a fallback:
        elif not image_data_for_llm:
            st.error("Image data is missing. Please re-upload images.")
    else:
        spinner_message = (
            f"Generating your story with {len(image_data_for_llm)} parts "
            f"in {selected_language} using {selected_llm}... Please wait."
        )
        with st.spinner(spinner_message):
            generated_parts = generate_story_with_llm(
                image_data_list=image_data_for_llm,
                llm_model_name=selected_llm,
                language=selected_language
            )
            st.session_state.generated_story_parts = generated_parts

# Display Output
if st.session_state.generated_story_parts:
    st.header("📜 Your Generated Story:")
 
    displayed_images = st.session_state.get('images_for_story_display', [])
    story_parts_to_display = st.session_state.generated_story_parts

    if displayed_images and story_parts_to_display and len(displayed_images) == len(story_parts_to_display):
        for i, (img_to_display, story_part) in enumerate(zip(displayed_images, story_parts_to_display)):
            st.subheader(f"Image {i+1} / Story Part {i+1}")
            st.image(img_to_display, width=400) # Adjust width as needed for story display
            st.markdown(story_part)
            if i < len(displayed_images) - 1: # Add separator if not the last item
                st.markdown("---")
    elif story_parts_to_display: # Fallback if image association fails but story parts exist
        st.warning("Could not perfectly match images to story parts. Displaying story parts sequentially.")
        for i, story_part in enumerate(story_parts_to_display):
            st.markdown(f"**Story Part {i+1}:**\n{story_part}")
            st.markdown("---")
    else:
        st.error("An issue occurred while generating or displaying the story. Please try again.")

    # Clear button
    if st.button("Clear Story & Start Over", key="clear_story"):
        st.session_state.generated_story_parts = None
        st.session_state.images_for_story_display = [] 
        st.session_state.upload_key_counter += 1
        st.rerun()

st.markdown("---")
st.caption("Made by Didier in Hsinchu with ❤️")