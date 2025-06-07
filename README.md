# Visual Storyteller AI 🖼️📖

This Streamlit application uses Google's Generative AI models (Gemini/Gemma) to craft engaging stories based on a sequence of 2 to 5 uploaded images. You can choose the AI model and the language (English or Spanish) for the story generation.

## Features

- Upload 2 to 5 images (PNG, JPG, JPEG).
- Choose between Gemma 3 (27B) and Gemini 2.0 Flash models.
- Select story language: English or Spanish.
- View the generated story parts alongside their corresponding images.
- Includes one-shot examples embedded in prompts for better model guidance.

## Prerequisites

- Python 3.13 (tested on this version)
- A Google API Key for Generative AI. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup & Installation

1. **Clone the repository.**

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3. **Install dependencies:**

    Make sure the `requirements.txt` file is present in the directory. Then run:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your Google API Key:**

    Create a file named `.env` in the root of your project directory and add your API key:

    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

    Alternatively, if deploying to Streamlit Community Cloud, you can set this as a secret named `GOOGLE_API_KEY`.

5. **Ensure example images are present:**

    The application uses one-shot example images for prompting. Make sure you have the following directory structure and images:

    ```
    .
    ├── story_app.py
    └── one_shot_examples/
        ├── english/
        │   ├── 00.png
        │   ├── 01.png
        │   ├── 02.png
        │   ├── 03.png
        │   └── 04.png
        └── spanish/
            ├── 00.jpg
            ├── 01.png
            ├── 02.jpg
            ├── 03.png
            └── 04.jpg
    ```

    *Note: Ensure these exact filenames and extensions match those expected in your script unless you modify the code.*

## Running the Application

Once the setup is complete, run the Streamlit app:

```bash
streamlit run story_app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

## How to Use

1. Upload 2 to 5 images using the file uploader.
2. Select your preferred LLM model from the dropdown.
3. Choose the language (English or Spanish) for the story.
4. Click the **"✨ Start Generating Story"** button.
5. View the generated story parts displayed below each corresponding image.
6. Click **"Clear Story & Start Over"** to reset.

## Benchmark Dataset Results (`final_results_dataset`)

The `final_results_dataset` folder contains story generations from running the models (Gemini 2.0 Flash and Gemma 3) on a subset of the **BLOOM VIST Dataset**. This dataset originates from the paper:

> Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks  
> Colin Leong, Joshua Nemecek, Jacob Mansdorfer, Anna Filighera, Abraham Owodunni, Daniel Whitenack (2022)  
> arXiv:2210.14712 [cs.CL] — https://arxiv.org/abs/2210.14712

### BibTeX

```bibtex
@misc{leong2022bloomlibrarymultimodaldatasets,
  title={Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks},
  author={Colin Leong and Joshua Nemecek and Jacob Mansdorfer and Anna Filighera and Abraham Owodunni and Daniel Whitenack},
  year={2022},
  eprint={2210.14712},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2210.14712}
}
```

## Dataset Filtering and Generation Statistics

- **Initial Filtering:** From 364 languages, the dataset was filtered to include only English and Spanish stories:  
  - 2,531 English stories  
  - 510 Spanish stories

- **Image Count Filtering:** Filtered to stories with ≤5 images:  
  - 479 English stories  
  - 128 Spanish stories

- **Final Generated Story Counts (after removing “None” responses):**

| Language | Gemini 2.0 Flash | Gemma 3 |
|----------|------------------|---------|
| English  | 473 stories      | 455     |
| Spanish  | 127 stories      | 112     |

---

## Subjective Analysis Results (`survey_data`)
The `survey_data` folder contains all the related data collected for subjective analysis in the report. 
It contains some google form screenshots of the format used to collect the data for evaluation of 5 stories.
The Stories used in the form, the text and images related to them.
And the raw data collected from the surveys plus the R code used to make the statistical analysis with the respective results, which were discussed in the report.

---
**Made by Didier in Hsinchu with ❤️**