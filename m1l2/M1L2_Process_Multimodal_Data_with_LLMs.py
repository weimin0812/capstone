import ast
import base64
import json
import os

import matplotlib.pyplot as plt
import requests
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables to safely store API key
load_dotenv()


# Suppress all unnecessary warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
warnings.filterwarnings('ignore')

# Initialize DeepSeek client with official base URL
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Extract all images from the zip file
# with zipfile.ZipFile("synthetic-recipe-images.zip", 'r') as zip_ref:
#     zip_ref.extractall()

# ======================
# Step 1: Load and inspect recipe dataset
# ======================
### Step 1.1: Load the json file. Define the loaded data as recipe_data.
with open("Recipes.json", "r", encoding="utf-8") as f:
    recipe_data = json.load(f)

### Step 1.2: Print each key-value pair of the first recipe with value type
print("===== First Recipe Detailed Info =====")
first_recipe = recipe_data[0]
for key, value in first_recipe.items():
    print(f"{key} ({type(value).__name__}): {value}")

### Step 1.3: Show the image of the first recipe
first_img_path = "synthetic_recipe_images/recipe1.png"
recipe_img = Image.open(first_img_path)
plt.imshow(recipe_img)
plt.axis("off")
plt.show()


# ======================
# Step 2: Define vision LLM function using DeepSeek multimodal model
# ======================
# ======================
# Step 2: Define vision LLM function using DeepSeek multimodal model
# ======================
def vision_llm(system_msg, prompt_txt, image_path):
    """
    Generate image description using DeepSeek vision model
    Supports both PNG and JPG formats
    :param system_msg: system role instruction
    :param prompt_txt: user task prompt
    :param image_path: local path of target image
    :return: generated text caption
    """
    from PIL import Image
    import io

    # 🔴 关键：自动压缩图片到小尺寸，防止超长 base64
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((600, 600))  # 压缩到 600px 以内

    # 保存到内存
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    buf.seek(0)

    # base64 编码
    base64_encoded_img = base64.b64encode(buf.read()).decode("utf-8")

    # DeepSeek 官方格式
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"{prompt_txt}\n![image](data:image/jpeg;base64,{base64_encoded_img})"
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content


# ======================
# Step 3: Design prompt template for food image caption
# ======================
def image_caption_prompt_template(food_name):
    """
    Create system and user prompt for food image caption task
    :param food_name: official name of target food
    :return: system message, user prompt text
    """
    ### Step 3.1: Design the prompts
    image_caption_system_msg = "You are a professional food content writer. Describe food images accurately and vividly in formal English."
    image_caption_prompt_txt = f"Describe the appearance, color and plating style of this {food_name} food image concisely, keep output within two sentences."
    return image_caption_system_msg, image_caption_prompt_txt


### Step 3.2: Get the prompts with the food name of the first recipe
test_food_name = first_recipe["name"]
test_sys_msg, test_user_prompt = image_caption_prompt_template(test_food_name)

### Step 3.3: Get the test response and print it
response = vision_llm(test_sys_msg, test_user_prompt, first_img_path)
print("\n===== Image Caption Test Result =====")
print(response)

# ======================
# Step 4: Batch generate image captions for all recipes
# ======================
for i in range(len(recipe_data)):
    if (i + 1) % 20 == 0:
        print(f'{i + 1} out of {len(recipe_data)} is done')

    ### Step 4.1: Get the caption prompts
    current_food = recipe_data[i]["name"]
    current_sys, current_prompt = image_caption_prompt_template(current_food)
    current_img_path = f"synthetic_recipe_images/recipe{i + 1}.png"

    ### Step 4.2: Get the response with the prompts
    response = vision_llm(current_sys, current_prompt, current_img_path)

    ### Save the response as another item in the recipe data
    recipe_data[i]['image_description'] = response

print('ALL DONE!')

# Save augmented recipe dataset to new json file
filename = 'augmented_food_recipe.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(recipe_data, f, indent=4)

# ======================
# Part 1: Process user review dataset
# ======================
### Step 1.1: Load the review dataset with variable name user_review_data
with open("Synthetic-User-Reviews.json", "r", encoding="utf-8") as f:
    user_review_data = json.load(f)

### Step 1.2: Print the first review by key-value pairs
print("\n===== First User Review Data =====")
first_review = user_review_data[0]
for k, v in first_review.items():
    print(f"{k}: {v}")

### Step 1.3: Use ast.literal_eval to convert string list to real python list
first_img_url_str = first_review["images"]
review_image_url_list = ast.literal_eval(first_img_url_str)

if review_image_url_list:
    ### Step 1.4: Use the requests.get() method to get the image content
    target_img_url = review_image_url_list[0]
    img_raw_content = requests.get(target_img_url).content

    ### Step 1.5: Write image content to temporary local file
    with open('review_image_placeholder.jpg', 'wb') as temp_img:
        temp_img.write(img_raw_content)

    ### Step 1.6: Open and show the temporary review image
    temp_show_img = Image.open('review_image_placeholder.jpg')
    plt.imshow(temp_show_img)
    plt.axis("off")
    plt.show()


# ======================
# Part 2: Design prompt with review context
# ======================
def review_context_image_caption_prompt_template(reviews):
    """
    Generate prompt to describe image combined with user written review
    :param reviews: original user text review content
    :return: system message, user prompt
    """
    ### Step 2.1: Design your prompts
    review_context_image_caption_system_msg = "You are an objective food reviewer. Combine user's comment to describe the corresponding food image."
    review_context_image_caption_prompt_txt = f"User actual review content: {reviews}\nPlease describe this food image according to user's review attitude and feelings."
    return review_context_image_caption_system_msg, review_context_image_caption_prompt_txt


### Step 2.2: Get the prompts
review_text_content = first_review["text"]
rev_sys_msg, rev_user_prompt = review_context_image_caption_prompt_template(review_text_content)

### Step 2.3: Get the response by the vision_llm you defined previously
response = vision_llm(rev_sys_msg, rev_user_prompt, 'review_image_placeholder.jpg')
print("\n===== Review Context Image Description =====")
print(response)


# ======================
# Part 3: Batch process all user review images with retry mechanism
# ======================
### URL Request function with Retry
# Retries up to 10 times, starting at 1s and doubling (1s, 2s, 4s...)
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()  # Must raise error for retry to trigger
    return response


# Loop through all user review records
for i in range(len(user_review_data)):
    ### Step 3.1: Convert the string to the Python list of image urls
    raw_url_str = user_review_data[i]["images"]
    review_images = ast.literal_eval(raw_url_str)

    review_image_captions = []
    if len(review_images) > 0:
        for img_url in review_images:
            try:
                ### Step 3.2: Use get_data_with_retry to get the image_data
                image_data = get_data_with_retry(img_url)
                print("Success!")
            except Exception as e:
                print(f"All retries failed at url {img_url}:", e)
                continue

            # Save downloaded image to local temp file
            image = image_data.content
            with open('review_image_placeholder.jpg', 'wb') as img_file:
                img_file.write(image)

            ### Step 3.3: Get the prompts, get the response, append result
            current_review_text = user_review_data[i]["text"]
            ctx_sys, ctx_prompt = review_context_image_caption_prompt_template(current_review_text)
            final_caption = vision_llm(ctx_sys, ctx_prompt, 'review_image_placeholder.jpg')
            review_image_captions.append(final_caption)

    # Add generated image captions into original review data
    user_review_data[i]['image_captions'] = review_image_captions

print('user review data ALL DONE!')

# Export final augmented user review dataset
filename = 'augmented_user_review.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(user_review_data, f, indent=4)
print('augmented_user_review saved !')
