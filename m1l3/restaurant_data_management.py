import json
import os
import shutil
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ===================== File Path Configuration =====================
FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'

EXAMPLE_RESTAURANT_PARAGRAPH = (
    'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, '
    '**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, '
    'it captures the salt-air energy of the coast through its signature beer-battered snapper '
    'tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:'
)

# Load environment variables
load_dotenv()

# Initialize DeepSeek client with official base URL
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ===================== Data Structure (MATCH YOUR JSON EXACTLY) =====================
class Restaurant(BaseModel):
    itemId: int
    name: str
    location: str
    type: str
    food_style: str
    rating: float
    price_range: int
    signatures: List[str]
    vibe: str
    environment: str
    shortcomings: List[str]


# ===================== Helper Functions =====================
def load_data(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(file_path, data, backup_path=None):
    if backup_path and os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def show_restaurant_card(restaurant, index):
    print(f"\n--- Restaurant #{index} ---")
    print(f"ID: {restaurant.get('itemId', 'N/A')}")
    print(f"Name: {restaurant.get('name', 'N/A')}")
    print(f"Location: {restaurant.get('location', 'N/A')}")
    print(f"Type: {restaurant.get('type', 'N/A')}")
    print(f"Food Style: {restaurant.get('food_style', 'N/A')}")
    print(f"Rating: {restaurant.get('rating', 'N/A')}")
    print(f"Price Range: {restaurant.get('price_range', 'N/A')}")
    print(f"Vibe: {restaurant.get('vibe', 'N/A')}")
    print(f"Environment: {restaurant.get('environment', 'N/A')}")
    print(f"Signatures: {', '.join(restaurant.get('signatures', []))}")
    print(f"Shortcomings: {', '.join(restaurant.get('shortcomings', []))}")
    print("-" * 40)


# ===================== LLM Core Functions =====================
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    prompt = f"""
    You are a data extraction assistant.
    Extract information from the paragraph below and return ONLY valid JSON.
    NO extra text, NO markdown, NO explanation.

    JSON SCHEMA (FOLLOW EXACTLY):
    - itemId: 0 (will be replaced later)
    - name: restaurant name
    - location: location
    - type: restaurant type (e.g. upscale bistro)
    - food_style: cuisine style
    - rating: float number (e.g. 4.5)
    - price_range: integer (1-5)
    - signatures: list of signature dishes
    - vibe: atmosphere vibe
    - environment: environment description
    - shortcomings: list of shortcomings (empty list if none)

    Paragraph: {restaurant_paragraph}
    """
    return prompt.strip()


def llm_model(system_msg, prompt_txt, params=None):
    default_params = {"temperature": 0.1, "max_tokens": 1200}
    if params:
        default_params.update(params)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_txt}
        ],
        **default_params
    )
    return response.choices[0].message.content.strip()


def JSON_auto_repair_prompts(response, error_message):
    repair_prompt = f"""
    Invalid JSON: {error_message}
    Original: {response}
    Fix to valid JSON matching the required schema. Return ONLY JSON.
    """
    return llm_model("You are a JSON repair expert.", repair_prompt)


# ===================== Core Function: new_data_entry_process =====================
def new_data_entry_process(paragraph, itemId):
    """
    Process restaurant paragraph using DeepSeek LLM
    Return structured data matching your official JSON format
    """
    prompt = restaurant_data_structure_prompt_generation(paragraph)
    system_msg = "You extract structured restaurant data accurately."

    # Call LLM
    raw_response = llm_model(system_msg, prompt)

    # Parse JSON
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        repaired = JSON_auto_repair_prompts(raw_response, str(e))
        data = json.loads(repaired)

    # Override item ID
    data["itemId"] = itemId

    # Validate
    try:
        Restaurant(**data)
    except ValidationError as e:
        raise ValueError(f"Validation failed: {str(e)}")

    return data


# ===================== Main UI =====================
def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ").strip()

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                print(f"{i}: {res.get('name', 'N/A')}")

        elif choice == '2':
            try:
                idx = int(input("Enter index: "))
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("invalid index.")
            except ValueError:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            print("\n❗ SECURITY WARNING: Write mode.")
            confirm = input("Are you sure? (yes): ").lower()
            if confirm != "yes":
                print("Cancelled.")
                continue

            if choice == '3':
                item_id = 1000001 + len(data)
                para = input("\nEnter restaurant paragraph:\n")
                new_record = new_data_entry_process(para, item_id)
                data.append(new_record)
                save_data(file_path, data, backup_path)
                print("✅ Restaurant added.")

            elif choice == '4':
                try:
                    idx = int(input("Enter index to edit: "))
                    if 0 <= idx < len(data):
                        res = data[idx]
                        print("\nLeave blank to keep current value.")
                        for key in res:
                            current = res[key]
                            new_val = input(f"{key} ({current}): ").strip()
                            if new_val:
                                res[key] = new_val
                        save_data(file_path, data, backup_path)
                        print("✅ Updated.")
                except ValueError:
                    print("invalid input.")

            elif choice == '5':
                try:
                    idx = int(input("Enter index to delete: "))
                    if 0 <= idx < len(data):
                        data.pop(idx)
                        save_data(file_path, data, backup_path)
                        print("✅ Deleted.")
                except ValueError:
                    print("invalid input.")

        elif choice == '6':
            print("👋 Exiting...")
            break

        else:
            print("Invalid input.")


# ===================== RUN =====================
if __name__ == "__main__":
    manage_restaurants(FILEPATH, BACKUP_PATH)