"""
FOODWISE AI: Transparent Food Category & Edibility Mapping Configuration
Maps all 85 Roboflow PROJ3 Food Waste Dataset classes into:
1. Broad SDG 12 Food Waste Categories
2. Semantic Item Types (edible food, plate/utensil, garbage/waste, beverage)
"""

from typing import Dict, List, Set

# Broad Food Waste Categories (SDG 12 & Nutritional Groupings)
CATEGORIES: List[str] = [
    "Cereals and Starches",
    "Vegetables",
    "Fruits",
    "Protein and Meat",
    "Dairy Products",
    "Sauces and Condiments",
    "Non-edible Materials",
    "Beverages",
    "Other / Unclassified"
]

# Dashboard category color codes (Hex)
CATEGORY_COLORS: Dict[str, str] = {
    "Cereals and Starches": "#F57C00",    # Amber / Orange
    "Vegetables": "#388E3C",              # Forest Green
    "Fruits": "#E91E63",                  # Vibrant Pink
    "Protein and Meat": "#D32F2F",        # Deep Red
    "Dairy Products": "#0288D1",          # Light Blue
    "Sauces and Condiments": "#7B1FA2",   # Purple
    "Non-edible Materials": "#78909C",    # Slate Grey
    "Beverages": "#009688",               # Teal
    "Other / Unclassified": "#546E7A"     # Grey
}

# The complete 85 classes from PROJ3 version 6
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "apple", 1: "asparagus", 2: "bacon", 3: "baked potatoes", 4: "banana",
    5: "beans", 6: "black bean", 7: "blueberries", 8: "board", 9: "boiled egg",
    10: "bowl", 11: "bread", 12: "breaded", 13: "brocolis", 14: "cabbage",
    15: "cabidela rice", 16: "cake", 17: "carrot", 18: "cereals", 19: "cheese",
    20: "chicken", 21: "chicken steak", 22: "chips", 23: "chorizo", 24: "coffee",
    25: "coffee cup", 26: "cucumber", 27: "cup", 28: "cutlet", 29: "fish",
    30: "fish hake", 31: "fork", 32: "french fries", 33: "fried cod", 34: "fried egg",
    35: "garbage", 36: "gelatin", 37: "grape", 38: "greens", 39: "grilled chop",
    40: "grilled steak", 41: "ham", 42: "knife", 43: "lasagna", 44: "lettuce",
    45: "lime", 46: "mashed potatoes", 47: "meatballs", 48: "melon", 49: "minced meat",
    50: "mushrooms", 51: "mussel", 52: "olives", 53: "omelet", 54: "onion",
    55: "pasta", 56: "pineapple", 57: "pizza", 58: "plate", 59: "pork",
    60: "pork belly", 61: "pork intestines", 62: "pork loin", 63: "rice", 64: "salmon",
    65: "scrambled eggs", 66: "scrambled eggs with bacon", 67: "soup", 68: "soupbowl", 69: "spaghetti",
    70: "spoon", 71: "steak", 72: "steaks with mushrooms", 73: "stewed veal", 74: "strawberry",
    75: "toasted bread", 76: "tomato", 77: "tuna", 78: "tuna with mushrooms", 79: "tuna with pasta",
    80: "turkey steak", 81: "vegetables", 82: "water", 83: "water cup", 84: "watermelon"
}

NAME_TO_CLASS_ID: Dict[str, int] = {name: cid for cid, name in CLASS_ID_TO_NAME.items()}

# Mapping from class name to broad category
CLASS_NAME_TO_CATEGORY: Dict[str, str] = {
    # Cereals and Starches
    "rice": "Cereals and Starches",
    "cabidela rice": "Cereals and Starches",
    "bread": "Cereals and Starches",
    "toasted bread": "Cereals and Starches",
    "baked potatoes": "Cereals and Starches",
    "mashed potatoes": "Cereals and Starches",
    "french fries": "Cereals and Starches",
    "chips": "Cereals and Starches",
    "pasta": "Cereals and Starches",
    "spaghetti": "Cereals and Starches",
    "lasagna": "Cereals and Starches",
    "pizza": "Cereals and Starches",
    "cereals": "Cereals and Starches",
    "cake": "Cereals and Starches",

    # Vegetables
    "asparagus": "Vegetables",
    "beans": "Vegetables",
    "black bean": "Vegetables",
    "brocolis": "Vegetables",
    "cabbage": "Vegetables",
    "carrot": "Vegetables",
    "cucumber": "Vegetables",
    "greens": "Vegetables",
    "lettuce": "Vegetables",
    "mushrooms": "Vegetables",
    "olives": "Vegetables",
    "onion": "Vegetables",
    "tomato": "Vegetables",
    "vegetables": "Vegetables",

    # Fruits
    "apple": "Fruits",
    "banana": "Fruits",
    "blueberries": "Fruits",
    "grape": "Fruits",
    "lime": "Fruits",
    "melon": "Fruits",
    "pineapple": "Fruits",
    "strawberry": "Fruits",
    "watermelon": "Fruits",

    # Protein and Meat
    "bacon": "Protein and Meat",
    "boiled egg": "Protein and Meat",
    "breaded": "Protein and Meat",
    "chicken": "Protein and Meat",
    "chicken steak": "Protein and Meat",
    "chorizo": "Protein and Meat",
    "cutlet": "Protein and Meat",
    "fish": "Protein and Meat",
    "fish hake": "Protein and Meat",
    "fried cod": "Protein and Meat",
    "fried egg": "Protein and Meat",
    "grilled chop": "Protein and Meat",
    "grilled steak": "Protein and Meat",
    "ham": "Protein and Meat",
    "meatballs": "Protein and Meat",
    "minced meat": "Protein and Meat",
    "mussel": "Protein and Meat",
    "omelet": "Protein and Meat",
    "pork": "Protein and Meat",
    "pork belly": "Protein and Meat",
    "pork intestines": "Protein and Meat",
    "pork loin": "Protein and Meat",
    "salmon": "Protein and Meat",
    "scrambled eggs": "Protein and Meat",
    "scrambled eggs with bacon": "Protein and Meat",
    "steak": "Protein and Meat",
    "steaks with mushrooms": "Protein and Meat",
    "stewed veal": "Protein and Meat",
    "tuna": "Protein and Meat",
    "tuna with mushrooms": "Protein and Meat",
    "tuna with pasta": "Protein and Meat",
    "turkey steak": "Protein and Meat",

    # Dairy Products
    "cheese": "Dairy Products",
    "gelatin": "Dairy Products",

    # Sauces and Condiments
    "soup": "Sauces and Condiments",

    # Beverages
    "coffee": "Beverages",
    "water": "Beverages",

    # Non-edible Utensils & Tableware
    "plate": "Non-edible Materials",
    "board": "Non-edible Materials",
    "bowl": "Non-edible Materials",
    "coffee cup": "Non-edible Materials",
    "cup": "Non-edible Materials",
    "fork": "Non-edible Materials",
    "knife": "Non-edible Materials",
    "soupbowl": "Non-edible Materials",
    "spoon": "Non-edible Materials",
    "water cup": "Non-edible Materials",

    # Waste / Discarded Material
    "garbage": "Non-edible Materials"
}

# Semantic Item Types as defined by the research paper methodology:
# - PLATE: Reference container for area calculation (class 58)
# - GARBAGE: Non-edible discarded item (napkins, bones, etc. - class 35)
# - IGNORE: Cutlery, cups, table elements that are not part of leftover edible food
# - EDIBLE_FOOD: Actual edible leftovers (food waste numerator)

PLATE_CLASS_ID: int = 58  # 'plate'
GARBAGE_CLASS_IDS: Set[int] = {35}  # 'garbage'

# Non-edible utensils & tableware to ignore from food area
UTENSIL_CLASS_IDS: Set[int] = {
    8,   # board
    10,  # bowl
    25,  # coffee cup
    27,  # cup
    31,  # fork
    42,  # knife
    58,  # plate
    68,  # soupbowl
    70,  # spoon
    83   # water cup
}

def get_category_for_class(class_id_or_name) -> str:
    """Returns the broad food waste category for a class ID or class name."""
    if isinstance(class_id_or_name, int):
        name = CLASS_ID_TO_NAME.get(class_id_or_name, "Other / Unclassified")
    else:
        name = str(class_id_or_name).strip().lower()
    return CLASS_NAME_TO_CATEGORY.get(name, "Other / Unclassified")

def is_edible_food(class_id: int) -> bool:
    """Returns True if the class corresponds to consumable food leftovers."""
    if class_id == PLATE_CLASS_ID:
        return False
    if class_id in GARBAGE_CLASS_IDS:
        return False
    if class_id in UTENSIL_CLASS_IDS:
        return False
    # Beverages (water, coffee) in cups are usually not considered solid food leftovers
    name = CLASS_ID_TO_NAME.get(class_id, "")
    if name in ["water", "coffee"]:
        return False
    return True

def get_item_semantic_type(class_id: int) -> str:
    """
    Returns one of: 'Plate', 'Garbage', 'Utensil', 'Edible Food', or 'Beverage'.
    """
    if class_id == PLATE_CLASS_ID:
        return "Plate"
    if class_id in GARBAGE_CLASS_IDS:
        return "Garbage"
    if class_id in UTENSIL_CLASS_IDS:
        return "Utensil"
    name = CLASS_ID_TO_NAME.get(class_id, "")
    if name in ["water", "coffee"]:
        return "Beverage"
    return "Edible Food"

