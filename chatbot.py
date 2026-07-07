'''
Travel Booking Chatbot - Main Module
This module implements a conversational chatbot for booking flights and hotels,
managing bookings, and providing travel recommendations.
'''

import json
import re
import os
import random

from config import CHATBOT_NAME, DATA_DIR, TEMPLATES, user_state
from utils import (
    load_flights, load_hotels, save_booking_pickle, load_bookings,
    update_booking_name, extract_city_from_input, extract_location_from_input,
    extract_flight_details, extract_name, get_intent, handle_qa,
    initialize_intent_classifier, initialize_qa_system,
    parse_valid_date, cancel_user_booking, 
    has_active_booking, handle_item_selection, get_time_of_day_greeting,          
)

CONTEXT_FILE = "data/incomplete_bookings.json"

# --- Initialise NLP systems with classifier --- #

intent_vectorizer, intent_classifier = initialize_intent_classifier()
qa_df, qa_vectorizer, qa_counts, qa_tfidf_vectorizer, qa_tfidf_counts = initialize_qa_system(DATA_DIR)


# --- Utility Functions --- #

# Generic confirmation handler for canceling bookings and returning to main menu
def handle_return_to_main_menu(context_name):
    confirm = input(f"{CHATBOT_NAME}: Are you sure you want to cancel this {context_name} booking and go back to the main menu? (yes/no)\nYou: ").strip().lower()
    
    if confirm in ["yes", "y"]:
        saved_key = f"saved_{context_name}_context"
        user_state[saved_key] = dict(user_state[f"{context_name}_context"])

        if user_state.get("name"):
            save_progress_to_file(user_state["name"], context_name, user_state[saved_key])

        keys_to_reset = user_state[f"{context_name}_context"].keys()
        reset_dict = {}
        for k in keys_to_reset:
            if "available" in k: 
                reset_dict[k] = []
            else:
                reset_dict[k] = None
        
        user_state[f"{context_name}_context"] = reset_dict

        if context_name == "flight":
            user_state["flight_context"]["return_flight"] = None

        user_state.update({
            "awaiting_profile_choice": True,
            "awaiting_post_flight_choice": False,
            "awaiting_post_view_choice": False
        })
        print(f"{CHATBOT_NAME}: {context_name.capitalize()} booking cancelled. Back at main menu. What would you like to do next?")
        return True

    print(f"{CHATBOT_NAME}: Ok, we'll continue with your {context_name} booking.")
    return False

# --- Core Handlers --- #

# Generates appropriate greeting based on time of day and user state
def handle_greeting():
    time_greeting = get_time_of_day_greeting()
    
    if user_state.get("name"):
        return TEMPLATES["greeting_known"].format(time_greeting=time_greeting, user_name=user_state["name"])
    
    return TEMPLATES["intro"].format(time_greeting=time_greeting, bot_name=CHATBOT_NAME)

# Handles user identity establishment and name changes by extracting name from input, updates booking records, and checks for existing user
def handle_identity(user_input):
    user_input_lower = user_input.lower()

    if re.search(r'\b(my name|who am i)\b', user_input_lower):
        if user_state.get("name"):
            return f"Your name is {user_state['name']}."
        else:
            return "I don't know your name yet. What's your name?"

    name_candidate = extract_name(user_input)
    if not name_candidate:
        return TEMPLATES["unknown_name"]

    old_name = user_state.get("name")
    if not old_name or old_name.lower() != name_candidate.lower():
        user_state["name"] = name_candidate.capitalize()
        if old_name:
            update_booking_name(old_name, user_state["name"])

        bookings = load_bookings()
        existing_user = any(b.get("user") == user_state["name"] for b in bookings)
        user_state["existing_user"] = existing_user
        
        user_state["awaiting_profile_choice"] = True

        if existing_user:
            return (
                f"Welcome back, {user_state['name']}! I found your profile.\n"
                "What would you like to do next? You can just tell me — for example, book a flight, book a hotel, or see my bookings."
            )
        else:
            return f"Nice to meet you, {user_state['name']}! Would you like to start by booking a flight or a hotel?"

    return f"We already have your name as {user_state['name']}."

# Persist incomplete booking data to JSON file for later resumption
def save_progress_to_file(user_name, context_type, context_data):
    """Saves incomplete booking data to a JSON file."""
    data = {}
    if os.path.exists(CONTEXT_FILE):
        try:
            with open(CONTEXT_FILE, 'r') as f:
                data = json.load(f)
        except: pass
    
    if user_name not in data:
        data[user_name] = {}
    
    data[user_name][context_type] = context_data
    
    with open(CONTEXT_FILE, 'w') as f:
        json.dump(data, f)

# Load previously saved incomplete booking data for a user
def load_progress_from_file(user_name):
    if not os.path.exists(CONTEXT_FILE):
        return
    try:
        with open(CONTEXT_FILE, 'r') as f:
            data = json.load(f)
        
        if user_name in data:
            if "flight" in data[user_name] and data[user_name]["flight"]:
                user_state["saved_flight_context"] = data[user_name]["flight"]
            
            if "hotel" in data[user_name] and data[user_name]["hotel"]:
                user_state["saved_hotel_context"] = data[user_name]["hotel"]
    except:
        pass

# Provide friendly, randomized responses to small talk using templates from config and personalises with user's name
def handle_small_talk():
    """
    Provides a friendly and randomized response for small talk, 
    using templates and the user's name if available.
    """
    responses = TEMPLATES.get("small_talk_responses", ["I'm doing great, {user_name}!"])
    template = random.choice(responses)
    
    user_name = user_state.get("name", "traveler")
    
    try:
        response = template.format(user_name=user_name)
    except KeyError:
        response = template.replace("{user_name}", user_name)
        
    return response

# Provide city-specific travel recommendations
def handle_recommendations(user_input):    
    city = extract_city_from_input(user_input)
    
    if not city:
        user_state["awaiting_city_for_recommendations"] = True
        return "Which city would you like recommendations for?"
    
    city_lower = city.lower()
    
    user_state["awaiting_city_for_recommendations"] = False
    
    if city_lower in TEMPLATES.get("city_recommendations", {}):
        recommendations = TEMPLATES["city_recommendations"][city_lower]
        response = f"Here are some great things to do in {city}:\n\n"
        
        for category, activities in recommendations.items():
            response += f"**{category}:**\n"
            for activity in activities:
                response += f"  • {activity}\n"
            response += "\n"
        
        return response + "Would you like to book a flight or hotel to visit?"
    else:
        return f"I don't have specific recommendations for {city} yet, but it sounds like a wonderful destination! Would you like to book travel there?"

# --- Flight Booking Handlers --- #

# Initialise or continue flight booking process by extracting flight details from user input and determines next step
def handle_flight_booking(user_input):
    fc = user_state["flight_context"]
    
    details = extract_flight_details(user_input)

    if details.get("departure_city") and not fc["departure_city"]:
        fc["departure_city"] = details["departure_city"]
    if details.get("arrival_city") and not fc["arrival_city"]:
        fc["arrival_city"] = details["arrival_city"]
    if details.get("date") and not fc["date"]:
        fc["date"] = details["date"]
    if details.get("num_passengers") and not fc["num_passengers"]:
        fc["num_passengers"] = details["num_passengers"]
    
    normalized = user_input.lower()
    if fc.get("return_flight") is None:
        if "return" in normalized or "round" in normalized or "two way" in normalized:
            fc["return_flight"] = True
        elif "one way" in normalized or "single" in normalized:
            fc["return_flight"] = False
    
    has_all_info = (
        fc["departure_city"] and 
        fc["arrival_city"] and 
        fc["date"] and 
        fc["num_passengers"] and 
        fc.get("return_flight") is not None
    )
    
    if has_all_info:
        return show_available_flights()
    
    if not fc["departure_city"]:
        fc["stage"] = "departure"
        return TEMPLATES["departure_request"]
    
    if not fc["arrival_city"]:
        fc["stage"] = "arrival"
        return TEMPLATES["arrival_request"]

    if fc.get("return_flight") is None:
        fc["stage"] = "return_type"
        return "Would you like a return flight or a one-way ticket?"

    if not fc["date"]:
        fc["stage"] = "date"
        return TEMPLATES["date_request"]
    
    if not fc["num_passengers"]:
        fc["stage"] = "passengers"
        return TEMPLATES["passenger_request"]

    return show_available_flights()

# Continues the flight booking conversation based on current stage. Handles all stages: departure, arrival, return type, date, passengers, selection, confirmation
def continue_booking_flow(user_input):
    fc = user_state["flight_context"]
    normalised = user_input.lower().strip()
    
    if not user_input:
        if fc["stage"] == "departure" or not fc["departure_city"]:
            return TEMPLATES["departure_request"]
        elif fc["stage"] == "arrival" or not fc["arrival_city"]:
            return TEMPLATES["arrival_request"]
        elif fc["stage"] == "return_type" or fc.get("return_flight") is None:
            return "Would you like a return flight or a one-way ticket?"
        elif fc["stage"] == "date" or not fc["date"]:
            return TEMPLATES["date_request"]
        elif fc["stage"] == "passengers" or not fc["num_passengers"]:
            return TEMPLATES["passenger_request"]
        elif fc["stage"] == "flight_selection":
            return "Please select a flight from the options above."
        elif fc["stage"] == "confirmation":
            return f"Shall we confirm the booking for flight {fc['selected_flight']['flight_number']}? (yes/no)"
    
    intent = get_intent(user_input, intent_vectorizer, intent_classifier)

    if intent == "return_main_menu" or normalised in ["main menu", "go back", "menu", "back", "home"] or "cancel" in normalised:
        if handle_return_to_main_menu("flight"):
            return None

    if fc["stage"] == "confirmation":
        if user_input.lower() in ["yes", "y", "confirm"]:
            booking = {
                "type": "flight",
                "user": user_state["name"],
                "flight": fc["selected_flight"],
                "passengers": fc["num_passengers"],
                "date": fc["date"],
                "return_flight": fc.get("return_flight", False)
            }
            save_booking_pickle(booking)
            user_state["booked_flight"] = booking
            
            user_state["flight_context"] = {k: None for k in fc}
            user_state["flight_context"]["return_flight"] = None
            user_state["saved_flight_context"] = {} 
            
            if user_state.get("name"):
                save_progress_to_file(user_state["name"], "flight", {})

            user_state["awaiting_post_flight_choice"] = True

            flight_type = "return flight" if booking.get("return_flight") else "one-way flight"
            response = f"All set! {booking['passengers']} passenger(s) are booked on a {flight_type} to {booking['flight']['arrival_city']}. Bon voyage!"
            return response + "\n\nWhat would you like to do next? Would you like to book a hotel, view your bookings, or return to the main menu?"
        
        fc["stage"] = "flight_selection"
        return TEMPLATES["booking_reselect"]

    if fc["stage"] == "flight_selection":
        return handle_item_selection(user_input, fc, fc["available_flights"], "flight", TEMPLATES, ["flight_number", "airline", "class", "price"])

    if fc["stage"] == "return_type":
        if any(word in normalised for word in ["return", "round", "two way", "yes", "y"]):
            fc["return_flight"] = True
            if not fc["date"]:
                fc["stage"] = "date"
                return TEMPLATES["date_request"]
            elif not fc["num_passengers"]:
                fc["stage"] = "passengers"
                return TEMPLATES["passenger_request"]
            else:
                return show_available_flights()
        elif any(word in normalised for word in ["one way", "single", "no", "n"]):
            fc["return_flight"] = False
            if not fc["date"]:
                fc["stage"] = "date"
                return TEMPLATES["date_request"]
            elif not fc["num_passengers"]:
                fc["stage"] = "passengers"
                return TEMPLATES["passenger_request"]
            else:
                return show_available_flights()
        return "Please say 'return' for a round trip or 'one way' for a single journey."

    if fc["stage"] in ["departure", "arrival"]:
        details = extract_flight_details(user_input)
        city = details["departure_city"] if fc["stage"] == "departure" else details["arrival_city"]
        city = city or extract_city_from_input(user_input)
        if not city:
            return "I don't recognize that city. Please enter a valid city."

        fc[f"{fc['stage']}_city"] = city
        
        if fc["departure_city"] and fc["arrival_city"]:
            if fc.get("return_flight") is None:
                fc["stage"] = "return_type"
                return "Would you like a return flight or a one-way ticket?"
            elif not fc["date"]:
                fc["stage"] = "date"
                return TEMPLATES["date_request"]
            elif not fc["num_passengers"]:
                fc["stage"] = "passengers"
                return TEMPLATES["passenger_request"]
            else:
                return show_available_flights()
        
        fc["stage"] = "arrival"
        return TEMPLATES["arrival_request"]

    if fc["stage"] == "date":
        match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})\b', user_input)
        if not match:
            return "Please provide a date (e.g. 2025-12-31 or 31-12-2025)."
        if not parse_valid_date(match.group(1)):
            return TEMPLATES["invalid_date"]
        fc["date"] = match.group(1)
        
        if not fc["num_passengers"]:
            fc["stage"] = "passengers"
            return TEMPLATES["passenger_request"]
        else:
            return show_available_flights()

    if fc["stage"] == "passengers":
        num_match = re.search(r'\b(\d+)\b', user_input)
        if num_match:
            num = int(num_match.group(1))
            if 1 <= num <= 9:
                fc["num_passengers"] = num
                return show_available_flights()
        return "Please tell me how many passengers (e.g. '2 people')."

    return "I'm not sure what you meant. Can you try again?"

# Query and display available flights based on user's criteria
def show_available_flights():
    fc = user_state["flight_context"]
    flights = load_flights()

    iso_date = None
    if fc.get("date"):
        if re.match(r'^\d{4}-\d{2}-\d{2}$', fc["date"]):
            iso_date = fc["date"]
        elif re.match(r'^\d{2}-\d{2}-\d{4}$', fc["date"]):
            d, m, y = fc["date"].split("-")
            iso_date = f"{y}-{m}-{d}"

    available = [
        f for f in flights
        if f["departure_city"].lower() == fc["departure_city"].lower()
        and f["arrival_city"].lower() == fc["arrival_city"].lower()
        and (not iso_date or f["departure_time"].startswith(iso_date))
    ]

    if not available:
        user_state["flight_context"] = {k: None for k in fc}
        return TEMPLATES["no_flights"]

    fc["available_flights"] = available
    fc["stage"] = "flight_selection"
    lines = [TEMPLATES["flight_options"].format(**fc)]
    for i, f in enumerate(available, 1):
        lines.append(f"\n{i}. {f['airline']} - {f['flight_number']}\n   Departure: {f['departure_time']} | Arrival: {f['arrival_time']}\n   Class: {f['class']} | Price: £{f['price']}")
    return "\n".join(lines) + "\n\nPlease pick an option — e.g. 'book flight 1'."

# --- Hotel Booking Handlers --- #

# Initialises hotel booking with pre-filled data from flight if available
def handle_hotel_booking(user_input=None):

    hc = user_state["hotel_context"]

    if user_state.get("booked_flight") and not hc["location"]:
        booked = user_state["booked_flight"]["flight"]
        hc.update({
            "location": booked["arrival_city"],
            "num_guests": user_state["booked_flight"]["passengers"],
            "check_in": user_state["booked_flight"]["date"],
        })

    if user_input:
        loc = extract_location_from_input(user_input)
        if loc and not hc["location"]:
            hc["location"] = loc
        
        date_matches = re.findall(r'\b(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b', user_input)
        if date_matches:
            if not hc["check_in"] and len(date_matches) >= 1:
                check_in_dt = parse_valid_date(date_matches[0])
                if check_in_dt:
                    hc["check_in"] = check_in_dt.strftime("%Y-%m-%d")
            
            if not hc["check_out"] and len(date_matches) >= 2:
                check_out_dt = parse_valid_date(date_matches[1])
                if check_out_dt:
                    hc["check_out"] = check_out_dt.strftime("%Y-%m-%d")
        
        guest_match = re.search(r'(?:for\s+)?(\d+)\s+(?:guests?|people|person)', user_input, re.IGNORECASE)
        if guest_match and not hc["num_guests"]:
            guests = int(guest_match.group(1))
            if 1 <= guests <= 10:
                hc["num_guests"] = guests

    if not hc["location"]:
        hc["stage"] = "location"
        return TEMPLATES["hotel_location_request"]
    
    if not hc["check_in"]: 
        hc["stage"] = "check_in"
        return TEMPLATES["hotel_checkin_request"]
    
    if not hc["check_out"]:
        hc["stage"] = "check_out"
        return TEMPLATES["hotel_checkout_request"]
    
    if not hc["num_guests"]:
        hc["stage"] = "guests"
        return TEMPLATES["hotel_guests_request"]

    hc["stage"] = "hotel_selection"
    return show_available_hotels()

# Continues the hotel booking conversation based on current stage. Handles all stages: location, check-in, check-out, guests, selection, confirmation
def handle_hotel_booking_flow(user_input=None):
    hc = user_state["hotel_context"]
    name = user_state.get("name", "traveler")
    normalised = user_input.lower().strip() if user_input else ""

    if normalised in ["main menu", "menu", "back", "home"] or "cancel" in normalised:
        if handle_return_to_main_menu("hotel"):
            return None 

    if user_state.get("booked_flight") and not hc["location"] and hc.get("stage") in [None, "location"]:
        booked = user_state["booked_flight"]["flight"]
        hc.update({
            "location": booked["arrival_city"],
            "num_guests": user_state["booked_flight"]["passengers"],
            "check_in": user_state["booked_flight"]["date"]
        })
        hc["stage"] = "check_out"
        return (
            f"{name}, since you're flying into {hc['location']}, I'll look for hotels there.\n"
            f"{TEMPLATES['hotel_checkout_request']}"
        )

    stage = hc.get("stage", "location")

    if user_input and stage in ["location", "check_in", "check_out", "num_guests"]:
        loc = extract_location_from_input(user_input)
        if loc and not hc["location"]:
            hc["location"] = loc
        
        date_matches = re.findall(r'\b(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b', user_input)
        if date_matches:
            if not hc["check_in"] and len(date_matches) >= 1:
                check_in_dt = parse_valid_date(date_matches[0])
                if check_in_dt:
                    hc["check_in"] = check_in_dt.strftime("%Y-%m-%d")
            
            if not hc["check_out"] and len(date_matches) >= 2:
                check_out_dt = parse_valid_date(date_matches[1])
                check_in_dt = parse_valid_date(hc["check_in"]) if hc.get("check_in") else None
                if check_out_dt and check_in_dt and check_out_dt > check_in_dt:
                    hc["check_out"] = check_out_dt.strftime("%Y-%m-%d")
        
        guest_match = re.search(r'(?:for\s+)?(\d+)\s+(?:guests?|people|person)', user_input, re.IGNORECASE)
        if guest_match and not hc["num_guests"]:
            guests = int(guest_match.group(1))
            if 1 <= guests <= 10:
                hc["num_guests"] = guests

    # --- Location Stage
    if stage == "location":
        if hc["location"]:
            hc["stage"] = "check_in" if not hc["check_in"] else ("check_out" if not hc["check_out"] else ("num_guests" if not hc["num_guests"] else "hotel_selection"))
            if hc["stage"] == "check_in":
                return f"{name}, you want to stay in {hc['location']}? Great choice.\n{TEMPLATES['hotel_checkin_request']}"
            elif hc["stage"] == "check_out":
                return f"Checking in on {hc['check_in']}. {TEMPLATES['hotel_checkout_request']}"
            elif hc["stage"] == "num_guests":
                return f"Checking out on {hc['check_out']}. {TEMPLATES['hotel_guests_request']}"
            elif hc["stage"] == "hotel_selection":
                return show_available_hotels()
        return TEMPLATES["hotel_location_request"]

    if stage == "check_in":
        if hc["check_in"]:
            hc["stage"] = "check_out" if not hc["check_out"] else ("num_guests" if not hc["num_guests"] else "hotel_selection")
            if hc["stage"] == "check_out":
                return f"Checking in on {hc['check_in']}. {TEMPLATES['hotel_checkout_request']}"
            elif hc["stage"] == "num_guests":
                return f"Checking out on {hc['check_out']}. {TEMPLATES['hotel_guests_request']}"
            elif hc["stage"] == "hotel_selection":
                return show_available_hotels()
        else:
            check_in_dt = parse_valid_date(user_input) 
            if check_in_dt:
                hc["check_in"] = check_in_dt.strftime("%Y-%m-%d") 
                hc["stage"] = "check_out"
                return f"Checking in on {hc['check_in']}. {TEMPLATES['hotel_checkout_request']}"
        return TEMPLATES["hotel_checkin_request"]

    if stage == "check_out":
        if hc["check_out"]:
            hc["stage"] = "num_guests" if not hc["num_guests"] else "hotel_selection"
            if hc["stage"] == "num_guests":
                return f"Checking out on {hc['check_out']}. {TEMPLATES['hotel_guests_request']}"
            elif hc["stage"] == "hotel_selection":
                return show_available_hotels()
        else:
            check_out_dt = parse_valid_date(user_input)
            if check_out_dt:
                check_in_dt = parse_valid_date(hc["check_in"])
                if check_out_dt and check_out_dt > check_in_dt:
                    hc["check_out"] = check_out_dt.strftime("%Y-%m-%d")
                    hc["stage"] = "num_guests"
                    return f"Checking out on {hc['check_out']}. {TEMPLATES['hotel_guests_request']}"
                return TEMPLATES["hotel_invalid_checkout"] 
        return TEMPLATES["hotel_checkout_request"]

    if stage == "num_guests":
        if hc["num_guests"]:
            hc["stage"] = "hotel_selection"
            return f"{hc['num_guests']} guests — perfect.\n{show_available_hotels()}"
        else:
            match = re.search(r'\d+', user_input)
            if match:
                guests = int(match.group())
                if 1 <= guests <= 10:
                    hc["num_guests"] = guests
                    hc["stage"] = "hotel_selection"
                    return f"{guests} guests — perfect.\n{show_available_hotels()}"
        return TEMPLATES["hotel_guests_request"]

    if stage == "hotel_selection":
        available = hc.get("available_hotels", [])
        if not available:
            return f"Sorry {name}, no hotels found in {hc.get('location')}."

        ordinal_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10
        }
        match = re.search(r'\b(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b', normalised)
        if match:
            token = match.group(1)
            index = int(token) if token.isdigit() else ordinal_map.get(token, None)
            if index and 1 <= index <= len(available):
                hc["selected_hotel"] = available[index - 1]
                hc["stage"] = "confirmation"
                return TEMPLATES["hotel_selected"].format(**hc["selected_hotel"])

        hotel = next((h for h in available if h["name"].lower() in normalised), None)
        if hotel:
            hc["selected_hotel"] = hotel
            hc["stage"] = "confirmation"
            return TEMPLATES["hotel_selected"].format(**hotel)

        return "Please pick a hotel by number or name."

    if stage == "confirmation":
        if normalised in ["yes", "y", "confirm"]:
            booking = {
                "type": "hotel",
                "user": name,
                "hotel": hc["selected_hotel"],
                "location": hc["location"],
                "check_in": hc["check_in"],
                "check_out": hc["check_out"],
                "guests": hc["num_guests"]
            }
            save_booking_pickle(booking)
            user_state["booked_hotel"] = booking
 
            user_state["hotel_context"] = {
                "location": None,
                "check_in": None,
                "check_out": None,
                "num_guests": None,
                "available_hotels": [],
                "selected_hotel": None,
                "stage": None
            }
            user_state["saved_hotel_context"] = {} 
            
            user_state["awaiting_post_view_choice"] = True
            
            return f"{booking['hotel']['name']} has been booked successfully!\nWhat would you like to do next? You can view your bookings, book another flight or hotel, or return to the main menu."
        else:
            hc["stage"] = "hotel_selection"
            return "Okay, let's pick another hotel.\n" + show_available_hotels()

    return show_available_hotels()

# Displays available hotels based on user's criteria
def show_available_hotels():
    hotels = load_hotels()
    hc = user_state["hotel_context"]
    name = user_state.get("name", "traveler")
    location = hc.get("location")

    if not location:
        return f"{name}, I need a location to show hotels. Where would you like to stay?"

    available = [h for h in hotels if h.get("location", "").lower() == location.lower()]
    hc["available_hotels"] = available

    if not available:
        return f"Sorry {name}, no hotels available in {location}."

    lines = [f"{name}, here are some hotels I found in {location}:\n"]
    for i, h in enumerate(available, 1):
        amenities = ", ".join(h.get("amenities", []))
        lines.append(
            f"{i}. {h.get('name', 'Unknown')} — {h.get('rating', 'N/A')} ★\n"
            f"   Price: £{h.get('price_per_night', 'N/A')}/night\n"
            f"   Amenities: {amenities}\n"
            f"   Available rooms: {h.get('available_rooms', 'N/A')}\n"
        )
    lines.append("Please choose a hotel by number or name.")
    return "\n".join(lines)

# --- View/Cancel Handlers --- #

# Displays user's current bookings with options to cancel or rebook
def handle_view_bookings():
    bookings = load_bookings()
    user_name = user_state.get("name", "traveler")
    user_bookings = [b for b in bookings if b.get("user") == user_name]

    if not user_bookings:
        return f"{user_name}, it looks like you don't have any bookings yet. Where shall we start by finding a flight or a hotel?"

    response = [f"{user_name}, here's a summary of your current bookings:\n"]
    has_flight, has_hotel = False, False

    for i, b in enumerate(user_bookings, 1):
        if b["type"] == "flight":
            has_flight = True
            f = b["flight"]
            response.append(
                f"{i}. Flight to {f['arrival_city']} with {f['airline']} (Flight {f['flight_number']})\n"
                f"   Date: {b['date']}, Passengers: {b['passengers']}\n"
            )
        elif b["type"] == "hotel":
            has_hotel = True
            h = b["hotel"]
            response.append(
                f"{i}. Hotel stay at {h['name']} in {b['location']}\n"
                f"   Check-in: {b['check_in']}, Check-out: {b['check_out']}, Guests: {b['guests']}\n"
            )

    if has_flight or has_hotel:
        user_state.update({"awaiting_post_view_choice": True, "has_flight": has_flight, "has_hotel": has_hotel})

    response.append(
        "\nWhat would you like to do next? You can cancel any of your bookings, "
        "or head back to the main menu — just tell me what you feel like doing."
    )

    return "".join(response)

# Handles booking cancellation flow with confirmation and post-cancellation options
def cancel_booking_flow(user_input):
    normalised = user_input.lower()
    user_name = user_state["name"]

    booking_type = None
    if "flight" in normalised:
        booking_type = "flight"
    elif "hotel" in normalised:
        booking_type = "hotel"

    if not booking_type:
        return "Please specify which booking you'd like to cancel — flight or hotel."

    confirm = input(
        f"{CHATBOT_NAME}: Are you sure you want to cancel your {booking_type}? (yes/no)\nYou: "
    ).strip().lower()

    if confirm not in ["yes", "y"]:
        return f"Ok, your {booking_type} booking was not cancelled."

    if cancel_user_booking(user_name, booking_type):
        user_state["last_cancelled_booking"] = booking_type
        user_state["awaiting_post_view_choice"] = True
        return (
            f"Your {booking_type} booking has been cancelled.\n"
            f"What would you like to do next?\n"
            f"- Rebook your {booking_type}\n"
            f"- See your bookings\n"
            f"- Return to main menu"
        )
    else:
        return f"You don't have a {booking_type} booking to cancel."

# Continues the post-view bookings choice flow based on user input
def continue_post_view_choice(user_input):
    normalised = user_input.lower().strip()
    last_cancelled = user_state.get("last_cancelled_booking")
    intent = get_intent(user_input, intent_vectorizer, intent_classifier)

    if "cancel" in normalised:
        return cancel_booking_flow(user_input)

    if last_cancelled and ("rebook" in normalised or f"book {last_cancelled}" in normalised):
        user_state["awaiting_post_view_choice"] = False

        if last_cancelled == "flight":
            user_state["flight_context"] = {
                "departure_city": None,
                "arrival_city": None,
                "date": None,
                "num_passengers": None,
                "return_flight": None,
                "available_flights": [],
                "selected_flight": None,
                "stage": "departure"
            }
            return TEMPLATES["departure_request"]

        elif last_cancelled == "hotel":
            user_state["hotel_context"] = {
                "location": None,
                "check_in": None, 
                "check_out": None,
                "num_guests": None,
                "available_hotels": [],
                "selected_hotel": None,
                "stage": "location"
            }
            return TEMPLATES["hotel_location_request"]

    if intent == "book_flight":
        if has_active_booking(user_state.get("name"), "flight"):
            return f"{user_state.get('name')}, you already have an active flight booking."
        user_state["awaiting_post_view_choice"] = False
        user_state["flight_context"]["stage"] = "departure"
        return handle_flight_booking(user_input)
    
    if intent == "book_hotel" or "hotel" in normalised:
        if has_active_booking(user_state.get("name"), "hotel"):
            return f"{user_state.get('name')}, you already have an active hotel booking."
        user_state["awaiting_post_view_choice"] = False
        user_state["hotel_context"]["stage"] = "location"
        return handle_hotel_booking_flow(user_input)

    if "view" in normalised or "bookings" in normalised:
        return handle_view_bookings()

    if "main menu" in normalised or "home" in normalised or "back" in normalised:
        user_state.update({
            "awaiting_post_view_choice": False,
            "awaiting_profile_choice": True
        })
        return "Back at main menu. What would you like to do next?"

    return (
        "You can:\n"
        "- Book a flight or hotel\n"
        "- Cancel a booking\n"
        "- Rebook your last cancelled booking\n"
        "- See your bookings\n"
        "- Return to main menu"
    )

# Continues the post-flight booking choice flow based on user input
def continue_post_flight_choice(user_input):
    normalised = user_input.lower().strip()
    intent = get_intent(user_input, intent_vectorizer, intent_classifier)
    
    if "hotel" in normalised or intent == "book_hotel":
        user_state["awaiting_post_flight_choice"] = False
        return handle_hotel_booking_flow(user_input)
    
    if "view" in normalised or "booking" in normalised or intent == "view_bookings":
        user_state["awaiting_post_flight_choice"] = False
        user_state["awaiting_post_view_choice"] = True  # NEW: Set the correct flag
        return handle_view_bookings()
    
    if "cancel" in normalised:
        if cancel_user_booking(user_state["name"], "flight"):
            user_state["awaiting_post_flight_choice"] = False
            return "Your flight booking has been cancelled."
        return "You don't have a flight booking to cancel."
    
    if "main menu" in normalised or "home" in normalised:
        user_state.update({
            "awaiting_post_flight_choice": False,
            "awaiting_profile_choice": True,
        })
        return "Back at main menu. What would you like to do next?"
    
    return "You can say 'book a hotel', 'view my bookings', 'cancel flight', or 'main menu'."

# Checks for saved incomplete booking and prompts user to resume if found
def check_and_resume_booking(booking_type):
    saved_key = f"saved_{booking_type}_context"
    context_key = f"{booking_type}_context"

    if user_state.get(saved_key) and any(user_state[saved_key].values()):
        resume = input(f"{CHATBOT_NAME}: I noticed you have an incomplete {booking_type} booking. Would you like to resume it? (yes/no)\nYou: ").strip().lower()
        if resume in ["yes", "y"]:

            user_state[context_key] = dict(user_state[saved_key])
            user_state[saved_key] = {}
            user_state["awaiting_profile_choice"] = False

            if booking_type == "flight":
                print(f"{CHATBOT_NAME}: Resuming your flight booking...")
                print(f"{CHATBOT_NAME}: {continue_booking_flow('')}")
            else:
                print(f"{CHATBOT_NAME}: Resuming your hotel booking...")
                print(f"{CHATBOT_NAME}: {handle_hotel_booking_flow('')}")
            return True

    return False

# --- Main chatbot loop --- #
# Main chatbot function that handles travel booking conversations. Manages flight bookings, hotel reservations, user profiles, and general queries.
def chatbot():
    exit_phrases = {"exit": "Goodbye", "quit": "Goodbye", "goodbye": "See you later!"}

    time_greeting = get_time_of_day_greeting()
    print(TEMPLATES["intro"].format(time_greeting=time_greeting, bot_name=CHATBOT_NAME))
    
    last_intent = None
    conversation_context = {"mood": "neutral", "topic": None}

    # --- Main interaction loop --- #

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        user_input_lower = user_input.lower()
        user_name = user_state.get("name", "there")

        # --- Exit Handling --- #
        if user_input_lower in exit_phrases:
            print(f"{CHATBOT_NAME}: {exit_phrases[user_input_lower]}, {user_name}! Safe travels!")
            break

        # --- City recommendations sub-flow --- #

        if user_state.get("awaiting_city_for_recommendations"):
            city = extract_city_from_input(user_input)
            if city:
                user_state["awaiting_city_for_recommendations"] = False
                city_lower = city.capitalize() 
                
                recs_dict = TEMPLATES["city_recommendations"].get(city_lower)

                if recs_dict:
                    response = f"Here are some great things to do in {city_lower}:\n\n"
                    
                    for category, activities in recs_dict.items():
                        response += f"**{category}:**\n"
                        for activity in activities:
                            response += f"  • {activity}\n"
                        response += "\n"
                    
                    print(f"{CHATBOT_NAME}: {response}Would you like to book a flight or hotel to visit?")
                else:
                    print(f"{CHATBOT_NAME}: I don't have specific recommendations for {city_lower} yet, but it sounds like a wonderful destination! Would you like to book travel there?")
            else:
                print(f"{CHATBOT_NAME}: I don't recognize that city. Could you try again? (e.g., London, Paris, Tokyo)")
            continue

        # --- Intent classification --- #

        intent = get_intent(user_input, intent_vectorizer, intent_classifier)
        
        # --- Name change handling --- #

        if re.search(r'\bchange (my )?name\b', user_input_lower):
            old_name = user_state.get("name", "")
            new_name = input(f"{CHATBOT_NAME}: Sure {old_name or 'there'}! What would you like your new name to be?\nYou: ").strip()
            if new_name:
                user_state["name"] = new_name.capitalize()
                if old_name:
                    update_booking_name(old_name, user_state["name"])
                print(f"{CHATBOT_NAME}: Got it! I'll call you {user_state['name']} from now on.")
            else:
                print(f"{CHATBOT_NAME}: I didn't catch that. Your name remains {user_state.get('name', 'Guest')}.")
            continue


        # --- Universal Main Menu Return --- #
        
        in_flight_flow = user_state["flight_context"].get("stage") is not None
        in_hotel_flow = user_state["hotel_context"].get("stage") is not None
        
        global_triggers = ["main menu", "go back", "menu", "back", "home", "cancel"]

        if user_input_lower.strip() in global_triggers and not (in_flight_flow or in_hotel_flow):
            user_state.update({
                "awaiting_profile_choice": True,
                "awaiting_post_flight_choice": False,
                "awaiting_post_view_choice": False,
            })
            user_state["flight_context"]["stage"] = None
            user_state["hotel_context"]["stage"] = None
            conversation_context["topic"] = "menu"
            print(f"{CHATBOT_NAME}: Back at the main menu, {user_name}! What would you like to do next?")
            continue

        # --- Main Menu State (awaiting_profile_choice) --- #

        if user_state.get("awaiting_profile_choice"):

            # --- View bookings --- #

            if intent == "view_bookings":
                user_state.update({"awaiting_profile_choice": False, "awaiting_post_view_choice": True})
                conversation_context["topic"] = "bookings"
                print(f"{CHATBOT_NAME}: {handle_view_bookings()}")
                last_intent = intent
                continue

            # --- Book flight --- #

            elif intent == "book_flight":
                if has_active_booking(user_name, "flight"):
                    print(f"{CHATBOT_NAME}: {user_name}, you already have an active flight booking.")
                    continue
                if check_and_resume_booking("flight"):
                    continue
                user_state["awaiting_profile_choice"] = False
                conversation_context["topic"] = "flight_booking"
                print(f"{CHATBOT_NAME}: {handle_flight_booking(user_input)}")
                last_intent = intent
                continue

            # --- Book hotel --- #

            elif intent == "book_hotel":
                if has_active_booking(user_name, "hotel"):
                    print(f"{CHATBOT_NAME}: {user_name}, you already have an active hotel booking.")
                    continue
                if check_and_resume_booking("hotel"):
                    continue
                user_state["awaiting_profile_choice"] = False
                conversation_context["topic"] = "hotel_booking"
                hc = user_state["hotel_context"]
                hc["stage"] = hc.get("stage") or "location"
                print(f"{CHATBOT_NAME}: {handle_hotel_booking_flow(user_input)}")
                last_intent = intent
                continue

            # --- Exit Handling --- #

            elif intent == "exit":
                print(f"{CHATBOT_NAME}: Thanks for using {CHATBOT_NAME}, {user_name}! Safe travels!")
                break

            # --- Greeting --- #

            elif intent == "greeting":
                if last_intent == "small_talk":
                    print(f"{CHATBOT_NAME}: Still here, {user_name}! How can I help you today?")
                else:
                    print(f"{CHATBOT_NAME}: {handle_greeting()}")
                conversation_context["mood"] = "friendly"
                last_intent = intent
                continue

            # ---- Small talk --- #

            elif intent == "small_talk":
                print(f"{CHATBOT_NAME}: {handle_small_talk()}")
                conversation_context["mood"] = "positive"
                last_intent = intent
                continue

            # --- Recommendations --- #

            elif intent == "recommendations":
                print(f"{CHATBOT_NAME}: {handle_recommendations(user_input)}")
                conversation_context["topic"] = "recommendations"
                last_intent = intent
                continue

            # --- Discoverability (help/what can you do) --- #

            elif intent == "discoverability":
                if last_intent in ["greeting", "small_talk"]:
                    print(f"{CHATBOT_NAME}: Still curious? I can help you find hotels, book flights, or view your bookings.")
                else:
                    print(f"{CHATBOT_NAME}: I can help you book flights, find hotels, or review your bookings. What would you like to do?")
                conversation_context["topic"] = "menu"
                last_intent = intent
                continue

            # --- Fallback to Q&A system --- #
            qa_response = handle_qa(
                user_input, qa_df, qa_vectorizer, qa_counts,
                qa_tfidf_vectorizer, qa_tfidf_counts
            )
            if qa_response:
                print(f"{CHATBOT_NAME}: {qa_response}")
            else:
                print(f"{CHATBOT_NAME}: {user_name}, what would you like to do next? I can book flights, hotels, or show your bookings.")
            last_intent = "qa"
            continue

        # --- Post Flow Handling --- #

        # --- After viewing bookings --- #

        if user_state.get("awaiting_post_view_choice"):
            print(f"{CHATBOT_NAME}: {continue_post_view_choice(user_input)}")
            last_intent = "post_view"
            continue

        # --- After completing flight booking --- #

        if user_state.get("awaiting_post_flight_choice"):
            print(f"{CHATBOT_NAME}: {continue_post_flight_choice(user_input)}")
            last_intent = "post_flight"
            continue

        # --- Active Flight and Hotel Flows --- #

        fc = user_state["flight_context"]
        hc = user_state["hotel_context"]

        # --- Continue flight booking if in progress --- #

        if fc["stage"]:
            response = continue_booking_flow(user_input)
            if response is not None:
                print(f"{CHATBOT_NAME}: {response}")
            conversation_context["topic"] = "flight_booking"
            last_intent = "book_flight"
            continue

        # --- Continue hotel booking if in progress --- #

        if hc["stage"]:
            print(f"{CHATBOT_NAME}: {handle_hotel_booking_flow(user_input)}")
            conversation_context["topic"] = "hotel_booking"
            last_intent = "book_hotel"
            continue

        # --- General Intents --- #

        if intent == "greeting":
            print(f"{CHATBOT_NAME}: {handle_greeting()}")
            
        elif intent == "small_talk":
            print(f"{CHATBOT_NAME}: {handle_small_talk()}")
            conversation_context["mood"] = "positive"
            
        elif intent == "recommendations":
            print(f"{CHATBOT_NAME}: {handle_recommendations(user_input)}")
            conversation_context["topic"] = "recommendations"
            
        elif intent == "discoverability":
            print(f"{CHATBOT_NAME}: I can help you book flights, find hotels, or view your bookings. What sounds good?")
            
        elif intent == "book_flight":
            print(f"{CHATBOT_NAME}: {handle_flight_booking(user_input)}")
            
        elif intent == "book_hotel":
            print(f"{CHATBOT_NAME}: {handle_hotel_booking_flow(user_input)}")
            
        elif intent == "view_bookings":
            user_state["awaiting_post_view_choice"] = True
            print(f"{CHATBOT_NAME}: {handle_view_bookings()}")

        # --- Identity handling (name collection) --- #

        elif intent == "identity" and not (fc["stage"] or hc["stage"] or user_state.get("awaiting_profile_choice")):
            resp = handle_identity(user_input)
            print(f"{CHATBOT_NAME}: {resp}")
            
            if user_state.get("name"):
                load_progress_from_file(user_state["name"])
                if check_and_resume_booking("flight") or check_and_resume_booking("hotel"):
                    continue 
                if user_state.get("existing_user"):
                    user_state["awaiting_profile_choice"] = True

        # --- Fallback to Q&A or error message --- #
            
        else:
            qa_response = handle_qa(
                user_input, qa_df, qa_vectorizer, qa_counts,
                qa_tfidf_vectorizer, qa_tfidf_counts
            )
            if qa_response:
                print(f"{CHATBOT_NAME}: {qa_response}")
            else:
                print(f"{CHATBOT_NAME}: Sorry {user_name}, I didn't understand that. You can tell me to book a flight, book a hotel, or see your bookings.")

        last_intent = intent

if __name__ == "__main__":
    chatbot()