'''
Travel Booking Chatbot - Configuration Module
This module defines constants, global state, templates, and intents
'''

# --- Constants --- #
CHATBOT_NAME = "Hermes"
DATA_DIR = "data"
BOOKINGS_FILE = f"{DATA_DIR}/bookings.pkl"
FLIGHTS_FILE = f"{DATA_DIR}/flights.csv"
HOTELS_FILE = f"{DATA_DIR}/hotels.csv"

# --- Evaluator Toggle --- #
EVALUATION_MODE = False   # Set True when running performance tests

# --- Global State --- #
user_state = {
    "name": None,
    "flight_context": {
        "stage": None,
        "departure_city": None,
        "arrival_city": None,
        "date": None,
        "num_passengers": None,
        "return_flight": None, 
        "selected_flight": None,
        "available_flights": []
    },
    "hotel_context": {
        "stage": None,
        "location": None,
        "check_in": None,
        "check_out": None,
        "num_guests": None,
        "selected_hotel": None,
        "available_hotels": []
    },
    "booked_flight": None,
    "booked_hotel": None,
    "awaiting_profile_choice": False,
    "awaiting_post_flight_choice": False,
    "awaiting_post_view_choice": False,
    "awaiting_city_for_recommendations": False
}

# --- Templates --- =
TEMPLATES = {
    # --- Intro & identity --- #
    "intro": (
        "{time_greeting}! I'm {bot_name}, your personal travel assistant! \n"
        "I can help you find and book flights, hotels, answer travel questions, "
        "and remember your details for a smoother experience. I also give recommendations of things to do in select cities!\n"
        "To get started, what should I call you?"
    ),
    "greeting_known": "{time_greeting}, {user_name}! Ready to discover your next adventure?",
    "name_confirmation": "Fantastic! Let's get started. How can I assist you today?",
    "unknown_name": "I don't know your name yet. What should I call you?",

    # --- Small Talk Responses --- #
    "small_talk_responses": [
        "I'm doing great, {user_name}! How's your day going?",
        "All systems go here, {user_name}! What about you?",
        "Feeling fantastic and ready to help you travel, {user_name}!",
        "Hey {user_name}, I'm all ears. Any exciting plans coming up?",
        "I'm energized and ready to explore new adventures with you, {user_name}!",
        "Doing well! Just waiting to help you book something amazing, {user_name}.",
        "All good here, {user_name}. Dreaming of faraway places, how about you?"
    ],

    # --- Flight templates --- #
    "departure_request": "Great! Which city will you be flying from?",
    "arrival_request": "Got it! And which city are you flying to?",
    "date_request": "When would you like to travel? Please use DD-MM-YYYY format.",
    "invalid_date": "Oops, that date has already passed. Could you pick a future date instead?",
    "passenger_request": "How many passengers will be flying? (Up to 9 per booking)",
    "flight_options": "Here are the flights I found from {departure_city} to {arrival_city} on {date}:",
    "flight_selected": "You picked flight {flight_number} with {airline} — {class}, £{price}. Shall we confirm this booking? (yes/no)",
    "booking_confirmed": "All set! {num_passengers} passenger(s) are booked to {arrival_city}. Bon voyage!",
    "booking_reselect": "No worries! Please select another flight option from the list.",
    "no_flights": "Sorry, I couldn't find any flights that match your request. Please restart your flight search or go back to the main menu.",

    # --- Hotel templates --- #
    "hotel_location_request": "Which city would you like to stay in?",
    "hotel_checkin_request": "When will you be checking in? Please use DD-MM-YYYY format.",
    "hotel_checkout_request": "And when will you be checking out? Please use DD-MM-YYYY format.",
    "hotel_invalid_checkout": "Hmm, the check-out date must be after the check-in date. Could you try again?",
    "hotel_guests_request": "How many guests will be staying? (Maximum 10 guests)",
    "hotel_options": "Here are some great hotels I found in {location}:",
    "hotel_selected": "You selected {name} — {rating}★, £{price_per_night} per night. Confirm this booking? (yes/no)",
    "hotel_booking_confirmed": "All done! {num_guests} guest(s) are booked from {check_in} to {check_out}. Enjoy your stay!",
    "hotel_booking_reselect": "No problem! Please choose another hotel from the options above.",
    "no_hotels": "Sorry, I couldn't find any hotels in {location}.",

    # --- Booking view --- #
    "view_bookings": "Here are your current bookings:",
    "no_bookings": "It seems you don't have any bookings yet. Let's start planning your next trip!",
    "city_recommendations": {
        "london": {
            "Attractions": [
                "Visit the Tower of London and see the Crown Jewels",
                "Explore the British Museum (free entry!)",
                "Take a ride on the London Eye",
                "Watch the Changing of the Guard at Buckingham Palace"
            ],
            "Food & Drink": [
                "Try traditional fish and chips",
                "Visit Borough Market for diverse food options",
                "Enjoy afternoon tea at a classic tea room"
            ],
            "Culture": [
                "Catch a show in the West End theatre district",
                "Visit the Tate Modern for contemporary art",
                "Explore the historic neighborhoods of Notting Hill and Camden"
            ]
        },
        "larnaca": {
            "Attractions": [
                "Relax on Finikoudes Beach and stroll the palm-lined promenade",
                "Visit the Church of Saint Lazarus, one of Cyprus' most historic landmarks",
                "Explore the Larnaca Salt Lake and spot flamingos in winter",
                "Walk around the Larnaca Marina and enjoy the coastal views"
            ],
            "Food & Drink": [
                "Try traditional Cypriot souvlaki at a local taverna",
                "Enjoy fresh seafood by the waterfront",
                "Sample halloumi dishes and local meze platters",
                "Try Cypriot coffee in a café along the Finikoudes strip"
            ],
            "Culture": [
                "Visit the Pierides Museum to learn about local history",
                "Explore the Medieval Castle of Larnaca",
                "Wander through the old Turkish Quarter (Skala)",
                "Experience a traditional Cypriot evening with music and dance"
            ]
        },
        "paris": {
            "Attractions": [
                "Visit the Eiffel Tower at sunset",
                "Explore the Louvre Museum and see the Mona Lisa",
                "Walk along the Champs-Élysées",
                "Visit Notre-Dame Cathedral and Sacré-Cœur"
            ],
            "Food & Drink": [
                "Try authentic French pastries at a local boulangerie",
                "Enjoy wine and cheese at a traditional bistro",
                "Visit a café in the Latin Quarter"
            ],
            "Culture": [
                "Stroll through the artistic Montmartre district",
                "Take a Seine River cruise",
                "Explore the Palace of Versailles"
            ]
        },
        "new york": {
            "Attractions": [
                "Visit the Statue of Liberty and Ellis Island",
                "Explore Central Park",
                "See the views from the Empire State Building or Top of the Rock",
                "Walk across the Brooklyn Bridge"
            ],
            "Food & Drink": [
                "Try a classic New York pizza slice",
                "Visit Chelsea Market for diverse food options",
                "Get bagels from a traditional NYC deli"
            ],
            "Culture": [
                "See a Broadway show in Times Square",
                "Visit the Metropolitan Museum of Art",
                "Explore the diverse neighborhoods like SoHo and Greenwich Village"
            ]
        },
        "tokyo": {
            "Attractions": [
                "Visit the historic Senso-ji Temple in Asakusa",
                "Experience the bustling Shibuya Crossing",
                "Explore the Meiji Shrine and nearby Harajuku",
                "Visit Tokyo Skytree for panoramic views"
            ],
            "Food & Drink": [
                "Try authentic sushi at Tsukiji Outer Market",
                "Experience a traditional ramen shop",
                "Visit an izakaya for Japanese pub food"
            ],
            "Culture": [
                "Explore the traditional Asakusa district",
                "Visit teamLab Borderless digital art museum",
                "Experience a traditional tea ceremony"
            ]
        },
        "rome": {
            "Attractions": [
                "Visit the Colosseum and Roman Forum",
                "Explore Vatican City and the Sistine Chapel",
                "Throw a coin in the Trevi Fountain",
                "Climb the Spanish Steps"
            ],
            "Food & Drink": [
                "Try authentic carbonara and cacio e pepe",
                "Enjoy gelato from a traditional gelateria",
                "Visit local trattorias in Trastevere"
            ],
            "Culture": [
                "Explore the Pantheon",
                "Wander through ancient Roman streets",
                "Visit the beautiful Borghese Gardens"
            ]
        }
    }
}

# --- Intents --- #
intents = {
    "greeting": [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon", 
        "hi there", "hello bot", "greetings", "hey there", "hiya", "sup", 
        "yo", "good day", "morning"
    ],
    "identity": [
        "what is my name", "do you know my name", "my name is", "who am i",
        "say my name", "remember my name", "tell me my name", "who is this user", 
        "do I have a name", "what do you call me"
    ],
    "small_talk": [
        "how are you", "how's it going", "what's up", "how you doing", "how are things",
        "are you ok", "how do you feel", "nice to meet you", "what's new",
        "who are you", "what are you", "are you a robot", "talk to me"
    ],
    "discoverability": [
        "what can you do", "help", "features", "tell me what you can do", 
        "can you help me", "what do you do", "list your functions", "how do i use this",
        "what are my options", "show me commands", "guide me", "assistance please",
        "what features do you have", "I need help", "list capabilities", "what services offer"
    ],
    "return_main_menu": [
        "go back", "return to main menu", "main menu", "menu", "home", "back", 
        "start over", "restart", "main screen", "cancel and go back", "exit to menu",
        "top menu", "I want to go back"
    ],
    "book_flight": [
        "i want to book a flight", "i would like to book a flight", "book a flight", 
        "book from", "fly from", "book to", "fly to", "schedule a flight", 
        "buy plane tickets", "looking for a flight", "reserve a seat on a plane",
        "flight booking", "I need to fly", "get me a plane ticket", "search for flights",
        "find me a flight", "purchase ticket", "book new travel", "fly now"
    ],
    "book_hotel": [
        "i want to book a hotel", "i would like to book a hotel", "book a hotel", 
        "book accommodation", "find a hotel", "reserve a room", "looking for a place to stay",
        "hotel booking", "find me a room", "stay at a hotel", "accommodation needed",
        "room reservation", "I need a place to sleep", "search for hotels",
        "make reservation", "hotel room needed"
    ],
    "view_bookings": [
        "show my bookings", "view bookings", "my bookings", "what have i booked",
        "list my trips", "check my reservations", "show me my flights", "show me my hotels",
        "booking status", "display my itinerary", "itinerary", "check my plans"
    ],
    "rebook_flight": [
        "cancel and rebook my flight", "I want to rebook my flight", "change my flight", 
        "book a new flight instead", "reschedule my flight", "modify flight", 
        "change flight dates", "flight reschedule", "move my flight", "I want to change my flight",
        "flight adjustment", "change date of flight"
    ],
    "rebook_hotel": [
        "cancel and rebook my hotel", "I want to rebook my hotel", "change my hotel", 
        "cancel hotel", "cancel my hotel booking", "reschedule hotel", "change room booking",
        "move my hotel dates", "modify hotel reservation", "change my room",
        "modify hotel", "switch hotel", "hotel change dates"
    ],
    "recommendations": [
        "what to do in", "things to do in", "recommend things in", 
        "what can i do in", "activities in", "places to visit in",
        "what should i see in", "tourist attractions in", "sightseeing in",
        "recommend activities", "things to see in", "what to see in",
        "best things to do", "must see in", "top attractions in",
        "what is there to do", "guide for", "travel guide"
    ]
}