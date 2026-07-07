'''
Travel Booking Chatbot - Utility Functions Module
This module provides NLP processing, data handling, and helper functions
for the travel booking chatbot system.
'''

import pickle
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from datetime import datetime

from config import BOOKINGS_FILE, FLIGHTS_FILE, HOTELS_FILE, intents, EVALUATION_MODE

# --- NLTK setup --- #

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

# --- Lemmatise text for NLP processing --- #

def lemmatise_text(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

# --- File Operations --- #

# Load flight data from CSV file
def load_flights():
    df = pd.read_csv(FLIGHTS_FILE)
    return df.to_dict(orient="records")

# Load hotel data from CSV file and parse amenities field
def load_hotels():
    df = pd.read_csv(HOTELS_FILE)

    if "amenities" in df.columns:
        df["amenities"] = df["amenities"].apply(
            lambda x: [a.strip() for a in x.split(",")]
        )

    return df.to_dict(orient="records")

# Append a new booking to the pickled bookings file creating file if it doesn't exist.
def save_booking_pickle(booking):
    try:
        bookings = []
        if os.path.exists(BOOKINGS_FILE):
            with open(BOOKINGS_FILE, "rb") as f:
                bookings = pickle.load(f)
        bookings.append(booking)
        with open(BOOKINGS_FILE, "wb") as f:
            pickle.dump(bookings, f)
    except Exception as e:
        print("Error saving booking:", e)

# Load all bookings from pickled file
def load_bookings():
    try:
        with open(BOOKINGS_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

# Update all bookings for a user when they change their name. Also normalises legacy booking formats to dict structure.
def update_booking_name(old_name, new_name):
    try:
        bookings = load_bookings()
        updated = False
        normalised_bookings = []

        for b in bookings:
            if isinstance(b, list) and len(b) >= 2:
                b = {"user": b[0], "flight": b[1]}
            elif not isinstance(b, dict):
                continue
            if b.get("user") == old_name:
                b["user"] = new_name
                updated = True
            normalised_bookings.append(b)

        if updated or normalised_bookings != bookings:
            with open(BOOKINGS_FILE, "wb") as f:
                pickle.dump(normalised_bookings, f)

    except Exception as e:
        print("Error updating bookings:", e)

# --- City & Location Extraction --- #

# Extract all unique cities from flight data (both departure and arrival)
def get_available_cities():
    flights = load_flights()
    cities = set()
    for flight in flights:
        cities.add(flight["departure_city"].lower())
        cities.add(flight["arrival_city"].lower())
    return cities

# Extract all unique locations from hotel data
def get_available_hotel_locations():
    hotels = load_hotels()
    locations = set()
    for hotel in hotels:
        locations.add(hotel["location"].lower())
    return locations

# Extract a valid flight city from user input by matching against available cities. Handles both single-word and two-word city names
def extract_city_from_input(user_input):
    available_cities = get_available_cities()
    words = user_input.lower().split()

    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in available_cities:
            return clean_word.capitalize()

    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        clean_two_word = re.sub(r'[^\w\s]', '', two_word)
        if clean_two_word in available_cities:
            return clean_two_word.title()

    return None

# Extracts a valid hotel location from user input by matching against available locations. Similar to extract_city_from_input but for hotel locations
def extract_location_from_input(user_input):
    available_locations = get_available_hotel_locations()
    words = user_input.lower().split()

    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in available_locations:
            return clean_word.capitalize()

    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        clean_two_word = re.sub(r'[^\w\s]', '', two_word)
        if clean_two_word in available_locations:
            return clean_two_word.title()

    return None

# --- Flight Details Extraction --- #

# Extract all flight booking details from a single user message. Uses regex patterns to identify cities, dates, passenger counts, and flight type
def extract_flight_details(user_input):
    details = {
        "departure_city": None, 
        "arrival_city": None, 
        "date": None, 
        "num_passengers": None, 
        "return_flight": None
    }

    user_lower = user_input.lower()
    if "one way" in user_lower or "one-way" in user_lower or "single" in user_lower:
        details["return_flight"] = False
    elif "return" in user_lower or "round trip" in user_lower or "round-trip" in user_lower or "two way" in user_lower or "two-way" in user_lower:
        details["return_flight"] = True

    from_to_match = re.search(r'from\s+([\w\s]+?)\s+to\s+([\w\s]+?)(?=\s+on|\s+for|\s+with|\s*$)', user_input, re.IGNORECASE)
    
    if from_to_match:
        departure_candidate = from_to_match.group(1).strip()
        arrival_candidate = from_to_match.group(2).strip()
        
        dep_city = extract_city_from_input(departure_candidate)
        arr_city = extract_city_from_input(arrival_candidate)
        
        if dep_city:
            details["departure_city"] = dep_city
        if arr_city:
            details["arrival_city"] = arr_city
    else:
        from_match = re.search(r'from\s+([\w\s]+?)(?=\s+(?:to|on|for|$))', user_input, re.IGNORECASE)
        to_match = re.search(r'to\s+([\w\s]+?)(?=\s+(?:from|on|for|$))', user_input, re.IGNORECASE)

        if from_match:
            city = extract_city_from_input(from_match.group(1).strip())
            if city: 
                details["departure_city"] = city
        
        if to_match:
            city = extract_city_from_input(to_match.group(1).strip())
            if city: 
                details["arrival_city"] = city

    date_match = re.search(r'\b(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b', user_input)
    if date_match: 
        details["date"] = date_match.group(1)

    passenger_patterns = [
        r'for\s+(\d+)\s+(?:people|passengers?|persons?)',
        r'(\d+)\s+(?:people|passengers?|persons?)',
        r'(\d+)\s+(?:pax|travellers?)',
    ]
    
    for pattern in passenger_patterns:
        passenger_match = re.search(pattern, user_input, re.IGNORECASE)
        if passenger_match:
            num = int(passenger_match.group(1))
            if 1 <= num <= 9:
                details["num_passengers"] = num
                break

    return details

# Extract all hotel booking details from a single user message
def extract_hotel_details(user_input):
    details = {
        "location": None,
        "check_in": None,
        "check_out": None,
        "num_guests": None
    }
    
    location_patterns = [
        r'(?:in|at)\s+([\w\s]+?)(?=\s+(?:from|for|on|$))',
        r'hotel\s+(?:in|at)\s+([\w\s]+?)(?=\s+|$)',
    ]
    
    for pattern in location_patterns:
        loc_match = re.search(pattern, user_input, re.IGNORECASE)
        if loc_match:
            loc = extract_location_from_input(loc_match.group(1).strip())
            if loc:
                details["location"] = loc
                break
    
    if not details["location"]:
        details["location"] = extract_location_from_input(user_input)
    
    date_matches = re.findall(r'\b(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b', user_input)
    if date_matches:
        if len(date_matches) >= 1:
            check_in_dt = parse_valid_date(date_matches[0])
            if check_in_dt:
                details["check_in"] = check_in_dt.strftime("%Y-%m-%d")
        
        if len(date_matches) >= 2:
            check_out_dt = parse_valid_date(date_matches[1])
            if check_out_dt:
                details["check_out"] = check_out_dt.strftime("%Y-%m-%d")
    
    guest_patterns = [
        r'(?:for\s+)?(\d+)\s+(?:guests?|people|persons?)',
        r'(\d+)\s+(?:pax|travellers?)',
    ]
    
    for pattern in guest_patterns:
        guest_match = re.search(pattern, user_input, re.IGNORECASE)
        if guest_match:
            num = int(guest_match.group(1))
            if 1 <= num <= 10:
                details["num_guests"] = num
                break
    
    return details

# Master extraction function that determines booking type and extracts all details. This is the main function to call when user provides comprehensive input
def extract_all_details_from_input(user_input):
    details = {
        "departure_city": None,
        "arrival_city": None,
        "date": None,
        "num_passengers": None,
        "return_flight": False,
        "booking_type": None 
    }
    
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["flight", "fly", "flying"]):
        details["booking_type"] = "flight"
    elif any(word in user_lower for word in ["hotel", "accommodation", "stay"]):
        details["booking_type"] = "hotel"
    
    flight_details = extract_flight_details(user_input)
    details.update(flight_details)
    
    return details

# --- Name Extraction --- #

# Extract user's name from input like "my name is John" or single-word responses
def extract_name(user_input):
    words = [w for w in user_input.split() if w.isalpha()]
    if "my" in words and "name" in words and "is" in words:
        idx = words.index("is") + 1
        if idx < len(words):
            return words[idx].capitalize()
    if len(words) == 1:
        return words[0].capitalize()
    return None

# --- Intent Recognition with Classifier --- #

# Trains a Logistic Regression classifier for intent recognition. Uses TF-IDF vectorization and balanced class weights.
# Optionally displays performance metrics if EVALUATION_MODE is enabled.

def initialize_intent_classifier():
    X_train = []
    y_train = []

    for intent, phrases in intents.items():
        for phrase in phrases:
            X_train.append(lemmatise_text(phrase))
            y_train.append(intent)

    # --- Vectoriser --- #

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_features=500,
        min_df=1
    )
    X_vectorized = vectorizer.fit_transform(X_train)

    # --- Classifier --- #

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    classifier.fit(X_vectorized, y_train)

    # --- Evaluation Mode --- #

    if EVALUATION_MODE:
        print("\n=== INTENT CLASSIFIER PERFORMANCE METRICS ===")

        # --- K-Fold Cross-Validation --- #

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(classifier, X_vectorized, y_train, cv=kf)
        print(f"K-Fold Accuracy Scores: {scores}")
        print(f"Mean Accuracy: {np.mean(scores):.3f}")

        # --- Confusion Matrix --- #

        y_pred = classifier.predict(X_vectorized)
        labels = sorted(list(set(y_train)))
        cm = confusion_matrix(y_train, y_pred, labels=labels)

        print("\nConfusion Matrix (Labels in order):")
        print(labels)
        print(cm)

        # --- Classification Report --- #
        
        print("\nClassification Report:")
        print(classification_report(y_train, y_pred))

        print("=== END METRICS ===\n")

    return vectorizer, classifier

# Predicts the intent using the trained Logistic Regression classifier and a 
# confidence-based decision process, relying solely on the model's output.
def get_intent(user_input, vectorizer, classifier, confidence_threshold=0.25):
    user_input_lem = lemmatise_text(user_input)
    
    # --- Classification Logic --- #

    X_vec = vectorizer.transform([user_input_lem])
    
    predicted_intent = classifier.predict(X_vec)[0]
    probabilities = classifier.predict_proba(X_vec)[0]
    max_prob = max(probabilities)
    
    NON_CRITICAL_THRESHOLD = 0.10
    
    if predicted_intent in ["identity", "small_talk"]:
        if max_prob >= NON_CRITICAL_THRESHOLD:
            return predicted_intent
            
    if max_prob >= confidence_threshold:
        return predicted_intent
    
    return None

# --- Q&A Handling --- #

# Initialise a Q&A system with both Count and TF-IDF vectorizers. Loads FAQ dataset and prepares similarity matching
def initialize_qa_system(data_dir):
    qa_df = pd.read_csv(f"{data_dir}/COMP3074-CW1-Dataset.csv")
    qa_df['Question_lem'] = qa_df['Question'].apply(lemmatise_text)

    qa_vectorizer = CountVectorizer(stop_words='english')
    qa_counts = qa_vectorizer.fit_transform(qa_df['Question_lem'].values)

    qa_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    qa_tfidf_counts = qa_tfidf_vectorizer.fit_transform(qa_df['Question_lem'].values)

    return qa_df, qa_vectorizer, qa_counts, qa_tfidf_vectorizer, qa_tfidf_counts

# Find best matching Q&A answer using cosine similarity and tires Count vectorization first, then TF-IDF if no good match
def handle_qa(user_input, qa_df, qa_vectorizer, qa_counts, qa_tfidf_vectorizer, qa_tfidf_counts):
    user_input_lem = lemmatise_text(user_input)
    vec = qa_vectorizer.transform([user_input_lem])
    sims = cosine_similarity(vec, qa_counts)
    idx = sims.argmax()
    if sims[0][idx] > 0.3:
        return qa_df.iloc[idx]['Answer']
    vec_tfidf = qa_tfidf_vectorizer.transform([user_input_lem])
    sims_tfidf = cosine_similarity(vec_tfidf, qa_tfidf_counts)
    idx_tfidf = sims_tfidf.argmax()
    if sims_tfidf[0][idx_tfidf] > 0.3:
        return qa_df.iloc[idx_tfidf]['Answer']
    return None

# --- Date Parsing and Validation --- #

# Parse and validate date string in multiple formats, ensuring date is in the future unless allow_past=True
def parse_valid_date(date_str, allow_past=False):
    formats = ["%Y-%m-%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            d = datetime.strptime(date_str, fmt)
            if not allow_past and d.date() <= datetime.now().date():
                return None
            return d
        except ValueError:
            continue
    return None

# Validate date and return in ISO format. Wrapper around parse_valid_date that returns string
def validate_date(date_str, allow_past=False):
    dt = parse_valid_date(date_str, allow_past=allow_past)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return None

# --- Booking Management helpers --- #

# Cancel a specific booking type for a user and removes matching booking from file and saves updated list
def cancel_user_booking(user_name, booking_type):
    bookings = load_bookings()
    updated = [b for b in bookings if not (b.get("user") == user_name and b.get("type") == booking_type)]
    if len(updated) != len(bookings):
        with open(BOOKINGS_FILE, "wb") as f:
            pickle.dump(updated, f)
        return True
    return False

# Checks if user has an active booking of a given type and prevents duplicate bookings
def has_active_booking(user_name, booking_type):
    return any(b.get("user") == user_name and b.get("type") == booking_type for b in load_bookings())

# Generic handler for selecting items from a list (flights or hotels). Supports numeric selection (1, 2, 3) or keyword matching
def handle_item_selection(user_input, context, available_items, item_type, templates, key_fields):
    if user_input.isdigit():
        index = int(user_input)
    else:
        digits = re.findall(r'\b(\d+)\b', user_input)
        index = int(digits[0]) if digits else None

    if index and 1 <= index <= len(available_items):
        item = available_items[index - 1]
    else:
        item = next((i for i in available_items if any(str(i[k]).lower() in user_input.lower() for k in key_fields)), None)

    if item:
        context[f"selected_{item_type}"] = item
        context["stage"] = "confirmation"
        return templates[f"{item_type}_selected"].format(**{k: item[k] for k in key_fields})

    return f"Please pick an option — for example, 'book {item_type} 1' or 'I want the second {item_type}'."

# Generates appropriate greeting based on current time of day.
def get_time_of_day_greeting():
    current_hour = datetime.now().hour 
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Hello"