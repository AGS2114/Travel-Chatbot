# Hermes - Travel Booking Chatbot

Hermes is a command-line, NLP-based conversational assistant that helps users book flights and hotels, manage existing bookings, get city recommendations, and ask general travel-related questions. It was built for COMP3074 (Human-AI Interaction) at the University of Nottingham, combining a machine-learning intent classifier with a rule-based, finite-state dialogue manager.

## Features

- **Intent classification** - TF-IDF (character trigrams) + Logistic Regression classifies user input into 11 intents (`book_flight`, `book_hotel`, `view_bookings`, `recommendations`, `small_talk`, etc.), with confidence thresholds to avoid acting on unclear input.
- **Flight booking** - collects departure city, arrival city, one-way/return, date, and passenger count; validates cities/dates against `flights.csv`; presents matching options for selection and confirmation. Supports providing several details in a single message (e.g. _"book a flight from London to Larnaca on 30-12-2025 for 2 people"_).
- **Hotel booking** - collects location, check-in/check-out dates, and guest count; validates against `hotels.csv`; auto-fills the destination city from a just-booked flight.
- **Booking management** - view current bookings and cancel a flight or hotel booking, with explicit yes/no confirmation and contextual follow-up options (rebook, view bookings, main menu).
- **Identity & personalisation** - remembers the user's name across the session, greets returning users, and varies small-talk responses.
- **City recommendations** - curated Attractions / Food & Drink / Culture suggestions for six cities (London, Larnaca, Paris, New York, Tokyo, Rome).
- **Question answering (FAQ)** - TF-IDF cosine-similarity matching against a general-knowledge Q&A dataset, used as a fallback when no booking intent is detected.
- **Session resumption** - incomplete flight/hotel bookings are saved to `data/incomplete_bookings.json` and offered for resumption the next time the user is identified.

## Requirements

- Python 3.9+
- `pandas`, `scikit-learn`, `nltk` (only `WordNetLemmatizer`), plus the Python standard library

```bash
pip install pandas scikit-learn nltk
python -m nltk.downloader wordnet omw-1.4
```

## Running the chatbot

```bash
python chatbot.py
```

You'll be greeted and asked for your name, after which you can type things like:

- `book a flight from London to Larnaca on 30-12-2025 for 2 people`
- `find me a hotel in Larnaca`
- `show my bookings`
- `things to do in Rome`
- `what is my name`
- `main menu` - returns to the top-level menu from anywhere

Dates must be entered in `DD-MM-YYYY` (or `YYYY-MM-DD`) format.

## How it works (brief architecture overview)

1. **NLP layer** (`utils.py`) - normalises input via lemmatisation, classifies intent using a `TfidfVectorizer` + `LogisticRegression` pipeline trained on the phrases in `config.intents`, and extracts entities (cities, dates, passenger/guest counts) using regex and CSV lookups.
2. **Dialogue manager** (`chatbot.py`) - a stage-based finite state machine drives each booking type independently via `user_state["flight_context"]` / `user_state["hotel_context"]`, deciding the next prompt based on which slots are still empty.
3. **Data layer** - flight/hotel inventory is loaded from CSV into pandas DataFrames for validation and search; confirmed bookings are persisted with `pickle`; in-progress bookings are persisted as JSON so a session can be resumed later.

For the full design write-up, evaluation (85% intent-classification accuracy, CUQ usability score of 70±8.1), and discussion, see `Andreas_Sergiou_-_HAI_Report_-_20433881.pdf`.

## Known limitations

- Only rigid `DD-MM-YYYY` / `YYYY-MM-DD` date formats are accepted (no "tomorrow", "next Friday", etc.).
- Single-language (English), command-line only interface.
- The `identity` intent has lower precision (0.40) due to overlap with other intents such as `recommendations`.
