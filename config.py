import os
import warnings

# Ignore common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Reddit API Credentials ---
# !! IMPORTANT !! Fill these in with your credentials obtained from reddit.com/prefs/apps
REDDIT_CLIENT_ID = "ggESYt5GB2jwq7l-Bcbeqw"
REDDIT_CLIENT_SECRET = "w-jic40bz5zFPj1Ea9LrzG9ioI2pLA"
REDDIT_USER_AGENT = "macos:analyzer:v1 (by /u/Aggressive-Food-978)"

# --- Sector Configurations ---
SECTOR_CONFIG = {
    "Tech": {
        "companies": [{'ticker': 'AAPL', 'name': 'Apple'}, {'ticker': 'MSFT', 'name': 'Microsoft'}],
        "etf": "XLK",
        "subreddits": ["technology", "stocks", "investing"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-30",
    },
    "Agriculture": {
        "companies": [{'ticker': 'DE', 'name': 'Deere'}, {'ticker': 'ADM', 'name': 'Archer Daniels Midland'}],
        "etf": "MOO",
        "subreddits": ["stocks", "investing", "farming"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-30",
    },
    "Chemical": {
        "companies": [{'ticker': 'DOW', 'name': 'Dow'}, {'ticker': 'DD', 'name': 'DuPont'}],
        "etf": "XLB",
        "subreddits": ["stocks", "chemistry"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-30",
    },
    "Arms_Defense": {
        "companies": [{'ticker': 'LMT', 'name': 'Lockheed Martin'}, {'ticker': 'RTX', 'name': 'Raytheon'}],
        "etf": "ITA",
        "subreddits": ["stocks", "CredibleDefense", "LessCredibleDefence"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-30",
    },
    "Banking": {
        "companies": [{'ticker': 'JPM', 'name': 'JPMorgan Chase'}, {'ticker': 'BAC', 'name': 'Bank of America'}],
        "etf": "XLF",
        "subreddits": ["finance", "banking", "wallstreetbets"],
        "start_date": "2025-01-01",
        "end_date": "2025-03-30",
    }
}

# --- Constants ---
MAX_POSTS_PER_SECTOR = 1000
PLOT_DIR = "sector_plots"
MODEL_DIR = "saved_models"

# Create necessary directories
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)