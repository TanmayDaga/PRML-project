import yfinance as yf
import pandas as pd
import datetime
import time
import praw

def fetch_reddit_data(reddit_instance, subreddit_str, query, start_date_str, end_date_str, max_results=1000):
    """ Fetches Reddit submissions using PRAW for a combined query from potentially multiple subreddits specified in subreddit_str (e.g., 'stocks+investing'). """
    posts_data = []
    subreddit = reddit_instance.subreddit(subreddit_str)
    try:
        start_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(days=1)
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
    except ValueError:
        print(f"Error: Invalid date format ('{start_date_str}' or '{end_date_str}').")
        return pd.DataFrame(posts_data, columns=['Timestamp', 'Text'])

    print(f"\nFetching up to {max_results} posts mentioning query terms for sector from subreddits: r/{subreddit_str}")
    print(f"Target range: {start_date_str} to {end_date_str} (UTC)")
    count = 0
    fetched_ids = set()
    max_attempts = max(max_results * 5, 1500)
    attempts = 0
    try:
        for submission in subreddit.search(query, sort='new', limit=None):
            attempts += 1
            time.sleep(0.2)
            if submission.id in fetched_ids: continue
            fetched_ids.add(submission.id)
            submission_timestamp = int(submission.created_utc)
            submission_dt = datetime.datetime.fromtimestamp(submission_timestamp, tz=datetime.timezone.utc)

            if submission_timestamp < start_timestamp:
                print(f"\n-> Encountered post before start date {start_date_str}. Stopping search.")
                break
            if start_timestamp <= submission_timestamp < end_timestamp:
                text_content = submission.title + " " + submission.selftext
                posts_data.append([submission_dt, text_content.strip()])
                count += 1
                print(f"   Fetched post #{count} ({submission_dt.strftime('%Y-%m-%d %H:%M')})", end='\r')
                if count >= max_results:
                    print(f"\n-> Reached max_results limit ({max_results}).")
                    break
            if attempts >= max_attempts:
                 print(f"\n-> Warning: Checked {attempts} posts. Stopping search.")
                 break
        print(" " * 80, end='\r')
        print(f"Finished fetching for r/{subreddit_str}. Found {count} posts potentially within the date range.")
    except praw.exceptions.PRAWException as e:
        if 'subreddits have been banned' in str(e) or 'cannot be accessed' in str(e):
             print(f"\nPRAW Error fetching Reddit data for r/{subreddit_str}: One or more subreddits might be private, banned, or misspelled. Details: {e}")
        else:
             print(f"\nPRAW Error fetching Reddit data for r/{subreddit_str}: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred fetching Reddit data for r/{subreddit_str}: {e}")
    return pd.DataFrame(posts_data, columns=['Timestamp', 'Text'])

def initialize_reddit():
    """Initialize and return a Reddit API instance"""
    from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    
    print("Initializing Reddit connection...")
    reddit_instance = None
    try:
        reddit_instance = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_async=False
        )
        print(f"Reddit instance created. Read-only: {reddit_instance.read_only}")
        try: # Verify connection
            for _ in reddit_instance.subreddit("popular").hot(limit=1): pass
            print("Reddit connection seems OK.")
        except Exception as conn_test_e:
            print(f"Warning: Could not fully verify Reddit connection: {conn_test_e}")
    except Exception as e:
        print(f"FATAL Error initializing Reddit: {e}.")
        return None
    return reddit_instance

def fetch_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading {ticker} stock data...")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, interval="1d")
        if stock_data.empty:
            print(f"Error: Stock data download failed for {ticker}.")
            return None
            
        stock_data['Daily_Label'] = (stock_data['Close'] > stock_data['Open']).astype(int)
        daily_labels_map = stock_data['Daily_Label']
        daily_labels_map.index = daily_labels_map.index.date
        print(f"Stock data ({ticker}) downloaded. Label distribution:\n{daily_labels_map.value_counts()}")
        return stock_data, daily_labels_map
    except Exception as e:
        print(f"Error processing stock data for {ticker}: {e}.")
        return None, None