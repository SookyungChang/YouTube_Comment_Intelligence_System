# data/fetchData.py # load YouTube comments by API

from googleapiclient.discovery import build
import pandas as pd
from ycis.data.preprocess import filter_english_comments
from ycis.config import Config

config = Config()

# Load the YouTube API key from environment variables (keeps it secure) and build its client using the google-api-python-client library
youtube = build('youtube', 'v3', developerKey=config.YOUTUBE_API_KEY)

def get_comments(video_id: str, max_results: int, max_pages: int) -> pd.DataFrame:
    """Fetch comments from a YouTube video using the YouTube Data API v3."""

    comments = []  # Will hold all collected comment data
    next_page_token = None  # Used to navigate through paginated API results
    page_count = 0  # Tracks how many pages we've fetched so far

    while True:
        # Build the API request to fetch comment threads for the given video
        request = youtube.commentThreads().list(
            part="snippet",  # "snippet" includes the actual comment content
            videoId=video_id,  # The ID of the YouTube video to fetch comments from
            maxResults=max_results,  # How many comments to fetch per page (default 100)
            pageToken=next_page_token,  # None on first run; set to next page after that
            textFormat="plainText",  # Return plain text instead of HTML-formatted text
        )

        # Execute the request and get the response from YouTube's API
        response = request.execute()

        # Loop through each comment thread returned in this page
        for item in response["items"]:
            # Drill into the nested response structure to get the comment details
            comment = item["snippet"]["topLevelComment"]["snippet"]

            # Store only the fields we care about
            comments.append(
                {
                    "author": comment["authorDisplayName"],
                    "text": comment["textDisplay"],  # The comment text
                    "likeCount": comment["likeCount"],  # Number of likes on the comment
                    "publishedAt": comment["publishedAt"],  # When posted
                    "commentId": item["id"],  # Unique ID for the comment thread
                }
            )

        page_count += 1  # Increment page counter after processing each page

        # Get the token for the next page (will be None if this is the last page)
        next_page_token = response.get("nextPageToken")

        # Stop if there are no more pages, or we've hit our page limit
        if not next_page_token or page_count >= max_pages:
            break

    # Return all collected comments as a pandas DataFrame for easy analysis
    return pd.DataFrame(comments)


if __name__ == "__main__":
    video_id = "SbNDmAJBtyU" # example vdieo id
    comments_df = get_comments(
        video_id, 30, 1
    )
    english_comments_df = filter_english_comments(comments_df)
    english_comments_df.to_csv(config.DATA_DIR / 'comments_sample.csv', index=False)
