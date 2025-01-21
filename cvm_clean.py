import os
import re
import json
import pandas as pd
import numpy as np

# Constants
TEXT_REMOVE = ("*Hello, users of CMV! This is a footnote from your moderators. We'd just like to remind you of a couple of things. "
               "Firstly, please remember to* ***[read through our rules](http://www.reddit.com/r/changemyview/wiki/rules)***. "
               "*If you see a comment that has broken one, it is more effective to report it than downvote it. "
               "Speaking of which,* ***[downvotes don't change views](http://www.reddit.com/r/changemyview/wiki/guidelines#wiki_upvoting.2Fdownvoting)****! "
               "If you are thinking about submitting a CMV yourself, please have a look through our* ***[popular topics wiki](http://www.reddit.com/r/changemyview/wiki/populartopics)*** *first. "
               "Any questions or concerns? Feel free to* ***[message us](http://www.reddit.com/message/compose?to=/r/changemyview)***. *Happy CMVing!*")

def load_data(file_path):
    """
    Load data from a JSON lines file.

    Args:
        file_path (str): The path to the JSON lines file.

    Returns:
        list: A list of dictionaries, each representing a JSON object.
    """
    try:
        with open(file_path, "r") as f:
            # Read each line in the file and parse it as JSON
            cmv_posts = [json.loads(line) for line in f]
        print("Finished loading data")
        return cmv_posts
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"File is not valid JSON: {file_path}")
        return []

def clean_text(text):
    """
    Clean the text by removing unwanted characters and URLs.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace('>', '').replace('\n', '')
    text = re.sub(r'(http\S+|www\S+)', '', text)
    return text

def clean_comments(data):
    """
    Clean and organize comments from the data.

    Args:
        data (list): A list of dictionaries containing comment data.

    Returns:
        list: A list of cleaned and organized comment dictionaries.
    """
    organized_comments = []
    for item in data:
        if "subreddit_id" in item:
            new_item = {
                'id': item['id'],
                'author': item['author'],
                'body': clean_text(item['body']),
                'children': item['replies']['data']['children'] if item['replies'] else []
            }
            organized_comments.append(new_item)
    return organized_comments

def clean_posts(data):
    """
    Clean and organize posts from the data.

    Args:
        data (list): A list of dictionaries containing post data.

    Returns:
        list: A list of cleaned and organized post dictionaries.
    """
    data_cleaned = []
    for post in data:
        # Add key/values for id, author of original post, title and body of original post, and comments on post
        cleaned_post = {
            'author': post['author'],
            'original_post': clean_text(post['title'] + " " + post['selftext'].replace(TEXT_REMOVE, '')),
            'comments': clean_comments(post['comments']),
            'delta_label': any(comment['author'] == 'DeltaBot' for comment in clean_comments(post['comments']))
        }
        data_cleaned.append(cleaned_post)
    return data_cleaned

def build_comment_tree(comments):
    """
    Build a tree structure from a list of comments.

    Args:
        comments (list): A list of comment dictionaries.

    Returns:
        list: A list of root comment dictionaries with nested children.
    """
    # Create a dictionary for quick lookup
    comment_dict = {comment['id']: {**comment, 'children': []} for comment in comments}

    # Build the tree structure
    for comment in comments:
        for child_id in comment.get('children', []):
            if child_id in comment_dict:
                comment_dict[comment['id']]['children'].append(comment_dict[child_id])
    
    # Extract the root comments (those not nested under any other)
    root_comments = [comment for comment in comment_dict.values() if comment['id'] not in [c for comment in comments for c in comment.get('children', [])]]
    return root_comments

def print_comment_tree(comment, level=0):
    """
    Recursively print a comment tree with indentation.

    Args:
        comment (dict): A dictionary representing a comment with potential nested children.
        level (int, optional): The current level of indentation. Defaults to 0.
    """
    # Print the current comment with indentation
    print("  " * level + f"Author: {comment['author']}, Comment: {comment['body']}")
    for child in comment['children']:
        print_comment_tree(child, level + 1)

def save_data(data, file_name):
    """
    Save data to a JSON file.

    Args:
        data (list): A list of dictionaries to save.
    """
    with open(file_name, 'w') as f:
        json.dump(data, f)
    

if __name__ == "__main__":
    # Load data from json file
    file_path = "data/cmv_20161111.jsonlist"
    cmv_posts = load_data(file_path)
    # Clean and organize posts
    cleaned_cmv_posts = clean_posts(cmv_posts)
    # Build comment trees for each post and remove flat comments list
    for post in cleaned_cmv_posts:
        post["comment_tree"] = build_comment_tree(post["comments"])
        post.pop("comments", None)

    # Save the cleaned data to a new file
    save_data(cleaned_cmv_posts, 'cmv_posts_cleaned.json')
    
