import re

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\S+')
HASHTAG_RE = re.compile(r'#')
NON_ALNUM_RE = re.compile(r'[^a-z0-9\s]')
MULTISPACE_RE = re.compile(r'\s+')

def clean_tweet_text(text: str) -> set:
    text = text.lower()
    text = URL_RE.sub('', text)
    text = MENTION_RE.sub('', text)
    text = HASHTAG_RE.sub('', text)
    text = NON_ALNUM_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()

    words = [w for w in text.split(' ') if w]
    return set(words)

def load_and_preprocess(filepath: str):
    tweets = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 3:
                    #fall back: treat the whole line as text if format differes
                    text = line
                else:
                    text = '|'.join(parts[2:])  # Join all parts after the first two as the tweet text
                word_set = clean_tweet_text(text)
                if word_set:
                    tweets.append(word_set)
    return tweets

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <filepath>")
        sys.exit(1)

    tweets = load_and_preprocess(sys.argv[1])
    print(f'Loaded and preprocessed {len(tweets)} tweets from {sys.argv[1]}')
    for t in tweets[:5]:
        print(t)