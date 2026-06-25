import requests
import json
import csv
import time
import re
from collections import Counter
from datetime import datetime

# 1. Target Subreddits - Kept active professional legal communities
subreddits = [
    "legaltech", 
    "Lawyertalk", 
    "lawfirm", 
    "paralegal", 
    "artificial",
    "LegalTechMakers",
    "legaltechAI",
    "LegalAITech",
    "remotelegaljobs",
    "techlaw",
    "LegaltechEurope",
    "computerforensics",
    "LegalAIHelp",
    "LegalAIPrompts",
    "AIforLawyers_",
    "TheAttorneyLounge",
    "legal",
    "law"
]

# 2. RESTRUCTURED THESIS KEYWORDS - Divided into Operational Intent and Tech Anchors
INTENT_WORDS = [
    "build", "buy", "software", "clm", "vendor", "saas", "license", "licence", 
    "procurement", "automation", "tool", "tools", "platform", "stack", "vibe", "coding", 
    "code", "develop", "developing", "custom", "ROI"
]

TECH_ANCHORS = [
    "harvey", "thomson", "reuters", "cocounsel", "lexis+", "gc", "spellbook", 
    "luminance", "vlex", "vincent", "ironclad", "legalon", "brightflag", 
    "relativity", "air", "genie", "everlaw", "kira", "legora", "libra", "streamline", 
    "blue", "neota", "logic", "clio", "bloomberg", "gavel", "eigen", 
    "chatgpt", "claude", "copilot", "gemini", "wordsmith", "openai"
]

# 3. Expanded Stop Words Layer
STOP_WORDS = set([
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "as", "at", 
    "by", "for", "from", "in", "into", "of", "off", "on", "onto", "out", 
    "over", "to", "up", "with", "is", "was", "were", "be", "been", "being", 
    "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", 
    "it", "we", "they", "this", "that", "these", "those", "are", "not", 
    "about", "your", "my", "their", "our", "its", "can", "will", "just", 
    "than", "so", "all", "any", "more", "most", "so", "no", "yes", "me", "them",
    "what", "which", "who", "whom", "this", "that", "am", "are", "there", "their",
    "one", "like", "get", "would", "some", "when", "only", "also", "going", 
    "because", "should", "really", "where", "other", "every", "things", "think", 
    "very", "lot", "much", "still", "how", "when", "just", "dont", "people",
    "even", "use", "good", "make", "know", "time", "way", "see", "want", "well"
])

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("🚀 Launching targeted legal domain semantic extraction on Reddit...\n")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"thesis_keyword_analysis_{current_time}.csv"

with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Subreddit', 'Thread_URL', 'Frequent_Word', 'Occurrence_Count'])

    for sub in subreddits:
        print(f"🔍 Scanning r/{sub}...")
        
        # Macro cooldown pause between subreddits
        time.sleep(6.0)
        
        search_url = f"https://www.reddit.com/r/{sub}/search.json?q=build+OR+buy&restrict_sr=on&sort=relevance&t=all"
        
        try:
            try:
                response = requests.get(search_url, headers=headers)
            except requests.exceptions.ConnectionError:
                print("   🚨 Network drop detected. Retrying subreddit search in 10 seconds...")
                time.sleep(10.0)
                response = requests.get(search_url, headers=headers)
            
            if response.status_code == 429:
                print(f"   ⚠️ Reddit is rate-limiting us (429). Pausing for 60 seconds to cool down your IP...")
                time.sleep(60.0)
                response = requests.get(search_url, headers=headers)
                
            if response.status_code != 200:
                print(f"   ⚠️ Connection skipped for r/{sub} (Status Code: {response.status_code})")
                continue
                
            search_data = response.json()
            threads = search_data['data']['children']
            print(f"   -> Found {len(threads)} potential target threads. Commencing data mining...")
            
            for thread in threads:
                if 'data' not in thread or 'permalink' not in thread['data']:
                    continue
                    
                permalink = thread['data']['permalink']
                thread_url = f"https://www.reddit.com{permalink}"
                
                # Enhanced micro delay to perfectly mimic human browsing speeds
                time.sleep(3.5)
                
                try:
                    thread_response = requests.get(f"{thread_url}.json", headers=headers)
                except requests.exceptions.ConnectionError:
                    print("   🚨 Network drop detected on thread download. Retrying in 10 seconds...")
                    time.sleep(10.0)
                    thread_response = requests.get(f"{thread_url}.json", headers=headers)
                
                if thread_response.status_code == 429:
                    print("   ⚠️ Rate-limit hit on thread. Pausing for 70 seconds to fully reset firewall...")
                    time.sleep(70.0)
                    thread_response = requests.get(f"{thread_url}.json", headers=headers)
                    
                if thread_response.status_code != 200:
                    continue
                    
                thread_data = thread_response.json()
                
                if not isinstance(thread_data, list) or len(thread_data) < 2:
                    continue
                if not thread_data[0]['data']['children']:
                    continue
                
                # --- THESIS PRECISION RE-ENGINEERING ---
                # Isolate the Title only to evaluate if the core subject is valid
                post_title = thread_data[0]['data']['children'][0]['data'].get('title', '').lower()
                
                # Title must contain an operational intent word AND/OR a relevant tech anchor
                has_intent = any(w in post_title for w in INTENT_WORDS)
                has_tech = any(t in post_title for t in TECH_ANCHORS)
                
                # Strict Thesis Rule: The thread title MUST involve operational intent (build/buy/software/stack)
                # combined with either legal/tech phrasing to bypass random casual noise.
                if has_intent and (has_tech or "legal" in post_title or sub in ["legaltech", "legaltechAI", "LegalAITech", "LegalAIPrompts"]):
                    
                    # Gather the full thread text (Title + Body + Comments) for semantic counting
                    post_body = thread_data[0]['data']['children'][0]['data'].get('selftext', '')
                    full_thread_text = f"{post_title} {post_body}"
                    
                    comments = thread_data[1]['data']['children']
                    for comment in comments:
                        if 'body' in comment['data']:
                            full_thread_text += " " + comment['data']['body']
                    
                    lowercase_full_text = full_thread_text.lower()
                    
                    all_tokens = re.findall(r'\b[a-z]{2,}\b', lowercase_full_text)
                    cleaned_tokens = [w for w in all_tokens if w not in STOP_WORDS]
                    token_counts = Counter(cleaned_tokens)
                    
                    significant_patterns = {word: count for word, count in token_counts.items() if count > 5}
                    
                    if significant_patterns:
                        print(f"   ✅ Targeted thread parsed successfully: {thread_url}")
                        for word, occurrences in significant_patterns.items():
                            writer.writerow([sub, thread_url, word, occurrences])
                            
        except Exception as e:
            print(f"🚨 Operational error encountered processing community r/{sub}: {e}")
            
print(f"\n🎉 Extraction execution finalized. Dataset safely committed to: '{filename}'")