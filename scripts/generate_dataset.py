"""
generate_dataset.py
--------------------
Builds data/raw/dataset.csv: a small (~700 row) customer-support-ticket
text classification dataset with 5 categories.

WHY A GENERATED DATASET INSTEAD OF A DOWNLOADED ONE:
This project was built in an offline sandbox with no internet access, so a
real dataset could not be downloaded at build time. Instead we assemble a
template-based dataset that mimics real customer support tickets (the same
domain as the brief's own examples, e.g. "I was charged an unexpected fee
on my credit card!"). Sentences are built by combining many independent
opening/detail/closing fragments and vocabulary substitutions, so the
resulting texts are realistic in style and vocabulary even though they are
not scraped from a live source.

If you have internet access and want a fully "real" dataset instead, good
drop-in replacements with the same shape (short text -> category) are:
  * Twitter US Airline Sentiment / Complaints (Kaggle)
  * Consumer Complaint Database (consumerfinance.gov, data.gov)
  * Ecommerce Customer Support Tickets (Kaggle / Hugging Face datasets)
Just replace data/raw/dataset.csv with a two-column `text,category` CSV
and re-run `python scripts/train.py`.

Run:
    python scripts/generate_dataset.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"

# ---------------------------------------------------------------------------
# Building blocks per category. Sentences are assembled from an opener, an
# optional detail clause, and an optional closer, then lightly varied.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Billing": {
        "openers": [
            "I was charged an unexpected fee on my credit card",
            "There is a duplicate charge on my last invoice",
            "My subscription renewed but I already cancelled it",
            "I was billed twice for the same order this month",
            "The amount charged does not match the price I saw at checkout",
            "I never received a refund for the item I returned",
            "My card was charged in a different currency than expected",
            "I'm being charged for a plan I never signed up for",
            "The invoice shows a tax amount that seems incorrect",
            "I want to dispute a charge from last week",
        ],
        "details": [
            "and I would like this fixed as soon as possible",
            "can you please check my billing history",
            "this has happened twice in the last three months",
            "I have already contacted my bank about it",
            "please confirm when the refund will be processed",
            "my account balance does not reflect this correctly",
            "I have attached a screenshot of the transaction",
            "",
        ],
        "closers": [
            "Please let me know how to proceed.",
            "Thank you for looking into this.",
            "I would appreciate a quick response.",
            "Waiting for your reply.",
            "",
        ],
    },
    "Delivery": {
        "openers": [
            "I ordered something but it has not arrived yet",
            "My package shows as delivered but I never received it",
            "The tracking number for my order is not working",
            "My order arrived several days later than promised",
            "The box arrived damaged and the item inside was broken",
            "I received the wrong item in my package",
            "My shipment has been stuck at the same location for a week",
            "The courier left my parcel outside without notifying me",
            "I never got a shipping confirmation for my recent order",
            "Only part of my order arrived, the rest is missing",
        ],
        "details": [
            "and the estimated delivery date has already passed",
            "can you check the shipping status for me",
            "I need this item urgently for an event this weekend",
            "the tracking page has not updated in days",
            "I already checked with my neighbors and the front desk",
            "this is the second time this has happened to me",
            "",
        ],
        "closers": [
            "Please advise on the next steps.",
            "I would like a replacement or a refund.",
            "Let me know what you can do.",
            "Thanks for your help with this.",
            "",
        ],
    },
    "Technical Support": {
        "openers": [
            "The app keeps crashing every time I try to log in",
            "I cannot reset my password using the link you sent",
            "The website shows an error message when I click checkout",
            "The mobile app is stuck on a loading screen",
            "I keep getting logged out randomly while using the service",
            "The search feature on your site is not returning any results",
            "I am unable to upload files to my account",
            "The page freezes whenever I try to update my settings",
            "Notifications from the app stopped working after the last update",
            "I get a 'server error' message every time I open my dashboard",
        ],
        "details": [
            "I have already tried restarting my device",
            "this happens on both my phone and my laptop",
            "I cleared the cache but the problem is still there",
            "the error code shown is not explained anywhere",
            "I updated to the latest version and it still fails",
            "this started happening after the recent update",
            "",
        ],
        "closers": [
            "Could you help me fix this?",
            "Please let me know a possible workaround.",
            "This is affecting my daily use of the product.",
            "Looking forward to a solution.",
            "",
        ],
    },
    "Account": {
        "openers": [
            "I cannot access my account even though my password is correct",
            "My account was suspended and I don't understand why",
            "I want to change the email address linked to my account",
            "Someone else may have logged into my account without permission",
            "I am not receiving the verification code to log in",
            "I would like to permanently delete my account and my data",
            "My account shows information that I never entered",
            "I need to merge two accounts that were created by mistake",
            "I lost access to the email used to register my account",
            "My profile picture and details reset on their own",
        ],
        "details": [
            "I have tried the password reset process multiple times",
            "this is urgent since I use this account for work",
            "I noticed some unfamiliar activity in my order history",
            "please verify my identity so I can regain access",
            "I have followed all the steps in the help center",
            "",
        ],
        "closers": [
            "Please help me restore access to my account.",
            "I would like this resolved quickly.",
            "Let me know what information you need from me.",
            "Thank you for your assistance.",
            "",
        ],
    },
    "Product Feedback": {
        "openers": [
            "The product I received does not match the description online",
            "I really love how easy this product is to use",
            "The build quality feels cheaper than I expected for the price",
            "This is exactly the product I was looking for, great job",
            "The instructions included with the product are confusing",
            "I think the new design is a big improvement over the last version",
            "The color of the item is different from what was shown in photos",
            "This product solved a problem I've had for years, thank you",
            "The size runs smaller than the size chart suggested",
            "I have a suggestion for a feature that would make this even better",
        ],
        "details": [
            "and I wanted to share this feedback with your team",
            "I have used similar products before for comparison",
            "several of my friends have had the same experience",
            "I hope this feedback is useful for future versions",
            "I would still recommend it despite this issue",
            "",
        ],
        "closers": [
            "Thanks for reading my feedback.",
            "Hope this helps improve the product.",
            "Just wanted this on record.",
            "Let me know if you'd like more details.",
            "",
        ],
    },
}

NOISE_PREFIXES = [
    "", "", "", "",
    "<p>", "Hello team, ", "Hi, ", "To whom it may concern, ",
]
NOISE_SUFFIXES = [
    "", "", "", "",
    "</p>", " Visit https://example.com/support for more info.",
    " More at http://status.example.com",
]

N_PER_CATEGORY = 140  # 5 categories * 140 = 700 rows


def build_text(category: str) -> str:
    parts = CATEGORIES[category]
    opener = random.choice(parts["openers"])

    # ~15% of the time, borrow the "detail" clause from a different
    # category instead of this one. Real support tickets often blend
    # topics (e.g. a delivery complaint that also mentions a refund), so
    # this keeps the dataset from being trivially separable by
    # vocabulary alone and gives the classifiers (and the confusion
    # matrix) something real to work with.
    if random.random() < 0.15:
        other_category = random.choice([c for c in CATEGORIES if c != category])
        detail = random.choice(CATEGORIES[other_category]["details"])
    else:
        detail = random.choice(parts["details"])

    closer = random.choice(parts["closers"])

    sentence = opener
    if detail:
        sentence += ", " + detail
    sentence = sentence.strip()
    if not sentence.endswith((".", "!", "?")):
        sentence += random.choice([".", "!"])
    if closer:
        sentence += " " + closer

    prefix = random.choice(NOISE_PREFIXES)
    suffix = random.choice(NOISE_SUFFIXES)
    return f"{prefix}{sentence}{suffix}".strip()


def main():
    rows = []
    for category in CATEGORIES:
        seen = set()
        while len(seen) < N_PER_CATEGORY:
            text = build_text(category)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((text, category))

    random.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    counts = {}
    for _, c in rows:
        counts[c] = counts.get(c, 0) + 1
    for c, n in counts.items():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
