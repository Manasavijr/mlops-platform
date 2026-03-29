import csv
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

POSITIVE = [
    "This product is absolutely fantastic and exceeded all my expectations.",
    "I love how well this works, highly recommend to everyone.",
    "Brilliant quality and superb customer service experience.",
    "Outstanding results, very impressed with the performance.",
    "Five stars, wonderful experience from start to finish.",
    "Amazing product, works exactly as advertised and more.",
    "Exceptional value for money, will definitely buy again.",
    "The best I have ever used, completely changed my life.",
]
NEGATIVE = [
    "This product is absolutely terrible and a total waste of money.",
    "I hate how poorly this works, would not recommend to anyone.",
    "Dreadful quality and horrible customer service experience.",
    "Disappointing results, very unimpressed with the performance.",
    "One star, awful experience from start to finish unfortunately.",
    "Terrible product, nothing like advertised, completely broken.",
    "Poor value for money, will never purchase from them again.",
    "The worst I have ever used, completely ruined my experience.",
]
AMBIGUOUS = [
    "It's okay I suppose, nothing special but not terrible either.",
    "Mixed feelings about this, some good and some bad aspects.",
    "Neither great nor awful, just an average product overall.",
    "Hard to say really, it does some things well, others not.",
    "Mediocre at best, somewhere between good and bad.",
    "Not what I expected, could be better or worse honestly.",
]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows -> {path}")


def gen(n, templates, label):
    return [{"text": random.choice(templates), "label": label, "source": "synthetic"} for _ in range(n)]


train_rows = gen(800, POSITIVE, 1) + gen(800, NEGATIVE, 0)
random.shuffle(train_rows)

val_rows = gen(100, POSITIVE, 1) + gen(100, NEGATIVE, 0)
random.shuffle(val_rows)

test_rows = gen(100, POSITIVE, 1) + gen(100, NEGATIVE, 0)
random.shuffle(test_rows)

drift_rows = [{"text": random.choice(AMBIGUOUS), "label": random.randint(0, 1), "source": "drift"} for _ in range(200)]

write_csv(RAW_DIR / "train.csv", train_rows)
write_csv(RAW_DIR / "val.csv", val_rows)
write_csv(RAW_DIR / "test.csv", test_rows)
write_csv(RAW_DIR / "drift_test.csv", drift_rows)
logger.info("All datasets generated.")
