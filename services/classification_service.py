import re
import joblib
from pathlib import Path
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = (
    BASE_DIR
    / "utils"
    / "classification_model"
)

model = joblib.load(
    MODEL_DIR / "judicial_classifier.joblib"
)

word_vectorizer = joblib.load(
    MODEL_DIR / "word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    MODEL_DIR / "char_vectorizer.joblib"
)

mlb = joblib.load(
    MODEL_DIR / "label_binarizer.joblib"
)

ARABIC_DIACRITICS = re.compile(
    r"ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ"
)


def normalize_arabic(text: str) -> str:

    text = str(text)
    text = re.sub(
        r"[إأآا]",
        "ا",
        text
    )
    text = re.sub(
        r"ى",
        "ي",
        text
    )
    text = re.sub(
        r"ة",
        "ه",
        text
    )
    text = re.sub(
        r"ؤ",
        "و",
        text
    )
    text = re.sub(
        r"ئ",
        "ي",
        text
    )
    text = re.sub(
        ARABIC_DIACRITICS,
        "",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def predict_crimes(facts: str):

    text = normalize_arabic(facts)

    word_features = word_vectorizer.transform(
        [text]
    )

    char_features = char_vectorizer.transform(
        [text]
    )

    features = hstack([
        word_features,
        char_features
    ])

    prediction = model.predict(features)

    predicted_crimes = mlb.inverse_transform(
        prediction
    )[0]

    if predicted_crimes:

        return {
            "classified": True,
            "crimes": list(predicted_crimes)
        }

    return {
        "classified": False,
        "crimes": [],
        "message": (
            "The provided information is not sufficient to classify the case."
        )
    }