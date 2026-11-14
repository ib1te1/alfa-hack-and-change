import html
import re
import unicodedata
from html.parser import HTMLParser

_BLOCK_TAGS = frozenset(
    "address article blockquote br dd div dl dt figcaption h1 h2 h3 h4 h5 h6 hr li ol p pre "
    "section table td th tr ul".split()
)
# Zero-width characters and byte-order marks pasted from editors.
_INVISIBLE = dict.fromkeys(map(ord, "﻿​‌‍⁠­"))
_SKIPPED_TAGS = {"script", "style", "noscript", "svg"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(value: str | None) -> str:
    """Strip markup and collapse whitespace; paragraph breaks become single newlines."""
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    text = html.unescape("".join(parser.parts))
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def truncate(text: str, limit: int) -> str:
    """Cut on a word boundary so a summary never ends mid-word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:—-") + "…"


_LATIN_TO_CYRILLIC = dict(zip("aceopxykABCEHKMOPTX", "асеорхукАВСЕНКМОРТХ", strict=True))
_CYRILLIC_TO_LATIN = {cyrillic: latin for latin, cyrillic in _LATIN_TO_CYRILLIC.items()}


def _is_cyrillic(char: str) -> bool:
    return "CYRILLIC" in unicodedata.name(char, "")


def _fix_homoglyphs(word: str) -> str:
    letters = [char for char in word if char.isalpha()]
    cyrillic = sum(_is_cyrillic(char) for char in letters)
    latin = len(letters) - cyrillic
    if not cyrillic or not latin:
        return word
    # Editors mixed look-alike letters into titles ("Маркeтплейсы", "HoReСa");
    # the script used by most letters of the word wins.
    table = _LATIN_TO_CYRILLIC if cyrillic >= latin else _CYRILLIC_TO_LATIN
    return "".join(table.get(char, char) for char in word)


def normalize_label(value: str | None) -> str:
    """Clean a dictionary title: no emoji, no stray spaces, one script per word."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = "".join(
        char for char in text if unicodedata.category(char) not in {"So", "Sk", "Cf", "Cc", "Mn"}
    )
    return " ".join(_fix_homoglyphs(word) for word in text.split())


# Typed with the English keyboard layout: "2 vbyens" means "2 минуты".
_QWERTY_TO_RUSSIAN = str.maketrans("qwertyuiopasdfghjklzxcvbnm", "йцукенгшщзфывапролдячсмить")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(час|мин)")


def parse_duration(value: str | None) -> int | None:
    """Turn "2,5 часа", "1 час 25 минут" or "~2 часа" into whole minutes."""
    if not value:
        return None
    text = value.casefold().replace(",", ".")
    if not any(_is_cyrillic(char) for char in text):
        text = text.translate(_QWERTY_TO_RUSSIAN)
    minutes = 0.0
    for amount, unit in _DURATION_PART.findall(text):
        minutes += float(amount) * (60 if unit == "час" else 1)
    return round(minutes) if minutes >= 1 else None
