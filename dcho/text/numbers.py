"""Expansion of numeric and symbolic entities into spoken Persian words.

This is the "بسط موجودیت‌های عددی و نمادین" stage of the normalizer. It runs
on plain text and produces plain text: `expand_numbers("۱۴:۳۰")` yields
"ساعت چهارده و سی دقیقه", never phonemes. Grapheme-to-phoneme conversion
happens later in the chain and is not this module's concern.

The module is deliberately dependency-free so that it can be imported by the
data pipeline, by the inference server and by tests without pulling in the
model stack.

Three entry points matter to callers:

    normalize_digits(text)  -- Persian/Arabic-Indic digits to ASCII
    number_to_words(n)      -- a single Python number to words
    expand_numbers(text)    -- every numeric entity in a piece of text

Everything else is internal, apart from the lookup tables (`UNITS`,
`JALALI_MONTHS`, ...) which the surrounding normalizer reuses.
"""

from __future__ import annotations

import re

__all__ = [
    "UNITS",
    "JALALI_MONTHS",
    "GREGORIAN_MONTHS",
    "HALF",
    "FRACTION_ALIASES",
    "ZWNJ",
    "normalize_digits",
    "number_to_words",
    "fraction_to_words",
    "expand_numbers",
]

ZWNJ = "‌"

# --------------------------------------------------------------------------
# Digit normalisation
# --------------------------------------------------------------------------

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"          # U+06F0 .. U+06F9
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"     # U+0660 .. U+0669
ASCII_DIGITS = "0123456789"

_DIGIT_TABLE = str.maketrans(
    PERSIAN_DIGITS + ARABIC_INDIC_DIGITS,
    ASCII_DIGITS + ASCII_DIGITS,
)

# U+066B ARABIC DECIMAL SEPARATOR and U+066C ARABIC THOUSANDS SEPARATOR.
_ARABIC_DECIMAL_RE = re.compile(r"(?<=\d)٫(?=\d)")
_GROUPED_RE = re.compile(r"(?<!\d)\d{1,3}(?:[,٬]\d{3})+(?!\d)")


def normalize_digits(text: str) -> str:
    """Map Persian (۰۱۲) and Arabic-Indic (٠١٢) digits to ASCII 0-9."""
    return text.translate(_DIGIT_TABLE)


def _normalize_marks(text: str) -> str:
    """Fold Arabic decimal/thousands separators and drop digit grouping."""
    text = _ARABIC_DECIMAL_RE.sub(".", text)
    return _GROUPED_RE.sub(
        lambda m: m.group(0).replace(",", "").replace("٬", ""), text
    )


# --------------------------------------------------------------------------
# Cardinal numbers
# --------------------------------------------------------------------------

ZERO = "صفر"
NEGATIVE = "منفی"
DECIMAL_POINT = "ممیز"

_ONES = ("", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه")
_TEENS = (
    "ده", "یازده", "دوازده", "سیزده", "چهارده",
    "پانزده", "شانزده", "هفده", "هجده", "نوزده",
)
_TENS = ("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
_HUNDREDS = (
    "", "صد", "دویست", "سیصد", "چهارصد",
    "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد",
)

# Scale word for group i, i.e. for 1000**i. Persian follows the long scale,
# where میلیارد is 10^9 and بیلیون is 10^12 (not the American 10^9).
_SCALES = ("", "هزار", "میلیون", "میلیارد", "بیلیون", "بیلیارد", "تریلیون")

_AND = "و"
_CONJ = " " + _AND + " "

_DIGIT_WORDS = (ZERO,) + _ONES[1:]


def _read_digits(digits: str) -> str:
    """Read a digit string one digit at a time: 0912 -> صفر نه یک دو."""
    return " ".join(_DIGIT_WORDS[int(c)] for c in digits if c.isdigit())


def _three_digits(n: int) -> str:
    """Words for 1..999."""
    parts = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        parts.append(_TEENS[rest - 10])
    else:
        tens, ones = divmod(rest, 10)
        if tens:
            parts.append(_TENS[tens])
        if ones:
            parts.append(_ONES[ones])
    return _CONJ.join(parts)


def _int_to_words(n: int) -> str:
    if n < 0:
        return NEGATIVE + " " + _int_to_words(-n)
    if n == 0:
        return ZERO

    groups = []
    remaining = n
    while remaining:
        remaining, rest = divmod(remaining, 1000)
        groups.append(rest)

    if len(groups) > len(_SCALES):
        # Past تریلیون there is no settled vocabulary; digit-by-digit is at
        # least unambiguous.
        return _read_digits(str(n))

    parts = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        if index == 0:
            parts.append(_three_digits(group))
        elif index == 1 and group == 1:
            # "هزار", never "یک هزار"; the higher scales do keep their یک.
            parts.append(_SCALES[1])
        else:
            parts.append(_three_digits(group) + " " + _SCALES[index])
    return _CONJ.join(parts)


# --------------------------------------------------------------------------
# Ordinals
# --------------------------------------------------------------------------

# Ordinals suffix ـم to the cardinal. Two irregulars: یک -> اول (only when the
# whole number is one; بیست و یکم keeps the regular form) and سه -> سوم.
# A final component ending in a vowel takes ـام behind a ZWNJ instead. Among
# the number words only سی qualifies: نُه and پنجاه end in /h/ and take the
# plain suffix (نهم، پنجاهم), and دو is fixed as دوم.
_ORDINAL_AM = {"سی"}


def _ordinalize(cardinal: str, n: int) -> str:
    if n == 1:
        return "اول"
    head, sep, last = cardinal.rpartition(_CONJ)
    if last == "سه":
        last = "سوم"
    elif last in _ORDINAL_AM:
        last = last + ZWNJ + "ام"
    else:
        last = last + "م"
    return head + sep + last


# --------------------------------------------------------------------------
# Decimals and fractions
# --------------------------------------------------------------------------

# Place word by number of digits after the separator.
_DECIMAL_PLACES = (
    "",
    "دهم",
    "صدم",
    "هزارم",
    "ده" + ZWNJ + "هزارم",
    "صدهزارم",
    "میلیونم",
    "ده" + ZWNJ + "میلیونم",
    "صدمیلیونم",
    "میلیاردم",
)

HALF = "نیم"

# Colloquial one-word readings. Not used unless a caller asks for them,
# because "یک دوم" is what a neutral reading of the written form 1/2 is.
FRACTION_ALIASES = {(1, 2): HALF}


def _decimal_to_words(int_part: str, frac_part: str, negative: bool = False) -> str:
    """Words for a decimal given as two literal digit strings."""
    head = _int_to_words(int(int_part) if int_part else 0)
    if negative:
        head = NEGATIVE + " " + head
    if not frac_part or not frac_part.strip("0"):
        return head
    if len(frac_part) < len(_DECIMAL_PLACES):
        tail = _int_to_words(int(frac_part)) + " " + _DECIMAL_PLACES[len(frac_part)]
    else:
        tail = _read_digits(frac_part)
    return head + " " + DECIMAL_POINT + " " + tail


def fraction_to_words(numerator: int, denominator: int, use_aliases: bool = False) -> str:
    """Words for a vulgar fraction: 3, 4 -> "سه چهارم"."""
    if use_aliases and (numerator, denominator) in FRACTION_ALIASES:
        return FRACTION_ALIASES[(numerator, denominator)]
    return (
        _int_to_words(numerator)
        + " "
        + _ordinalize(_int_to_words(denominator), denominator)
    )


# --------------------------------------------------------------------------
# Public number entry point
# --------------------------------------------------------------------------


def number_to_words(n: int | float, ordinal: bool = False) -> str:
    """Spell out a number in Persian words.

    >>> number_to_words(1375)
    'هزار و سیصد و هفتاد و پنج'
    >>> number_to_words(3, ordinal=True)
    'سوم'
    >>> number_to_words(3.14)
    'سه ممیز چهارده صدم'
    """
    if isinstance(n, float):
        if n.is_integer():
            n = int(n)
        else:
            text = repr(n)
            if "e" in text or "E" in text:
                text = f"{n:.20f}".rstrip("0")
            negative = text.startswith("-")
            int_part, _, frac_part = text.lstrip("-").partition(".")
            return _decimal_to_words(int_part, frac_part, negative)

    words = _int_to_words(n)
    return _ordinalize(words, n) if ordinal else words


# --------------------------------------------------------------------------
# Units, months, currencies
# --------------------------------------------------------------------------

UNITS = {
    # length
    "km": "کیلومتر",
    "m": "متر",
    "cm": "سانتی" + ZWNJ + "متر",
    "mm": "میلی" + ZWNJ + "متر",
    "nm": "نانومتر",
    # mass
    "kg": "کیلوگرم",
    "g": "گرم",
    "mg": "میلی" + ZWNJ + "گرم",
    "t": "تن",
    # volume
    "l": "لیتر",
    "L": "لیتر",
    "ml": "میلی" + ZWNJ + "لیتر",
    "cc": "سی" + ZWNJ + "سی",
    # data
    "B": "بایت",
    "KB": "کیلوبایت",
    "MB": "مگابایت",
    "GB": "گیگابایت",
    "TB": "ترابایت",
    "kb": "کیلوبیت",
    "Mb": "مگابیت",
    "Gb": "گیگابیت",
    # frequency, power, electricity
    "Hz": "هرتز",
    "kHz": "کیلوهرتز",
    "MHz": "مگاهرتز",
    "GHz": "گیگاهرتز",
    "W": "وات",
    "kW": "کیلووات",
    "V": "ولت",
    "A": "آمپر",
    "mAh": "میلی" + ZWNJ + "آمپر ساعت",
    # time
    "s": "ثانیه",
    "ms": "میلی" + ZWNJ + "ثانیه",
    "min": "دقیقه",
    "h": "ساعت",
    # compound and misc
    "km/h": "کیلومتر بر ساعت",
    "m/s": "متر بر ثانیه",
    "°C": "درجه سانتی" + ZWNJ + "گراد",
    "°F": "درجه فارنهایت",
    "%": "درصد",
}

JALALI_MONTHS = (
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
)

GREGORIAN_MONTHS = (
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)

CURRENCIES = {
    "تومان": "تومان",
    "ریال": "ریال",
    "دلار": "دلار",
    "یورو": "یورو",
    "پوند": "پوند",
    "درهم": "درهم",
    "ت": "تومان",
}

PERCENT = "درصد"
RANGE_WORD = "تا"
HOUR_WORD = "ساعت"
MINUTE_WORD = "دقیقه"
SECOND_WORD = "ثانیه"

# Jalali years in running text sit between roughly 1300 and 1500; anything
# from 1700 up is a Gregorian year whatever separator was used.
_GREGORIAN_YEAR_FLOOR = 1700


# --------------------------------------------------------------------------
# Text-level expansion
# --------------------------------------------------------------------------


def _number_token(token: str) -> str:
    """Words for a literal integer token, honouring a leading zero.

    A leading zero means the digits are an identifier -- an area code, an
    extension -- rather than a quantity, so they are read one by one.
    """
    negative = token.startswith("-")
    digits = token.lstrip("-")
    if len(digits) > 1 and digits[0] == "0":
        words = _read_digits(digits)
    else:
        words = _int_to_words(int(digits))
    return NEGATIVE + " " + words if negative else words


def _numeric_token(token: str) -> str:
    """Words for a literal integer or decimal token."""
    if "." in token:
        negative = token.startswith("-")
        int_part, _, frac_part = token.lstrip("-").partition(".")
        return _decimal_to_words(int_part, frac_part, negative)
    return _number_token(token)


# A clock time is announced with "ساعت" unless the sentence already supplies
# it, either literally or as the second half of a range ("از ۹:۰۰ تا ۱۱:۰۰").
_TIME_PREFIX_SUPPRESSORS = (HOUR_WORD, RANGE_WORD)


def _preceded_by(match: re.Match, words: tuple[str, ...]) -> bool:
    head = match.string[: match.start()].rstrip()
    return any(head.endswith(word) for word in words)


def _handle_date(match: re.Match, jalali: bool) -> str | None:
    year, month, day = (int(g) for g in match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    if jalali and year >= _GREGORIAN_YEAR_FLOOR:
        jalali = False
    months = JALALI_MONTHS if jalali else GREGORIAN_MONTHS
    return "{} {} {}".format(
        number_to_words(day, ordinal=True), months[month - 1], _int_to_words(year)
    )


def _handle_jalali(match: re.Match) -> str | None:
    return _handle_date(match, jalali=True)


def _handle_gregorian(match: re.Match) -> str | None:
    return _handle_date(match, jalali=False)


def _handle_time(match: re.Match) -> str | None:
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3)) if match.group(3) else 0
    if hour > 23 or minute > 59 or second > 59:
        return None
    parts = [] if _preceded_by(match, _TIME_PREFIX_SUPPRESSORS) else [HOUR_WORD]
    parts.append(_int_to_words(hour))
    if minute:
        parts.append(_AND + " " + _int_to_words(minute) + " " + MINUTE_WORD)
    if second:
        parts.append(_AND + " " + _int_to_words(second) + " " + SECOND_WORD)
    return " ".join(parts)


def _handle_percent(match: re.Match) -> str:
    return _numeric_token(match.group(1)) + " " + PERCENT


def _handle_ordinal_suffix(match: re.Match) -> str:
    n = int(match.group(1))
    return _ordinalize(_int_to_words(n), n)


def _handle_currency(match: re.Match) -> str:
    return _numeric_token(match.group(1)) + " " + CURRENCIES[match.group(2)]


def _handle_unit(match: re.Match) -> str:
    return _numeric_token(match.group(1)) + " " + UNITS[match.group(2)]


def _handle_digit_run(match: re.Match) -> str:
    return _read_digits(match.group(1))


def _handle_grouped_run(match: re.Match) -> str | None:
    """Read a separated identifier such as 021-88776655 digit by digit.

    Only accepted when the whole thing is long enough to be an identifier
    and looks like one: a leading zero, or a group too long to be a plain
    number. Otherwise 1000-2000 would stop being a range.
    """
    groups = re.split(r"[-– ]", match.group(1))
    digits = "".join(groups)
    if len(digits) < 8:
        return None
    if not groups[0].startswith("0") and not any(len(g) >= 6 for g in groups):
        return None
    return _read_digits(digits)


def _handle_fraction(match: re.Match) -> str | None:
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return fraction_to_words(int(match.group(1)), denominator)


def _handle_range(match: re.Match) -> str:
    return "{} {} {}".format(
        _numeric_token(match.group(1)), RANGE_WORD, _numeric_token(match.group(2))
    )


def _handle_number(match: re.Match) -> str:
    return _numeric_token(match.group(0))


_UNIT_ALTERNATION = "|".join(
    re.escape(u) for u in sorted(UNITS, key=len, reverse=True)
)

_NUM = r"\d+(?:\.\d+)?"

# Rules are tried in order at each position and the first one that both
# matches and accepts (returns a string rather than None) wins. Ordering
# encodes the disambiguation policy:
#
#   * A slash- or dash-separated triple is a date only when its FIRST
#     component has four digits, i.e. is a year. Slashes select the Jalali
#     calendar, dashes the Gregorian one, matching how the two are written
#     in Persian text.
#   * A single slash between short components -- 3/4 -- is therefore never
#     reachable by the date rules and is read as a fraction.
#   * Percent, currency and unit rules run before the bare-number rule so
#     that the trailing symbol is consumed together with its number.
#   * A run of eight or more digits is an identifier (phone number, national
#     id) and is read digit by digit; reading it as a cardinal is always
#     wrong.
_RULES: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?![\d\-])"), _handle_gregorian),
    (re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])"), _handle_jalali),
    (re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])"), _handle_time),
    (re.compile(r"[٪%]\s?(" + _NUM + r")"), _handle_percent),
    (re.compile(r"(" + _NUM + r")\s?[٪%]"), _handle_percent),
    (re.compile(r"(\d+)-?(?:ام|م)(?!\w)"), _handle_ordinal_suffix),
    (
        re.compile(
            r"(" + _NUM + r")\s*(" + "|".join(sorted(CURRENCIES, key=len, reverse=True)) + r")(?!\w)"
        ),
        _handle_currency,
    ),
    (
        re.compile(r"(" + _NUM + r")\s*(" + _UNIT_ALTERNATION + r")(?![A-Za-z0-9])"),
        _handle_unit,
    ),
    (re.compile(r"(\d{8,})(?!\d)"), _handle_digit_run),
    (re.compile(r"(\d+(?:[-– ]\d+)+)(?!\d)"), _handle_grouped_run),
    (re.compile(r"(\d{1,3})/(\d{1,3})(?![\d/])"), _handle_fraction),
    (re.compile(r"(" + _NUM + r")[-–—](" + _NUM + r")(?![\d.])"), _handle_range),
    (re.compile(r"-?" + _NUM), _handle_number),
]

# Only positions starting with one of these can begin a match, which keeps
# the scan cheap on ordinary prose.
_TRIGGERS = frozenset("0123456789%٪-–—")


def expand_numbers(text: str) -> str:
    """Rewrite every numeric or symbolic entity in `text` as Persian words.

    >>> expand_numbers("قیمت ۵۰۰۰ت است")
    'قیمت پنج هزار تومان است'
    """
    text = _normalize_marks(normalize_digits(text))

    out: list[str] = []
    i = 0
    end = len(text)
    while i < end:
        char = text[i]
        if char in _TRIGGERS:
            for pattern, handler in _RULES:
                match = pattern.match(text, i)
                if match is None:
                    continue
                replacement = handler(match)
                if replacement is None:
                    continue
                out.append(replacement)
                i = match.end()
                break
            else:
                out.append(char)
                i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)
