"""Rule-based ezafe prediction.

The ezafe is the unwritten /e/ that links a Persian noun to whatever
modifies it: کتابِ من, رنگِ آسمان, برادرِ بزرگِ من. Roughly one word in six
carries it in running prose, it is never spelled, and getting it wrong is
the single most audible failure mode a Persian TTS frontend has - it does
not merely mispronounce a word, it re-brackets the sentence.

Precision / recall tradeoff
---------------------------
This module is deliberately biased towards precision. A missing ezafe
sounds like a slightly clipped reading; a spurious one glues two unrelated
phrases together and makes the sentence ungrammatical to a listener. So
every rule here has to be one that a syntactician would defend, and when
two analyses are available the module picks the one that emits no linker.
The consequences are visible and intended:

* Noun + noun and noun + adjective adjacency are the productive rules, but
  they are switched off wherever the sequence is more likely to be
  subject + predicate than head + modifier (before a copula, after a bare
  demonstrative, across a light-verb compound).
* Words with an enclitic possessive (پدرم), numerals, adverbs, verbs and
  every closed-class item are refused outright.
* Nothing links across punctuation, and the last token of a clause never
  carries a linker because there is nothing left for it to link to.

The result recovers most genitive chains and almost never invents one. The
residue - predicative versus attributive readings that need real syntax,
and genitives that need world knowledge - is what the neural G2P stage in
`docs/DESIGN.fa.md` section 6 exists to absorb.

`predict_ezafe` accepts optional `pos_hints`, one tag per token, which
override the module's own guess. Tags are the strings this module uses
internally: noun, adj, verb, prep, pron, det, adv, conj, num, ord, name,
punct.
"""

from __future__ import annotations

from .lexicon import CURATED_WORDS, DEFAULT, VERB_FORMS, normalize_key

__all__ = [
    "EZAFE_NEVER",
    "EZAFE_ALWAYS",
    "predict_ezafe",
    "tag_token",
    "split_ezafe_marker",
    "BREAKS",
]

BREAKS = frozenset({"|", "||", "?", "!", "،", "؛", ".", "؟", ":", "…", "(", ")"})

ZWNJ = "‌"
_HAMZA_ABOVE = "ٔ"

_VOWEL_PHONEMES = "Aiueao"


def _words(text: str) -> frozenset:
    return frozenset(normalize_key(w) for w in text.split())


# --------------------------------------------------------------------------
# Closed classes
# --------------------------------------------------------------------------

PRONOUNS = _words(
    """
    من تو او وی ما شما آنها ایشان خود خویش خودم خودت خودش خودمان خودتان
    خودشان همدیگر یکدیگر همه هرکس هیچکس کسی چیزی
    """
)

DETERMINERS = _words(
    """
    این آن همین همان اینها آنان هر چند چندین هیچ بعضی برخی چنین چنان
    فلان همگی
    """
)

PREPOSITIONS = _words(
    """
    به از با در بر تا برای بدون مانند مثل درباره روی زیر بالای بالا پایین
    کنار میان بین پیش پس نزد سوی سمت طرف علیه ضمن طی جز بجز مقابل مقابلِ
    درون بیرون جلوی پشت عقب زیرِ برابر همراه بابت بهجای نسبتبه پای
    """
)

CONJUNCTIONS = _words(
    """
    و یا که اگر اما ولی چون زیرا بلکه یعنی هرچند اگرچه وگرنه چنانچه
    بنابراین همچنین اینکه ولیکن
    """
)

# نه is left out on purpose: it is also the numeral 9, and the numeral
# reading is the one that has to reach the ezafe rules (کلاسِ نهِ صبح).
PARTICLES = _words("را بله بلی آری خیر مگر البته فقط حتی نیز هم دیگر")

ADVERBS = _words(
    """
    امروز دیروز فردا امسال پارسال پریروز امشب دیشب همیشه هرگز هنوز اکنون
    الان حالا اینجا آنجا همینجا بسیار خیلی کمی زیاد زود دیر گاهی ناگهان دوباره
    بیشتر کمتر تقریبا حتما لطفا مثلا اصلا ظاهرا نسبتا دقیقا عملا ابدا
    حقیقتا واقعا معمولا احتمالا کاملا فورا مستقیما اتفاقا شخصا بالاخره
    چرا کجا چطور چگونه چقدر آیا کی سپس بعدا قبلا اکثرا شاید البته حتماً
    """
)

QUESTION_WORDS = _words("چه چی چرا کجا کی چطور چگونه کدام چقدر چند آیا چیست کیست")

CARDINALS = _words(
    """
    صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده
    پانزده شانزده هفده هجده نوزده بیست سی چهل پنجاه شصت هفتاد هشتاد نود
    صد دویست سیصد چهارصد پانصد ششصد هفتصد هشتصد نهصد هزار میلیون میلیارد
    بیلیون نیم ربع نصف
    """
)

ORDINALS = _words(
    """
    اول دوم سوم چهارم پنجم ششم هفتم هشتم نهم دهم یکم یازدهم دوازدهم
    بیستم سیام چهلم صدم هزارم آخر آخرین نخست نخستین
    """
)

# Measure and classifier words. They stand between a numeral and the thing
# counted and never head a genitive: پنج کیلو برنج, بیست درصد تخفیف.
MEASURE_WORDS = _words(
    """
    کیلو گرم تن متر کیلومتر سانتیمتر لیتر میلیمتر درصد نفر دانه
    دستگاه فقره باب رأس قطعه جفت تومان ریال دلار یورو درجه
    """
)

# Time-of-day nouns. A numeral may take the linker before one of these -
# ساعتِ هشتِ صبح - which is the one place a numeral heads a genitive.
TIME_OF_DAY = _words("صبح ظهر عصر شب بعدازظهر سحر شامگاه بامداد")

# Nouns whose reference is a point or span of time. They combine freely
# with each other (صبحِ روزِ جمعه) but a following ordinary noun almost
# always starts a new phrase (هر روز صبح چای می‌خورد).
TIME_NOUNS = TIME_OF_DAY | _words(
    """
    روز شب هفته ماه سال ساعت دقیقه ثانیه قرن دهه فصل بهار تابستان پاییز
    زمستان شنبه یکشنبه دوشنبه سهشنبه چهارشنبه پنجشنبه جمعه
    """
)

# Heads that take the linker before a bare numeral: ساعتِ هفت, عددِ هفت.
NUMBER_HEADS = _words("ساعت عدد شماره کلاس صفحه قرن سال ردیف بند ماده فصل سطر پلاک")

# Relational nouns. They are prepositional in distribution but nominal in
# form, and before a noun they take the linker essentially always.
RELATIONAL = _words(
    """
    روی زیر بالای کنار وسط میان بین پیش سر جلوی پشت عقب درون بیرون سوی
    سمت طرف برای مال سراسر نزدیک نزد دور اطراف پای دم رأس عمق قلب
    """
)

TITLES = _words("آقا آقای خانم دکتر مهندس استاد سرکار جناب حاج سید")

COPULAS = _words(
    """
    است هست نیست بود باشد بودند هستند هستیم هستم هستی هستید بودم بودی
    بودیم بودید نبود نباشد شد شده میشود
    """
)

# Stems of the verbs that carry Persian compound verbs. A noun sitting
# immediately before one of them is the nominal half of the predicate,
# not a genitive dependent: امتحان دادند, صحبت کنم, کار کرده است.
_LIGHT_STEMS = (
    "کرد", "کن", "شد", "شو", "داد", "ده", "زد", "زن", "گرفت", "گیر",
    "خورد", "خور", "کشید", "کش", "آورد", "آور", "یافت", "یاب",
    "نمود", "نما",
)

# The public contract of this module.
EZAFE_NEVER = frozenset(
    PRONOUNS
    | DETERMINERS
    | PREPOSITIONS
    | CONJUNCTIONS
    | PARTICLES
    | ADVERBS
    | QUESTION_WORDS
    | CARDINALS
    | MEASURE_WORDS
    | BREAKS
)

EZAFE_ALWAYS = frozenset(RELATIONAL | TITLES)

# Words that refuse to be the dependent of a linker: nothing may link to
# them from the left.
_NEVER_AFTER = frozenset(
    PREPOSITIONS
    | CONJUNCTIONS
    | PARTICLES
    | ADVERBS
    | QUESTION_WORDS
    | COPULAS
    | BREAKS
)

# A copula, a coordinator or the "full of" frame after the candidate
# dependent means the sequence is a predicate rather than a phrase.
_COORDINATORS = frozenset({normalize_key("و"), normalize_key("از")})

_POSSESSIVE_SUFFIXES = ("شان", "تان", "مان", "یم", "م", "ت", "ش")


# --------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------


def _is_verb(key: str) -> bool:
    """Inflected verb form, and not a noun that merely looks like one."""
    return key in VERB_FORMS and key not in CURATED_WORDS


def _has_enclitic_possessive(key: str) -> bool:
    """True for پدرم, دوستم, کتاب‌هایم - a filled possessive slot."""
    if key in DEFAULT or _is_verb(key):
        return False
    for suffix in _POSSESSIVE_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix) + 1:
            base = key[: -len(suffix)]
            if base in DEFAULT or (base.endswith("ی") and base[:-1] in DEFAULT):
                return True
    return False


def _indefinite(key: str) -> bool:
    """True for کتابی, خوبی - a known word plus the indefinite ـی."""
    if key in DEFAULT or _is_verb(key) or not key.endswith("ی"):
        return False
    return key[:-1] in DEFAULT


def _indefinite_noun(key: str) -> bool:
    """True for شعری - specifically a NOUN plus the indefinite ـی."""
    return _indefinite(key) and key[:-1] not in ADJECTIVES


ADJECTIVES = _words(
    """
    آبی آرام آسان آماده بد بزرگ بلند بهتر بدتر بیدار پاک تازه تمیز تنها
    جالب جوان چاق خالی خراب خشک خنک خوب خوشحال دراز درست دشوار راست روشن
    زیبا سبز سخت سرد سفید سنگین سیاه شاد شلوغ شور صحیح ضعیف عالی غمگین
    قرمز قوی کوتاه کوچک گرم گران گرسنه مشهور معروف مفید مناسب مهربان مهم
    نازک نرم نزدیک نو وسیع زرد سالم شیرین تلخ ترش خوشمزه پیر ارزان باهوش
    تیز جدی خسته دقیق زنده ساده شجاع عجیب غنی فقیر کافی کامل گذشته
    متأسف متشکر محکم مرتب مطمئن ممکن واقعی ایرانی فارسی تاریخی بارانی
    کویری غربی شرقی شمالی جنوبی قدیمی جدید ملی ریاضی پزشکی مهندسی زیادی
    پر خالی بینظیر بیپایان خوشبخت سالانه روزانه پیاده زودتر بلندتر
    """
)

NAMES = _words(
    """
    علی محمد حسن حسین رضا مهدی احمد امیر مریم فاطمه زهرا سارا نرگس لیلا
    زینب پریسا بهرام کوروش داریوش بابک سینا آرش نیما سهراب رستم فرهاد
    شیرین یاسمن الهام مینا حمید مجید سعید جواد ناصر کریم رحیم یوسف
    ابراهیم موسی عیسی عبدالله عبدالحسین محمدی حسینی احمدی حافظ سعدی
    فردوسی مولوی خیام نظامی رودکی ایران تهران اصفهان شیراز مشهد تبریز
    یزد کرمان اهواز رشت قم کرج همدان اراک زاهدان ساری گرگان ارومیه کیش
    البرز دماوند زاگرس خزر فارس آسیا اروپا آمریکا آلمان فرانسه انگلیس
    ژاپن چین هند ترکیه عراق افغانستان مصر شاهنامه
    """
)

GIVEN_NAMES = _words(
    """
    علی محمد حسن حسین رضا مهدی احمد امیر مریم فاطمه زهرا سارا نرگس لیلا
    زینب پریسا بهرام کوروش داریوش بابک سینا آرش نیما سهراب رستم فرهاد
    شیرین یاسمن الهام مینا حمید مجید سعید جواد ناصر کریم رحیم یوسف
    ابراهیم موسی عیسی
    """
)


def tag_token(token: str) -> str:
    """Best guess at the word class of `token`."""
    if token in BREAKS:
        return "punct"
    key = normalize_key(token)
    if not key:
        return "punct"
    if key in BREAKS:
        return "punct"
    if key in PRONOUNS:
        return "pron"
    if key in DETERMINERS:
        return "det"
    if key.endswith("ً"):
        # Arabic accusative adverbial: مثلاً, تقریباً, لطفاً.
        return "adv"
    if key in QUESTION_WORDS or key in ADVERBS:
        return "adv"
    if key in CONJUNCTIONS:
        return "conj"
    if key in PARTICLES:
        return "part"
    if key in ORDINALS:
        return "ord"
    if key in CARDINALS:
        return "num"
    if key in NAMES:
        return "name"
    if key in PREPOSITIONS or key in RELATIONAL:
        return "prep"
    if key in ADJECTIVES:
        return "adj"
    if _is_verb(key):
        return "verb"
    if key in DEFAULT:
        return "noun"
    # Morphology, for words the lexicon has never seen.
    for suffix in _POSSESSIVE_SUFFIXES:
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in PREPOSITIONS:
                return "prep"
    if key.endswith("ی") and key[:-1] in ADJECTIVES:
        return "adj"
    return "noun"


_NOMINAL = frozenset({"noun", "name", "pron", "det", "adj", "ord", "num"})


def split_ezafe_marker(token: str):
    """Split an explicitly written ezafe off a token.

    Persian writes the linker after a vowel-final word, either with the
    hamza of خانهٔ or with a ی: خانه‌ی, صدای, کتاب‌های. Returns
    `(bare_token, True)` when such a marker is present.
    """
    if not token:
        return token, False
    if _HAMZA_ABOVE in token:
        return token.replace(_HAMZA_ABOVE, ""), True
    if token.endswith(ZWNJ + "ی"):
        return token[:-2], True
    if token.endswith("ی") and len(token) > 2:
        key = normalize_key(token)
        if key in DEFAULT or _is_verb(key):
            return token, False
        base = token[:-1]
        if base.endswith(ZWNJ):
            base = base[:-1]
        if base.endswith(("ا", "و")) and normalize_key(base) in DEFAULT:
            return base, True
    return token, False


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def _is_light_verb(key: str) -> bool:
    if not _is_verb(key):
        return False
    return any(stem in key for stem in _LIGHT_STEMS)


def predict_ezafe(tokens: list[str], pos_hints: list[str] | None = None) -> list[bool]:
    """One boolean per token: does it carry the ezafe linker?

    >>> predict_ezafe(["کتاب", "من", "روی", "میز", "است"])
    [True, False, True, False, False]
    """
    n = len(tokens)
    flags = [False] * n
    if n == 0:
        return flags

    bare = []
    explicit = []
    for token in tokens:
        stripped, marked = split_ezafe_marker(token)
        bare.append(normalize_key(stripped))
        explicit.append(marked)

    if pos_hints is not None and len(pos_hints) == n:
        tags = list(pos_hints)
    else:
        tags = [tag_token(token) for token in bare]

    def tag_at(index: int) -> str:
        if 0 <= index < n:
            return tags[index]
        return "punct"

    def key_at(index: int) -> str:
        if 0 <= index < n:
            return bare[index]
        return ""

    for i in range(n):
        head, dependent = bare[i], key_at(i + 1)
        head_tag, dep_tag = tags[i], tag_at(i + 1)

        # Nothing to link to.
        if i == n - 1 or head_tag == "punct" or dep_tag == "punct":
            continue
        if head in BREAKS or dependent in BREAKS:
            continue

        # An explicitly written linker is not a prediction at all.
        if explicit[i]:
            flags[i] = True
            continue

        if dependent in _NEVER_AFTER or dep_tag in ("verb", "conj", "part", "adv"):
            continue

        always = head in EZAFE_ALWAYS

        # Numerals are refused everywhere except in front of a time of day
        # (ساعتِ هشتِ صبح); ordinals behave like adjectives and do head a
        # phrase, unless they open the sentence, where they are adverbial
        # ("اول کتاب را بخوان" = first, read the book).
        if not always and head_tag in ("num", "ord"):
            if head_tag == "ord":
                flags[i] = i > 0 and dep_tag in _NOMINAL
            else:
                flags[i] = dependent in TIME_OF_DAY
            continue

        if not always:
            if head in EZAFE_NEVER or head_tag in (
                "verb", "prep", "conj", "part", "adv", "pron", "det"
            ):
                continue
            if _has_enclitic_possessive(head):
                continue
            # A bare demonstrative or quantifier closes the phrase it opens:
            # اینِ کتاب is not a thing, so کتاب in این کتاب heads its own NP.
            if tag_at(i - 1) == "det":
                continue
            # The nominal half of a compound verb is not a dependent.
            if dep_tag == "noun" and _is_light_verb(key_at(i + 2)):
                continue
            # N1 closing a prepositional phrase, N2 the object of the verb
            # that follows: پیش از خواب کتاب می‌خواند. A copula is not a
            # verb for this purpose - it takes no object.
            if (
                dep_tag in ("noun", "name")
                and tag_at(i - 1) == "prep"
                and tag_at(i + 2) == "verb"
                and key_at(i + 2) not in COPULAS
            ):
                continue
            # An indefinite noun is not a genitive dependent.
            if dep_tag == "noun" and _indefinite_noun(dependent):
                continue

            # Predicate versus phrase. A copula just past the dependent -
            # or, for an adjective, a coordinator or the "full of" frame -
            # means the pair is a predicate, unless a subject or a
            # preposition sits in front of the head and accounts for the
            # clause structure some other way.
            after = key_at(i + 2)
            if after in COPULAS:
                # A subject in front of the head, a preposition governing
                # it, or an incoming linker on an adjective head all say
                # the head is inside a phrase rather than opening the
                # predicate. So does an indefinite adjective dependent,
                # which cannot be a predicate at all (فیلمِ خوبی بود).
                anchored = i > 0 and (
                    tag_at(i - 1) == "prep"
                    or (tag_at(i - 1) in ("name", "pron") and not flags[i - 1])
                    or (head_tag == "adj" and flags[i - 1])
                )
                if not anchored and not (dep_tag == "adj" and _indefinite(dependent)):
                    continue
            elif dep_tag == "adj" and after in _COORDINATORS:
                if not _indefinite(dependent):
                    continue

        if always:
            flags[i] = dep_tag in _NOMINAL
            continue
        if head_tag == "name":
            continue
        if dep_tag == "num":
            flags[i] = head in NUMBER_HEADS
            continue
        if head in TIME_NOUNS and dep_tag == "noun" and dependent not in TIME_NOUNS:
            continue
        if head_tag == "adj":
            flags[i] = dep_tag in ("noun", "name", "pron", "det")
            continue
        if head_tag == "noun":
            flags[i] = dep_tag in _NOMINAL
            continue

    return flags
