"""Persian pronunciation dictionary.

The lexicon exists because of one fact about Persian orthography: the three
short vowels /a e o/ are not written. `کتب` is /kotob/, `کرم` is /kerm/ or
/karam/ or /kerem/, and no amount of letter-to-sound rule writing recovers
that. `rules.py` guesses; this module knows.

Entries are stored WITHOUT stress marks and WITHOUT the ezafe linker. Both
are decided in context and are applied by `g2p.py`; a lexicon entry is a
bare segmental transcription in the notation of `dcho.text.phonemes`.

Words are keyed after normalisation - Arabic letter variants folded and
ZWNJ removed - so `کتاب‌ها` and `کتابها` are the same key.

A word may have several readings. `lookup` returns them all, most frequent
first; choosing between them is homograph disambiguation and belongs to the
caller, which has the sentence context this module does not.

Module-level names
------------------
SEED           the built-in dictionary
DEFAULT        a Lexicon over SEED
VERB_FORMS     every inflected verb form the conjugator produced
STRESS_INDEX   for words whose stressed syllable is not the last one, the
               index of the stressed vowel counting vowels from the left.
               Verb morphology is the reason this table exists: `رفتم` is
               /ˈraftam/, not /rafˈtam/, and the stem/ending split that
               explains it is known here and nowhere else.
"""

from __future__ import annotations

import os

from .rules import normalize_word

__all__ = [
    "Lexicon",
    "SEED",
    "DEFAULT",
    "CURATED_WORDS",
    "VERB_FORMS",
    "VERB_STEMS",
    "STRESS_INDEX",
    "normalize_key",
]


def normalize_key(word: str) -> str:
    """The canonical lexicon key for a surface word."""
    return normalize_word(word)


def _count_vowels(phonemes: str) -> int:
    n = 0
    i = 0
    while i < len(phonemes):
        if phonemes[i] in "Aiueao":
            n += 1
        i += 1
    return n


# ==========================================================================
# Curated entries
# ==========================================================================
# One word per line: the orthographic form followed by its reading(s).
# Where a word has more than one reading they are ordered by frequency,
# because a caller with no context takes the first.

_CURATED = """
# --- pronouns, determiners, question words -----------------------------
من man
تو to
او u
وی vey
ما mA
شما shomA
آن An
آنها AnhA
این in
اینها inhA
ایشان ishAn
خود xod
خویش xish
خودم xodam
خودت xodat
خودش xodash
همدیگر hamdigar
یکدیگر yekdigar
همه hame
همین hamin
همان hamAn
هرکس harkas
هیچکس hichkas
هیچ hich
کسی kasi
بعضی ba'zi
برخی barxi
چند chand
چندین chandin
چنین chenin
چنان chenAn
چه che
چی chi
چرا cherA
کجا kojA
کی key
کیست kist
چیست chist
چطور chetovr
چگونه chegune
کدام kodAm
چقدر cheqadr
آیا AyA
فلان folAn
# --- prepositions, conjunctions, particles ------------------------------
به be
از az
با bA
در dar
بر bar
تا tA
را rA
و o
که ke
یا yA
اگر agar
اما ammA
ولی vali
چون chun
زیرا zirA
پس pas
هم ham
نیز niz
هر har
بی bi
برای barAy
بدون bedun
مانند mAnand
مثل mesl
درباره darbAre
روی ruy
پای pAy
زیر zir
بالا bAlA
پایین pAyin
کنار kenAr
وسط vasat
میان miyAn
بین beyn
پیش pish
بعد ba'd
قبل qabl
هنگام hengAm
نزد nazd
سوی suy
سمت samt
طرف taraf
علیه 'aleyh
نسبت nesbat
توسط tavassot
بلکه balke
یعنی ya'ni
البته albatte
حتی hattA
فقط faqat
دیگر digar
هنوز hanuz
باز bAz
همیشه hamishe
هرگز hargez
شاید shAyad
بله bale
بلی bali
آری Ari
خیر xeyr
نه na noh
اینکه inke
چنانچه chenAnche
وگرنه vagarne
همچنین hamchenin
بنابراین banAbarin
ضمن zemn
طی tey
جز joz
مگر magar
هرچند harchand
اگرچه agarche
# --- adverbs -------------------------------------------------------------
بسیار besyAr
خیلی xeyli
کم kam
کمی kami
زیاد ziyAd
سریع sari'
آهسته Aheste
تند tond
زود zud
دیر dir
حالا hAlA
اکنون aknun
الان al'An
اینجا injA
آنجا AnjA
همینجا haminjA
گاهی gAhi
ناگهان nAgahAn
دوباره dobAre
تنها tanhA
بیشتر bishtar
کمتر kamtar
بهتر behtar
بدتر badtar
امروز emruz
دیروز diruz
فردا fardA
امسال emsAl
پارسال pArsAl
پریروز pariruz
امشب emshab
دیشب dishab
هرگاه hargAh
سراسر sarAsar
# --- Arabic adverbs in ـاً ----------------------------------------------
حتماً hatman
لطفاً lotfan
مثلاً masalan
تقریباً taqriban
اصلاً aslan
ظاهراً zAheran
نسبتاً nesbatan
دقیقاً daqiqan
عملاً 'amalan
ابداً abadan
حقیقتاً haqiqatan
واقعاً vAqe'an
معمولاً ma'mulan
احتمالاً ehtemAlan
کاملاً kAmelan
فوراً fovran
مستقیماً mostaqiman
اتفاقاً ettefAqan
شخصاً shaxsan
بالاخره belaxare
# --- numerals -------------------------------------------------------------
صفر sefr
یک yek
دو do
سه se
چهار chahAr
پنج panj
شش shesh
هفت haft
هشت hasht
ده dah
یازده yAzdah
دوازده davAzdah
سیزده sizdah
چهارده chahArdah
پانزده pAnzdah
شانزده shAnzdah
هفده hefdah
هجده hejdah
نوزده nuzdah
بیست bist
سی si
چهل chehel
پنجاه panjAh
شصت shast
هفتاد haftAd
هشتاد hashtAd
نود navad
صد sad
دویست devist
سیصد sisad
چهارصد chahArsad
پانصد pAnsad
ششصد sheshsad
هفتصد haftsad
هشتصد hashtsad
نهصد nohsad
هزار hezAr
میلیون milyun
میلیارد milyArd
بیلیون bilyun
ممیز momayyez
درصد darsad
منفی manfi
نیم nim
ربع rob'
نصف nesf
اول avval
دوم dovvom
سوم sevvom
چهارم chahArom
پنجم panjom
ششم sheshom
هفتم haftom
هشتم hashtom
نهم nohom
دهم dahom
یکم yekom
بیستم bistom
صدم sadom
هزارم hezArom
# --- calendar -------------------------------------------------------------
فروردین farvardin
اردیبهشت ordibehesht
خرداد xordAd
تیر tir
مرداد mordAd
شهریور shahrivar
مهر mehr mohr
آبان AbAn
آذر Azar
دی dey
بهمن bahman
اسفند esfand
ژانویه zhAnviye
فوریه fevriye
مارس mArs
آوریل Avril
مه me
ژوئن zhu'an
ژوئیه zhu'iye
اوت ut
سپتامبر septAmbr
اکتبر oktobr
نوامبر novAmbr
دسامبر desAmbr
شنبه shanbe
یکشنبه yekshanbe
دوشنبه doshanbe
سهشنبه seshanbe
چهارشنبه chahArshanbe
پنجشنبه panjshanbe
جمعه jom'e
هفته hafte
روز ruz
ماه mAh
سال sAl
ساعت sA'at
دقیقه daqiqe
ثانیه sAniye
قرن qarn
صبح sobh
ظهر zohr
عصر 'asr
شب shab
بعدازظهر ba'dazzohr
نیمروز nimruz
# --- given names, titles, places -----------------------------------------
علی 'ali
محمد mohammad
حسن hasan
حسین hoseyn
رضا rezA
مهدی mahdi
احمد ahmad
امیر amir
مریم maryam
فاطمه fAteme
زهرا zahrA
سارا sArA
نرگس narges
لیلا leylA
زینب zeynab
پریسا parisA
بهرام bahrAm
کوروش kurosh
داریوش dAryush
بابک bAbak
سینا sinA
آرش Arash
نیما nimA
سهراب sohrAb
رستم rostam
فرهاد farhAd
شیرین shirin
یاسمن yAsaman
الهام elhAm
مینا minA
حمید hamid
مجید majid
سعید sa'id
جواد javAd
ناصر nAser
کریم karim
رحیم rahim
یوسف yusef
ابراهیم ebrAhim
موسی musA
عیسی 'isA
عبدالله 'abdollAh
عبدالحسین 'abdolhoseyn
محمدی mohammadi
حسینی hoseyni
احمدی ahmadi
آقا AqA
خانم xAnom
دکتر doktor
مهندس mohandes
استاد ostAd
حافظ hAfez
سعدی sa'di
فردوسی ferdovsi
مولوی movlavi
خیام xayyAm
نظامی nezAmi
رودکی rudaki
ایران irAn
تهران tehrAn
اصفهان esfahAn
شیراز shirAz
مشهد mashhad
تبریز tabriz
یزد yazd
کرمان kermAn
اهواز ahvAz
رشت rasht
قم qom
کرج karaj
همدان hamedAn
اراک arAk
زاهدان zAhedAn
ساری sAri
گرگان gorgAn
ارومیه orumiye
کیش kish
البرز alborz
دماوند damAvand
زاگرس zAgros
خزر xazar
فارس fArs
خلیج xalij
آسیا AsiyA
اروپا orupA
آمریکا AmrikA
آلمان AlmAn
فرانسه farAnse
انگلیس engelis
ژاپن zhApon
چین chin
هند hend
ترکیه torkiye
عراق 'erAq
افغانستان afqAnestAn
مصر mesr
شاهنامه shAhnAme
# --- nouns ----------------------------------------------------------------
آب Ab
آتش Atash
آدم Adam
آسمان AsemAn
آشپزخانه AshpazxAne
آفتاب AftAb
آواز AvAz
اتاق otAq
اثر asar
اداره edAre
ارتباط ertebAt
استفاده estefAde
اسم esm
اطلاعات ettelA'At
اخبار axbAr
امتحان emtahAn
انسان ensAn
اهمیت ahammiyat
بازار bAzAr
بازی bAzi
باغ bAq
باد bAd
باران bArAn
بخش baxsh
برادر barAdar
برف barf
برنامه barnAme
برنج berenj
بهار bahAr
بیمارستان bimArestAn
پدر pedar
پدربزرگ pedarbozorg
پارک pArk
پرنده parande
پسر pesar
پنجره panjare
پنیر panir
پول pul
پوست pust
پیراهن pirAhan
تابستان tAbestAn
تاریخ tArix
تخت taxt
تخته taxte
تخفیف taxfif
تصمیم tasmim
تعطیلات ta'tilAt
تفاوت tafAvot
تلفن telefon
تیم tim
جا jA
جامعه jAme'e
جان jAn
جدول jadval
جشن jashn
جلد jeld
جمعیت jam'iyat
جمله jomle
جنگ jang
جنوب jonub
جهان jahAn
جواب javAb
چاقو chAqu
چای chAy
چشم cheshm
چوب chub
چیز chiz
حال hAl
حرف harf
حیاط hayAt
خانه xAne
خاک xAk
خبر xabar
خدا xodA
خواب xAb
خواهر xAhar
خیابان xiyAbAn
دانشجو dAneshju
دانشگاه dAneshgAh
دانشآموز dAneshAmuz
دانش dAnesh
درخت deraxt
درد dard
درس dars
درجه daraje
دریا daryA
دست dast
دشت dasht
دفتر daftar
دل del
دما damA
دندان dandAn
دنیا donyA
دوست dust
دولت dovlat
دهکده dehkade
دیوار divAr
راه rAh
رنگ rang
روزنامه ruznAme
ریال riyAl
زبان zabAn
زمان zamAn
زمستان zemestAn
زمین zamin
زندگی zendegi
زن zan
زنگ zang
سؤال so'Al
سبد sabad
سخن soxan
سر sar
سفر safar
سفارش sefAresh
سلام salAm
سنگ sang
سود sud
سوال so'Al
شاعر shA'er
شانس shAns
شب shab
شرکت sherkat
شعر she'r
شغل shoql
شمال shomAl
شنا shenA
شهر shahr
صحبت sohbat
صدا sedA
صفحه safhe
صندلی sandali
ضرر zarar
طعم ta'm
عدد 'adad
عکس 'aks
علاقه 'alAqe
غذا qazA
غرب qarb
فردوس ferdovs
فرودگاه forudgAh
فرهنگ farhang
فریاد faryAd
فصل fasl
فکر fekr
قرار qarAr
قسمت qesmat
قیمت qeymat
کار kAr
کارخانه kArxAne
کاغذ kAqaz
کتاب ketAb
کتابخانه ketAbxAne
کشور keshvar
کلاس kelAs
کلمه kalame
کوچه kuche
کوه kuh
کویر kavir
کیف kif
کیلو kilu
کیلومتر kilumetr
گاو gAv
گردش gardesh
گروه goruh
گل gol gel
گندم gandom
گم gom
کد kod
ساختمان sAxtemAn
گوش gush
گوشت gusht
لباس lebAs
لیوان livAn
مادر mAdar
مار mAr
مال mAl
ماشین mAshin
مدرسه madrase
مدیر modir
مردم mardom
مرد mard
مسئله mas'ale
مسجد masjed
مساجد masAjed
مدارس madAres
کتب kotob
مشکل moshkel
مطلب matlab
معلم mo'allem
مقصد maqsad
منظره manzare
مهر mehr mohr
موضوع mozu'
موسیقی musiqi
میز miz
میوه mive
نام nAm
نامه nAme
نان nAn
نظر nazar
نفر nafar
نویسنده nevisande
نور nur
هدف hadaf
هوا havA
همسایه hamsAye
همکار hamkAr
همکاری hamkAri
وزیر vazir
وصف vasf
ورزش varzesh
ورق varaq
وقت vaqt
یاد yAd
پادشاه pAdeshAh
پایتخت pAytaxt
پزشک pezeshk
پزشکی pezeshki
پژوهش pazhuhesh
تعطیل ta'til
توجه tavajjoh
تومان tumAn
ترجمه tarjome
رعایت re'Ayat
رباعیات robA'iyAt
زبانزد zabAnzad
سرود sorud
شاهکار shAhkAr
شروع shoru'
عبارت 'ebArat
فناوری fanAvari
قطع qat'
کرم kerm karam kerem
مهندسی mohandesi
ملی melli
ریاضی riyAzi
فارسی fArsi
ایرانی irAni
تاریخی tArixi
بارانی bArAni
کویری kaviri
غربی qarbi
شرقی sharqi
شمالی shomAli
جنوبی jonubi
قدیمی qadimi
جدید jadid
بچه bachche
تنگ tang tong
پر par por
کشت kesht
دهم dahom
# --- adjectives -----------------------------------------------------------
آبی Abi
آرام ArAm
آسان AsAn
آماده AmAde
بد bad
بزرگ bozorg
بلند boland
بهتر behtar
بیدار bidAr
پاک pAk
تازه tAze
تمیز tamiz
تنها tanhA
جالب jAleb
جوان javAn
چاق chAq
خالی xAli
خراب xarAb
خشک xoshk
خنک xonak
خوب xub
خوشحال xoshhAl
دراز derAz
درست dorost
دشوار doshvAr
راست rAst
روشن rovshan
زیبا zibA
سبز sabz
سخت saxt
سرد sard
سفید sefid
سنگین sangin
سیاه siyAh
شاد shAd
شلوغ sholuq
شور shur
صحیح sahih
ضعیف za'if
عالی 'Ali
غمگین qamgin
قرمز qermez
قوی qavi
کوتاه kutAh
کوچک kuchak
گرم garm
گران gerAn
گرسنه gorosne
مشهور mashhur
معروف ma'ruf
مفید mofid
مناسب monAseb
مهربان mehrabAn
مهم mohemm
نازک nAzok
نرم narm
نزدیک nazdik
نو nov
وسیع vasi'
زرد zard
سالم sAlem
شیرین shirin
تلخ talx
ترش torsh
خوشمزه xoshmaze
پیر pir
ارزان arzAn
باهوش bAhush
تیز tiz
جدی jeddi
خسته xaste
دقیق daqiq
زنده zende
ساده sAde
شجاع shojA'
عجیب 'ajib
غنی qani
فقیر faqir
کافی kAfi
کامل kAmel
گذشته gozashte
متأسف mota'assef
متشکر motashakker
محکم mohkam
مرتب morattab
مطمئن motma'enn
ممکن momken
نامرتب nAmorattab
واقعی vAqe'i
همگی hamegi
سالانه sAlAne
روزانه ruzAne
گرد gerd
تلاش talAsh
ساله sAle
پیاده piyAde
زودتر zudtar
# --- loanwords ------------------------------------------------------------
آسانسور AsAnsor
اتوبوس otobus
اینترنت internet
ایمیل imeyl
پروژه porozhe
پیتزا pitzA
تاکسی tAksi
تلویزیون telvizyun
رادیو rAdiyo
سینما sinamA
کامپیوتر kAmpiyuter
گیتار gitAr
لپتاپ laptAp
موبایل mobAyl
هتل hotel
ایستگاه istgAh
بانک bAnk
پارکینگ pArking
تراکتور trAktor
دلار dolAr
فیلم film
کافه kAfe
مترو metro
متر metr
یورو yoro
ویدیو vidiyo
سوپرمارکت supermArket
# --- fixed compounds and derived forms ------------------------------------
خداحافظ xodAhAfez
نیمفاصله nimfAsele
بینظیر binazir
بیپایان bipAyAn
دستور dastur
سلامتی salAmati
خوشبختانه xoshbaxtAne
متأسفانه mota'assefAne
"""

# Words that resist the default "stress the last syllable" rule for
# reasons that are lexical rather than morphological.
_CURATED_STRESS = {
    "کافی": 0,
    "خداحافظ": 2,
    # گذشته is a lexicalised adjective ("past, last"), not the perfect
    # participle of گذشتن, and takes ordinary final stress.
    "گذشته": 2,
    "بله": 0,
    "بلی": 0,
    "آری": 0,
    "البته": 1,
    "یعنی": 0,
    "شاید": 0,
    "اینکه": 0,
    "بنابراین": 3,
}


def _parse_curated(text: str) -> dict:
    entries: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = normalize_key(parts[0])
        readings = entries.setdefault(key, [])
        for reading in parts[1:]:
            if reading not in readings:
                readings.append(reading)
    return entries


# ==========================================================================
# Verb conjugation
# ==========================================================================
# Persian verbs are almost entirely regular once the two stems are known,
# and every inflected form is high frequency, so they are generated rather
# than listed. Each row is:
#     (past stem orth, past stem phon, present stem orth, present stem phon)
# plus optional flags:
#     "nomi"  - present indicative takes no می (داشتن, دانستن in some uses)
#     "nobe"  - subjunctive takes no بـ (بودن)
#     "pre"   - has a separable preverb (برگشت, برداشت), which takes the
#               stress the way می/بـ/نـ do

_VERB_TABLE = [
    ("رفت", "raft", "رو", "rav", ""),
    ("کرد", "kard", "کن", "kon", ""),
    ("شد", "shod", "شو", "shav", ""),
    ("داشت", "dAsht", "دار", "dAr", "nomi"),
    ("گفت", "goft", "گو", "gu", ""),
    ("دید", "did", "بین", "bin", ""),
    ("داد", "dAd", "ده", "dah", ""),
    ("خواست", "xAst", "خواه", "xAh", ""),
    ("توانست", "tavAnest", "توان", "tavAn", ""),
    ("گرفت", "gereft", "گیر", "gir", ""),
    ("آمد", "Amad", "آی", "Ay", ""),
    ("خورد", "xord", "خور", "xor", ""),
    ("زد", "zad", "زن", "zan", ""),
    ("برد", "bord", "بر", "bar", ""),
    ("آورد", "Avord", "آور", "Avar", ""),
    ("خواند", "xAnd", "خوان", "xAn", ""),
    ("نوشت", "nevesht", "نویس", "nevis", ""),
    ("دانست", "dAnest", "دان", "dAn", ""),
    ("شناخت", "shenAxt", "شناس", "shenAs", ""),
    ("گذاشت", "gozAsht", "گذار", "gozAr", ""),
    ("گذشت", "gozasht", "گذر", "gozar", ""),
    ("ماند", "mAnd", "مان", "mAn", ""),
    ("نشست", "neshast", "نشین", "neshin", ""),
    ("ایستاد", "istAd", "ایست", "ist", ""),
    ("افتاد", "oftAd", "افت", "oft", ""),
    ("خرید", "xarid", "خر", "xar", ""),
    ("فروخت", "foruxt", "فروش", "forush", ""),
    ("پرسید", "porsid", "پرس", "pors", ""),
    ("رسید", "resid", "رس", "res", ""),
    ("دوید", "david", "دو", "dav", ""),
    ("فرستاد", "ferestAd", "فرست", "ferest", ""),
    ("کشید", "keshid", "کش", "kesh", ""),
    ("کشت", "kosht", "کش", "kosh", ""),
    ("مرد", "mord", "میر", "mir", ""),
    ("سرود", "sorud", "سرا", "sarA", ""),
    ("بست", "bast", "بند", "band", ""),
    ("شکست", "shekast", "شکن", "shekan", ""),
    ("ساخت", "sAxt", "ساز", "sAz", ""),
    ("انداخت", "andAxt", "انداز", "andAz", ""),
    ("پرداخت", "pardAxt", "پرداز", "pardAz", ""),
    ("سوخت", "suxt", "سوز", "suz", ""),
    ("آموخت", "Amuxt", "آموز", "Amuz", ""),
    ("ریخت", "rixt", "ریز", "riz", ""),
    ("گریخت", "gorixt", "گریز", "goriz", ""),
    ("دوخت", "duxt", "دوز", "duz", ""),
    ("پخت", "poxt", "پز", "paz", ""),
    ("باخت", "bAxt", "باز", "bAz", ""),
    ("شست", "shost", "شور", "shur", ""),
    ("جست", "jast", "جه", "jah", ""),
    ("خواب", "xAb", "خواب", "xAb", ""),
    ("خوابید", "xAbid", "خواب", "xAb", ""),
    ("خندید", "xandid", "خند", "xand", ""),
    ("گریست", "gerist", "گری", "gery", ""),
    ("ترسید", "tarsid", "ترس", "tars", ""),
    ("فهمید", "fahmid", "فهم", "fahm", ""),
    ("شنید", "shenid", "شنو", "shenav", ""),
    ("چید", "chid", "چین", "chin", ""),
    ("پوشید", "pushid", "پوش", "push", ""),
    ("رقصید", "raqsid", "رقص", "raqs", ""),
    ("پرید", "parid", "پر", "par", ""),
    ("چرخید", "charxid", "چرخ", "charx", ""),
    ("جنگید", "jangid", "جنگ", "jang", ""),
    ("بخشید", "baxshid", "بخش", "baxsh", ""),
    ("کوشید", "kushid", "کوش", "kush", ""),
    ("طلبید", "talabid", "طلب", "talab", ""),
    ("چسبید", "chasbid", "چسب", "chasb", ""),
    ("رنجید", "ranjid", "رنج", "ranj", ""),
    ("بارید", "bArid", "بار", "bAr", ""),
    ("پاشید", "pAshid", "پاش", "pAsh", ""),
    ("کاشت", "kAsht", "کار", "kAr", ""),
    ("داشت", "dAsht", "دار", "dAr", "nomi"),
    ("برداشت", "bardAsht", "بردار", "bardAr", "pre"),
    ("برگشت", "bargasht", "برگرد", "bargard", "pre"),
    ("پذیرفت", "paziroft", "پذیر", "pazir", ""),
    ("فرمود", "farmud", "فرما", "farmA", ""),
    ("نمود", "nemud", "نما", "namA", ""),
    ("افزود", "afzud", "افزا", "afzA", ""),
    ("گشود", "goshud", "گشا", "goshA", ""),
    ("ربود", "robud", "ربا", "robA", ""),
    ("آسود", "Asud", "آسا", "AsA", ""),
    ("زیست", "zist", "زی", "ziy", ""),
    ("رست", "rost", "ره", "rah", ""),
    ("خاست", "xAst", "خیز", "xiz", ""),
    ("کاست", "kAst", "کاه", "kAh", ""),
    ("یافت", "yAft", "یاب", "yAb", ""),
    ("بافت", "bAft", "باف", "bAf", ""),
    ("تافت", "tAft", "تاب", "tAb", ""),
    ("شتافت", "shetAft", "شتاب", "shetAb", ""),
    ("رفت", "raft", "رو", "rav", ""),
    ("بود", "bud", "باش", "bAsh", "nobe,nomi"),
]

_PERSON_ENDINGS = [
    ("م", "am"),
    ("ی", "i"),
    ("د", "ad"),
    ("یم", "im"),
    ("ید", "id"),
    ("ند", "and"),
]

_PAST_ENDINGS = [
    ("م", "am"),
    ("ی", "i"),
    ("", ""),
    ("یم", "im"),
    ("ید", "id"),
    ("ند", "and"),
]

_PHON_VOWELS = "Aiueao"


def _attach(stem_orth: str, stem_phon: str, ending_orth: str, ending_phon: str):
    """Attach an ending to a stem in both spellings at once.

    A vowel-final stem meeting a vowel-initial ending takes the /y/ glide,
    and the spelling grows the matching ی: گو + ـم -> گویم /guyam/,
    گو + ـیم -> گوییم /guyim/.
    """
    if not ending_orth and not ending_phon:
        return stem_orth, stem_phon
    glide = (
        stem_phon
        and ending_phon
        and stem_phon[-1] in _PHON_VOWELS
        and ending_phon[0] in _PHON_VOWELS
    )
    if glide:
        return stem_orth + "ی" + ending_orth, stem_phon + "y" + ending_phon
    return stem_orth + ending_orth, stem_phon + ending_phon


def _prefix_forms(prefix_orth: str, prefix_phon: str, stem_orth: str, stem_phon: str):
    """Attach a verbal prefix, returning every plausible spelling.

    Vowel-initial stems take an epenthetic /y/ after بـ and نـ and are
    respelled with ی: نـ + آمد -> نیامد /nayAmad/.
    """
    if stem_phon and stem_phon[0] in _PHON_VOWELS and prefix_phon in ("be", "na"):
        phon = prefix_phon[0] + ("iy" if prefix_phon == "be" else "ay") + stem_phon
        respelled = stem_orth
        if respelled.startswith("آ"):
            respelled = "ا" + respelled[1:]
        return [
            (prefix_orth + "ی" + respelled, phon),
            (prefix_orth + stem_orth, phon),
        ]
    return [(prefix_orth + stem_orth, prefix_phon + stem_phon)]


def _build_verbs():
    forms: dict[str, list[str]] = {}
    stress: dict[str, int] = {}
    stems: set[str] = set()

    def add(orth: str, phon: str, index: int) -> None:
        key = normalize_key(orth)
        readings = forms.setdefault(key, [])
        if phon not in readings:
            readings.append(phon)
        stress.setdefault(key, index)

    for past_o, past_p, pres_o, pres_p, flags in _VERB_TABLE:
        flagset = set(flags.split(",")) if flags else set()
        stems.add(normalize_key(past_o))
        stems.add(normalize_key(pres_o))
        past_head = 0 if "pre" in flagset else _count_vowels(past_p) - 1
        pres_head = 0 if "pre" in flagset else _count_vowels(pres_p) - 1

        def spread(stem_o, stem_p, endings, index):
            for end_o, end_p in endings:
                orth, phon = _attach(stem_o, stem_p, end_o, end_p)
                add(orth, phon, index)

        # Order matters, because the first reading recorded for a spelling
        # wins. Persian collapses the past stem and the present 3sg in
        # several verbs - خواند is both /xAnd/ and /xAnad/ - and the
        # frequent reading differs by shape: bare خورد is the past
        # /xord/, but می‌خورد is the present /mixorad/.
        spread(past_o, past_p, _PAST_ENDINGS, past_head)

        present_prefixes = [("می", "mi")]
        if "nomi" not in flagset:
            present_prefixes.append(("نمی", "nemi"))
        for prefix_o, prefix_p in present_prefixes:
            for orth, phon in _prefix_forms(prefix_o, prefix_p, pres_o, pres_p):
                spread(orth, phon, _PERSON_ENDINGS, 0)

        # Bare present: needed for داشتن, for the bare imperative in
        # compound verbs (روشن کن) and for compound-verb subjunctives
        # (صحبت کنم).
        add(pres_o, pres_p, pres_head)
        spread(pres_o, pres_p, _PERSON_ENDINGS, pres_head)
        subjunctive = [("ن", "na")]
        if "nobe" not in flagset:
            subjunctive.append(("ب", "be"))
        for prefix_o, prefix_p in subjunctive:
            for orth, phon in _prefix_forms(prefix_o, prefix_p, pres_o, pres_p):
                spread(orth, phon, _PERSON_ENDINGS, 0)

        for prefix_o, prefix_p in (("می", "mi"), ("نمی", "nemi"), ("ن", "na")):
            for orth, phon in _prefix_forms(prefix_o, prefix_p, past_o, past_p):
                spread(orth, phon, _PAST_ENDINGS, 0)

        # Imperative and past participle.
        if "nobe" not in flagset:
            for orth, phon in _prefix_forms("ب", "be", pres_o, pres_p):
                add(orth, phon, 0)
        for orth, phon in _prefix_forms("ن", "na", pres_o, pres_p):
            add(orth, phon, 0)
        add(past_o + "ه", past_p + "e", past_head)
        add(past_o + "ن", past_p + "an", past_head + 1)

    return forms, stress, stems


_IRREGULAR_VERBS = """
است ast
هست hast
هستم hastam
هستی hasti
هستیم hastim
هستید hastid
هستند hastand
نیست nist
نیستم nistam
نیستی nisti
نیستیم nistim
نیستید nistid
نیستند nistand
باید bAyad
نباید nabAyad
بگو begu
بگویید beguyid
بده bedeh
بدهید bedehid
بیا biyA
بیایید biyAyid
برو boro
بروید boravid
نرو naro
باش bAsh
نباش nabAsh
"""

_IRREGULAR_STRESS = {
    "هستم": 0, "هستی": 0, "هست": 0, "هستیم": 0, "هستید": 0, "هستند": 0,
    "نیستم": 0, "نیستی": 0, "نیست": 0, "نیستیم": 0, "نیستید": 0,
    "نیستند": 0, "باید": 0, "نباید": 0, "بگو": 0, "بگویید": 0,
    "بده": 0, "بدهید": 0, "بیا": 0, "بیایید": 0, "برو": 0, "بروید": 0,
    "نرو": 0, "باش": 0, "نباش": 0, "است": 0,
}


def _build_seed():
    seed = _parse_curated(_CURATED)
    curated = frozenset(seed)
    verbs, verb_stress, stems = _build_verbs()

    irregular = _parse_curated(_IRREGULAR_VERBS)
    for key, readings in irregular.items():
        target = verbs.setdefault(key, [])
        for reading in readings:
            if reading not in target:
                target.insert(0, reading)

    for key, readings in verbs.items():
        target = seed.setdefault(key, [])
        for reading in readings:
            if reading not in target:
                target.append(reading)

    # A verb form that is also a listed noun keeps the noun's ordinary
    # final stress: مردم is /mardom/ "people", not /mordam/ "I died".
    stress = {k: v for k, v in verb_stress.items() if k not in curated}
    for word, index in _IRREGULAR_STRESS.items():
        stress[normalize_key(word)] = index
    for word, index in _CURATED_STRESS.items():
        stress[normalize_key(word)] = index

    return seed, curated, frozenset(verbs), frozenset(stems), stress


SEED, CURATED_WORDS, VERB_FORMS, VERB_STEMS, STRESS_INDEX = _build_seed()


# ==========================================================================
# The container
# ==========================================================================


class Lexicon:
    """A word -> readings dictionary with TSV persistence."""

    def __init__(self, entries: dict[str, list[str]] | None = None) -> None:
        self._entries: dict[str, list[str]] = {}
        if entries:
            for word, readings in entries.items():
                if isinstance(readings, str):
                    readings = [readings]
                self._entries[normalize_key(word)] = list(readings)

    def lookup(self, word: str) -> list[str]:
        """Every reading of `word`, best first; `[]` when unknown."""
        return list(self._entries.get(normalize_key(word), ()))

    def add(self, word: str, phonemes: str) -> None:
        """Record a reading, keeping any earlier ones for the same word."""
        key = normalize_key(word)
        readings = self._entries.setdefault(key, [])
        if phonemes not in readings:
            readings.append(phonemes)

    def __contains__(self, word: str) -> bool:
        return normalize_key(word) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def items(self):
        return self._entries.items()

    @classmethod
    def load(cls, path) -> "Lexicon":
        """Read a TSV file: word<TAB>phonemes[<TAB>phonemes...]."""
        entries: dict[str, list[str]] = {}
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                word = parts[0].strip()
                readings = [p.strip() for p in parts[1:] if p.strip()]
                if not word or not readings:
                    continue
                key = normalize_key(word)
                target = entries.setdefault(key, [])
                for reading in readings:
                    if reading not in target:
                        target.append(reading)
        return cls(entries)

    def save(self, path) -> None:
        """Write the lexicon back out in the format `load` reads."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for word in sorted(self._entries):
                handle.write(word + "\t" + "\t".join(self._entries[word]) + "\n")


DEFAULT = Lexicon(SEED)
