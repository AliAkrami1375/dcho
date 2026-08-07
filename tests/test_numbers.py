"""Tests for dcho.text.numbers."""

import unittest

from dcho.text.numbers import (
    UNITS,
    ZWNJ,
    expand_numbers,
    fraction_to_words,
    normalize_digits,
    number_to_words,
)


class TestNormalizeDigits(unittest.TestCase):
    def test_persian_digits(self):
        self.assertEqual(normalize_digits("۰۱۲۳۴۵۶۷۸۹"), "0123456789")

    def test_arabic_indic_digits(self):
        self.assertEqual(normalize_digits("٠١٢٣٤٥٦٧٨٩"), "0123456789")

    def test_mixed_with_text(self):
        self.assertEqual(normalize_digits("خانه ۱۲ پلاک ٣"), "خانه 12 پلاک 3")

    def test_ascii_untouched(self):
        self.assertEqual(normalize_digits("abc 42"), "abc 42")


class TestCardinals(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(number_to_words(0), "صفر")

    def test_single_digits(self):
        self.assertEqual(number_to_words(1), "یک")
        self.assertEqual(number_to_words(5), "پنج")
        self.assertEqual(number_to_words(9), "نه")

    def test_teens(self):
        self.assertEqual(number_to_words(10), "ده")
        self.assertEqual(number_to_words(11), "یازده")
        self.assertEqual(number_to_words(15), "پانزده")
        self.assertEqual(number_to_words(17), "هفده")
        self.assertEqual(number_to_words(19), "نوزده")

    def test_tens(self):
        self.assertEqual(number_to_words(20), "بیست")
        self.assertEqual(number_to_words(30), "سی")
        self.assertEqual(number_to_words(90), "نود")

    def test_compound_tens(self):
        self.assertEqual(number_to_words(21), "بیست و یک")
        self.assertEqual(number_to_words(42), "چهل و دو")
        self.assertEqual(number_to_words(99), "نود و نه")

    def test_hundreds(self):
        self.assertEqual(number_to_words(100), "صد")
        self.assertEqual(number_to_words(300), "سیصد")
        self.assertEqual(number_to_words(500), "پانصد")
        self.assertEqual(number_to_words(900), "نهصد")

    def test_compound_hundreds(self):
        self.assertEqual(number_to_words(101), "صد و یک")
        self.assertEqual(number_to_words(110), "صد و ده")
        self.assertEqual(number_to_words(999), "نهصد و نود و نه")

    def test_thousand_has_no_yek(self):
        self.assertEqual(number_to_words(1000), "هزار")

    def test_thousands(self):
        self.assertEqual(number_to_words(1001), "هزار و یک")
        self.assertEqual(number_to_words(1100), "هزار و صد")
        self.assertEqual(number_to_words(1375), "هزار و سیصد و هفتاد و پنج")
        self.assertEqual(number_to_words(2000), "دو هزار")
        self.assertEqual(number_to_words(100000), "صد هزار")

    def test_millions(self):
        self.assertEqual(number_to_words(1000000), "یک میلیون")
        self.assertEqual(number_to_words(1000001), "یک میلیون و یک")
        self.assertEqual(
            number_to_words(1234567),
            "یک میلیون و دویست و سی و چهار هزار و پانصد و شصت و هفت",
        )

    def test_large_scales(self):
        self.assertEqual(number_to_words(10**9), "یک میلیارد")
        self.assertEqual(number_to_words(2 * 10**9), "دو میلیارد")
        self.assertEqual(number_to_words(10**12), "یک بیلیون")
        self.assertEqual(number_to_words(10**15), "یک بیلیارد")

    def test_negative(self):
        self.assertEqual(number_to_words(-5), "منفی پنج")
        self.assertEqual(number_to_words(-1375), "منفی هزار و سیصد و هفتاد و پنج")


class TestOrdinals(unittest.TestCase):
    def test_irregulars(self):
        self.assertEqual(number_to_words(1, ordinal=True), "اول")
        self.assertEqual(number_to_words(3, ordinal=True), "سوم")

    def test_regular_units(self):
        self.assertEqual(number_to_words(2, ordinal=True), "دوم")
        self.assertEqual(number_to_words(4, ordinal=True), "چهارم")
        self.assertEqual(number_to_words(5, ordinal=True), "پنجم")
        self.assertEqual(number_to_words(9, ordinal=True), "نهم")
        self.assertEqual(number_to_words(10, ordinal=True), "دهم")
        self.assertEqual(number_to_words(11, ordinal=True), "یازدهم")

    def test_thirty_takes_am(self):
        self.assertEqual(number_to_words(30, ordinal=True), "سی" + ZWNJ + "ام")
        self.assertEqual(number_to_words(130, ordinal=True), "صد و سی" + ZWNJ + "ام")

    def test_three_hundred_is_not_thirty(self):
        self.assertEqual(number_to_words(300, ordinal=True), "سیصدم")

    def test_compound_ordinals(self):
        self.assertEqual(number_to_words(20, ordinal=True), "بیستم")
        self.assertEqual(number_to_words(21, ordinal=True), "بیست و یکم")
        self.assertEqual(number_to_words(23, ordinal=True), "بیست و سوم")
        self.assertEqual(number_to_words(29, ordinal=True), "بیست و نهم")

    def test_scale_ordinals(self):
        self.assertEqual(number_to_words(100, ordinal=True), "صدم")
        self.assertEqual(number_to_words(1000, ordinal=True), "هزارم")


class TestDecimals(unittest.TestCase):
    def test_pi(self):
        self.assertEqual(number_to_words(3.14), "سه ممیز چهارده صدم")

    def test_half(self):
        self.assertEqual(number_to_words(0.5), "صفر ممیز پنج دهم")

    def test_one_and_a_half(self):
        self.assertEqual(number_to_words(1.5), "یک ممیز پنج دهم")

    def test_thousandths(self):
        self.assertEqual(number_to_words(2.125), "دو ممیز صد و بیست و پنج هزارم")

    def test_integral_float(self):
        self.assertEqual(number_to_words(4.0), "چهار")

    def test_negative_decimal(self):
        self.assertEqual(number_to_words(-0.25), "منفی صفر ممیز بیست و پنج صدم")

    def test_in_text(self):
        self.assertEqual(expand_numbers("۳٫۱۴"), "سه ممیز چهارده صدم")
        self.assertEqual(expand_numbers("0.5"), "صفر ممیز پنج دهم")


class TestFractions(unittest.TestCase):
    def test_three_quarters(self):
        self.assertEqual(expand_numbers("3/4"), "سه چهارم")

    def test_one_half(self):
        self.assertEqual(expand_numbers("1/2"), "یک دوم")

    def test_two_thirds(self):
        self.assertEqual(expand_numbers("۲/۳"), "دو سوم")

    def test_alias(self):
        self.assertEqual(fraction_to_words(1, 2, use_aliases=True), "نیم")
        self.assertEqual(fraction_to_words(1, 2), "یک دوم")


class TestPercent(unittest.TestCase):
    def test_ascii_suffix(self):
        self.assertEqual(expand_numbers("45%"), "چهل و پنج درصد")

    def test_persian_prefix(self):
        self.assertEqual(expand_numbers("٪۴۵"), "چهل و پنج درصد")

    def test_hundred_percent(self):
        self.assertEqual(expand_numbers("100%"), "صد درصد")

    def test_decimal_percent(self):
        self.assertEqual(expand_numbers("۱۲.۵٪"), "دوازده ممیز پنج دهم درصد")


class TestCurrency(unittest.TestCase):
    def test_toman(self):
        self.assertEqual(expand_numbers("۵۰۰۰ تومان"), "پنج هزار تومان")

    def test_abbreviated_toman(self):
        self.assertEqual(expand_numbers("۵۰۰۰ت"), "پنج هزار تومان")

    def test_rial(self):
        self.assertEqual(expand_numbers("۱۲۰۰ ریال"), "هزار و دویست ریال")

    def test_dollar_and_euro(self):
        self.assertEqual(expand_numbers("۵۰ دلار"), "پنجاه دلار")
        self.assertEqual(expand_numbers("۲۰ یورو"), "بیست یورو")

    def test_grouped_thousands(self):
        self.assertEqual(expand_numbers("1,000 تومان"), "هزار تومان")

    def test_ta_is_not_toman(self):
        self.assertEqual(expand_numbers("۵ تا سیب"), "پنج تا سیب")


class TestDates(unittest.TestCase):
    def test_jalali(self):
        self.assertEqual(
            expand_numbers("1403/5/12"), "دوازدهم مرداد هزار و چهارصد و سه"
        )

    def test_jalali_zero_padded_persian(self):
        self.assertEqual(
            expand_numbers("۱۴۰۳/۰۵/۱۲"), "دوازدهم مرداد هزار و چهارصد و سه"
        )

    def test_jalali_first_of_farvardin(self):
        self.assertEqual(
            expand_numbers("1403/1/1"), "اول فروردین هزار و چهارصد و سه"
        )

    def test_gregorian_iso(self):
        self.assertEqual(
            expand_numbers("2024-03-15"), "پانزدهم مارس دو هزار و بیست و چهار"
        )

    def test_gregorian_year_with_slashes(self):
        self.assertEqual(
            expand_numbers("2024/03/15"), "پانزدهم مارس دو هزار و بیست و چهار"
        )

    def test_invalid_month_is_not_a_date(self):
        self.assertNotIn("فروردین", expand_numbers("1403/19/12"))


class TestTime(unittest.TestCase):
    def test_hours_and_minutes(self):
        self.assertEqual(expand_numbers("14:30"), "ساعت چهارده و سی دقیقه")

    def test_zero_padded_minutes(self):
        self.assertEqual(expand_numbers("۹:۰۵"), "ساعت نه و پنج دقیقه")

    def test_whole_hour(self):
        self.assertEqual(expand_numbers("8:00"), "ساعت هشت")
        self.assertEqual(expand_numbers("12:00"), "ساعت دوازده")

    def test_with_seconds(self):
        self.assertEqual(
            expand_numbers("14:30:15"), "ساعت چهارده و سی دقیقه و پانزده ثانیه"
        )

    def test_no_duplicate_saat(self):
        self.assertEqual(expand_numbers("ساعت ۸:۰۰"), "ساعت هشت")


class TestDigitRuns(unittest.TestCase):
    def test_mobile_number(self):
        self.assertEqual(
            expand_numbers("۰۹۱۲۳۴۵۶۷۸۹"),
            "صفر نه یک دو سه چهار پنج شش هفت هشت نه",
        )

    def test_eight_digits(self):
        self.assertEqual(
            expand_numbers("12345678"), "یک دو سه چهار پنج شش هفت هشت"
        )

    def test_seven_digits_is_a_cardinal(self):
        self.assertEqual(expand_numbers("1234567").split()[0], "یک")

    def test_leading_zero_is_read_out(self):
        self.assertEqual(expand_numbers("۰۲۱"), "صفر دو یک")

    def test_dashed_landline(self):
        self.assertEqual(
            expand_numbers("۰۲۱-۸۸۷۷۶۶۵۵"),
            "صفر دو یک هشت هشت هفت هفت شش شش پنج پنج",
        )

    def test_spaced_mobile(self):
        self.assertEqual(
            expand_numbers("۰۹۱۲ ۳۴۵ ۶۷۸۹"),
            "صفر نه یک دو سه چهار پنج شش هفت هشت نه",
        )


class TestRanges(unittest.TestCase):
    def test_dash_range(self):
        self.assertEqual(expand_numbers("۱۰-۲۰"), "ده تا بیست")

    def test_word_range(self):
        self.assertEqual(expand_numbers("۱۰ تا ۲۰"), "ده تا بیست")

    def test_negative_is_not_a_range(self):
        self.assertEqual(expand_numbers("دمای -۵ درجه"), "دمای منفی پنج درجه")

    def test_wide_range_stays_a_range(self):
        self.assertEqual(expand_numbers("۱۰۰۰-۲۰۰۰"), "هزار تا دو هزار")

    def test_time_range(self):
        self.assertEqual(
            expand_numbers("از ۹:۰۰ تا ۱۱:۳۰"),
            "از ساعت نه تا یازده و سی دقیقه",
        )


class TestOrdinalSuffixInText(unittest.TestCase):
    def test_bare_suffix(self):
        self.assertEqual(expand_numbers("۳م"), "سوم")

    def test_dashed_suffix(self):
        self.assertEqual(expand_numbers("۳-ام"), "سوم")

    def test_am_suffix(self):
        self.assertEqual(expand_numbers("۲۰ام"), "بیستم")

    def test_million_is_not_an_ordinal_suffix(self):
        self.assertEqual(expand_numbers("۳ میلیون"), "سه میلیون")


class TestUnits(unittest.TestCase):
    def test_table_entries(self):
        self.assertEqual(UNITS["km"], "کیلومتر")
        self.assertEqual(UNITS["kg"], "کیلوگرم")
        self.assertEqual(UNITS["MB"], "مگابایت")

    def test_expansion(self):
        self.assertEqual(expand_numbers("5 km"), "پنج کیلومتر")
        self.assertEqual(expand_numbers("۱۰ kg"), "ده کیلوگرم")
        self.assertEqual(expand_numbers("700MB"), "هفتصد مگابایت")


class TestSentences(unittest.TestCase):
    def test_mixed_sentence(self):
        self.assertEqual(
            expand_numbers("جلسه ساعت ۱۴:۳۰ روز ۱۴۰۳/۰۵/۱۲ برگزار می‌شود."),
            "جلسه ساعت چهارده و سی دقیقه روز دوازدهم مرداد "
            "هزار و چهارصد و سه برگزار می‌شود.",
        )

    def test_price_sentence(self):
        self.assertEqual(
            expand_numbers("قیمت ۲۵۰۰۰ تومان و تخفیف ۱۵٪ است"),
            "قیمت بیست و پنج هزار تومان و تخفیف پانزده درصد است",
        )

    def test_text_without_numbers_is_unchanged(self):
        text = "این جمله هیچ عددی ندارد."
        self.assertEqual(expand_numbers(text), text)

    def test_empty(self):
        self.assertEqual(expand_numbers(""), "")


if __name__ == "__main__":
    unittest.main()
