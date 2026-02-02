"""
PII Validation using External Libraries

This module provides validation for detected PII entities using industry-standard
libraries instead of pattern-only matching.

Libraries used:
- python-stdnum: Validates national IDs, IBAN, BIC/SWIFT, credit cards for 35+ countries
- phonenumbers: International phone number validation for 249 countries

Sources:
- IBAN patterns: php-iban registry (116 countries)
  https://github.com/globalcitizen/php-iban
- Phone patterns: ariankoochak regex patterns
  https://github.com/ariankoochak/regex-patterns-of-all-countries

Usage:
    from hush_engine.detectors.validators import (
        validate_iban, validate_phone, validate_national_id, validate_credit_card
    )

    if validate_iban("DE89370400440532013000"):
        print("Valid IBAN")
"""

from typing import Optional, Tuple, Dict, Any, List
import re


# =============================================================================
# IBAN COUNTRY SPECIFICATIONS (from php-iban registry - 116 countries)
# =============================================================================
# Format: country_code -> (iban_length, bban_pattern_description)
# Pattern key: !n = numeric, !a = alphabetic, !c = alphanumeric

IBAN_COUNTRY_SPECS = {
    "AL": (28, "8!n16!c"),      # Albania
    "DZ": (24, "20!n"),         # Algeria
    "AD": (24, "4!n4!n12!c"),   # Andorra
    "AO": (25, "21!n"),         # Angola
    "AT": (20, "5!n11!n"),      # Austria
    "AZ": (28, "4!a20!c"),      # Azerbaijan
    "BH": (22, "4!a14!c"),      # Bahrain
    "BY": (28, "4!c4!n16!c"),   # Belarus
    "BE": (16, "3!n7!n2!n"),    # Belgium
    "BJ": (28, "1!a23!n"),      # Benin
    "BA": (20, "3!n3!n8!n2!n"), # Bosnia and Herzegovina
    "BR": (29, "8!n5!n10!n1!a1!c"),  # Brazil
    "VG": (24, "4!a16!n"),      # British Virgin Islands
    "BG": (22, "4!a4!n2!n8!c"), # Bulgaria
    "BF": (27, "23!n"),         # Burkina Faso
    "BI": (16, "12!n"),         # Burundi
    "CM": (27, "23!n"),         # Cameroon
    "CV": (25, "21!n"),         # Cape Verde
    "CF": (27, "5!n5!n11!n2!n"),  # Central African Republic
    "TD": (27, "5!n5!n11!n2!n"),  # Chad
    "KM": (27, "5!n5!n13!n2!n"),  # Comoros
    "CG": (27, "5!n5!n11!n2!n"),  # Congo
    "CR": (22, "4!n14!n"),      # Costa Rica
    "CI": (28, "1!a23!n"),      # Côte d'Ivoire
    "HR": (21, "7!n10!n"),      # Croatia
    "CY": (28, "3!n5!n16!c"),   # Cyprus
    "CZ": (24, "4!n6!n10!n"),   # Czech Republic
    "DK": (18, "4!n9!n1!n"),    # Denmark
    "DJ": (27, "5!n5!n13!n2!n"),  # Djibouti
    "DO": (28, "4!c20!n"),      # Dominican Republic
    "EG": (29, "4!n4!n17!n"),   # Egypt
    "SV": (28, "4!a20!n"),      # El Salvador
    "GQ": (27, "5!n5!n11!n2!n"),  # Equatorial Guinea
    "EE": (20, "2!n2!n11!n1!n"),  # Estonia
    "FO": (18, "4!n9!n1!n"),    # Faroe Islands
    "FI": (18, "6!n7!n1!n"),    # Finland
    "AX": (18, "6!n7!n1!n"),    # Åland Islands
    "FR": (27, "5!n5!n11!c2!n"),  # France
    "GF": (27, "5!n5!n11!c2!n"),  # French Guiana
    "PF": (27, "5!n5!n11!c2!n"),  # French Polynesia
    "TF": (27, "5!n5!n11!c2!n"),  # French Southern Territories
    "GA": (27, "5!n5!n11!n2!n"),  # Gabon
    "GE": (22, "2!a16!n"),      # Georgia
    "DE": (22, "8!n10!n"),      # Germany
    "GI": (23, "4!a15!c"),      # Gibraltar
    "GR": (27, "3!n4!n16!c"),   # Greece
    "GL": (18, "4!n9!n1!n"),    # Greenland
    "GP": (27, "5!n5!n11!c2!n"),  # Guadeloupe
    "GT": (28, "4!c20!c"),      # Guatemala
    "GW": (25, "2!c2!n4!n11!n2!n"),  # Guinea-Bissau
    "HN": (28, "4!a20!n"),      # Honduras
    "HU": (28, "3!n4!n1!n15!n1!n"),  # Hungary
    "IS": (26, "4!n2!n6!n10!n"),  # Iceland
    "AA": (16, "12!a"),         # IIBAN (Internet)
    "IR": (26, "22!n"),         # Iran
    "IQ": (23, "4!a3!n12!n"),   # Iraq
    "IE": (22, "4!a6!n8!n"),    # Ireland
    "IL": (23, "3!n3!n13!n"),   # Israel
    "IT": (27, "1!a5!n5!n12!c"),  # Italy
    "JO": (30, "4!a4!n18!c"),   # Jordan
    "KZ": (20, "3!n13!c"),      # Kazakhstan
    "XK": (20, "4!n10!n2!n"),   # Kosovo
    "KW": (30, "4!a22!c"),      # Kuwait
    "LV": (21, "4!a13!c"),      # Latvia
    "LB": (28, "4!n20!c"),      # Lebanon
    "LI": (21, "5!n12!c"),      # Liechtenstein
    "LT": (20, "5!n11!n"),      # Lithuania
    "LU": (20, "3!n13!c"),      # Luxembourg
    "MK": (19, "3!n10!c2!n"),   # Macedonia
    "MG": (27, "23!n"),         # Madagascar
    "ML": (28, "1!a23!n"),      # Mali
    "MT": (31, "4!a5!n18!c"),   # Malta
    "MR": (27, "5!n5!n11!n2!n"),  # Mauritania
    "MU": (30, "4!a2!n2!n12!n3!n3!a"),  # Mauritius
    "MD": (24, "2!c18!c"),      # Moldova
    "MC": (27, "5!n5!n11!c2!n"),  # Monaco
    "ME": (22, "3!n13!n2!n"),   # Montenegro
    "MA": (28, "3!n5!n14!n2!n"),  # Morocco
    "MZ": (25, "21!n"),         # Mozambique
    "MQ": (27, "5!n5!n11!c2!n"),  # Martinique
    "YT": (27, "5!n5!n11!c2!n"),  # Mayotte
    "NC": (27, "5!n5!n11!c2!n"),  # New Caledonia
    "NL": (18, "4!a10!n"),      # Netherlands
    "NI": (32, "4!a24!n"),      # Nicaragua
    "NE": (28, "2!a3!n5!n12!n2!n"),  # Niger
    "NO": (15, "4!n6!n1!n"),    # Norway
    "PK": (24, "4!a16!c"),      # Pakistan
    "PS": (29, "4!a21!c"),      # Palestine
    "PL": (28, "8!n16!n"),      # Poland
    "PT": (25, "4!n4!n11!n2!n"),  # Portugal
    "QA": (29, "4!a4!n17!c"),   # Qatar
    "RE": (27, "5!n5!n11!c2!n"),  # Réunion
    "RO": (24, "4!a16!c"),      # Romania
    "LC": (32, "4!a24!c"),      # Saint Lucia
    "BL": (27, "5!n5!n11!c2!n"),  # Saint Barthélemy
    "MF": (27, "5!n5!n11!c2!n"),  # Saint Martin
    "PM": (27, "5!n5!n11!c2!n"),  # Saint-Pierre and Miquelon
    "SM": (27, "1!a5!n5!n12!c"),  # San Marino
    "ST": (25, "8!n11!n2!n"),   # São Tomé and Príncipe
    "SA": (24, "2!n18!c"),      # Saudi Arabia
    "SN": (28, "1!a23!n"),      # Senegal
    "RS": (22, "3!n13!n2!n"),   # Serbia
    "SC": (31, "4!a2!n2!n16!n3!a"),  # Seychelles
    "SK": (24, "4!n6!n10!n"),   # Slovakia
    "SI": (19, "5!n8!n2!n"),    # Slovenia
    "ES": (24, "4!n4!n1!n1!n10!n"),  # Spain
    "SE": (24, "3!n16!n1!n"),   # Sweden
    "CH": (21, "5!n12!c"),      # Switzerland
    "TL": (23, "3!n14!n2!n"),   # Timor-Leste
    "TG": (28, "2!a3!n5!n12!n2!n"),  # Togo
    "TN": (24, "2!n3!n13!n2!n"),  # Tunisia
    "TR": (26, "5!n1!n16!c"),   # Turkey
    "UA": (29, "6!n19!c"),      # Ukraine
    "AE": (23, "3!n16!n"),      # United Arab Emirates
    "GB": (22, "4!a6!n8!n"),    # United Kingdom
    "WF": (27, "5!n5!n11!c2!n"),  # Wallis and Futuna
}


# =============================================================================
# PHONE NUMBER PATTERNS (from ariankoochak - 100+ countries)
# =============================================================================
# Format: country_code -> regex_pattern

PHONE_PATTERNS = {
    "AF": r"^(\+93|0)?(2{1}[0-8]{1}|[3-5]{1}[0-4]{1})(\d{7})$",  # Afghanistan
    "AL": r"^\+355[2-9]\d{7,8}$",  # Albania
    "DZ": r"^(\+?213|0)(5|6|7)\d{8}$",  # Algeria
    "AD": r"^(\+376)?[346]\d{5}$",  # Andorra
    "AO": r"^(\+244)\d{9}$",  # Angola
    "AR": r"^\+?549(11|[2368]\d)\d{8}$",  # Argentina
    "AM": r"^(\+?374|0)(33|4[134]|55|77|88|9[13-689])\d{6}$",  # Armenia
    "AU": r"^(\+?61|0)4\d{8}$",  # Australia
    "AT": r"^\+43[1-9][0-9]{3,12}$",  # Austria
    "AZ": r"^(\+994|0)(10|5[015]|7[07]|99)\d{7}$",  # Azerbaijan
    "BH": r"^(\+?973)?(3|6)\d{7}$",  # Bahrain
    "BD": r"^(\+?880|0)1[13456789][0-9]{8}$",  # Bangladesh
    "BY": r"^(\+?375)?(24|25|29|33|44)\d{7}$",  # Belarus
    "BE": r"^(\+?32|0)4\d{8}$",  # Belgium
    "BJ": r"^(\+229)\d{8}$",  # Benin
    "BT": r"^(\+?975|0)?(17|16|77|02)\d{6}$",  # Bhutan
    "BO": r"^(\+?591)?(6|7)\d{7}$",  # Bolivia
    "BA": r"^((((\+|00)3876)|06))((([0-3]|[5-6])\d{6})|(4\d{7}))$",  # Bosnia and Herzegovina
    "BW": r"^(\+?267)?(7[1-8]{1})\d{6}$",  # Botswana
    "BR": r"^((\+?55 ?[1-9]{2} ?)|(\+?55 ?\([1-9]{2}\) ?)|(0[1-9]{2} ?)|(\([1-9]{2}\) ?)|([1-9]{2} ?))((\d{4}\-?\d{4})|(9[1-9]{1}\d{3}\-?\d{4}))$",  # Brazil
    "BG": r"^(\+?359|0)?8[789]\d{7}$",  # Bulgaria
    "BF": r"^(\+226|0)[67]\d{7}$",  # Burkina Faso
    "CM": r"^(\+?237)6[0-9]{8}$",  # Cameroon
    "CA": r"^((\+1|1)?( |-)?)?(\(?\d{3}\)?( |-)?)?(\d{3}( |-)?[0-9]{4})$",  # Canada
    "CF": r"^(\+?236| ?)(70|75|77|72|21|22)\d{6}$",  # Central African Republic
    "CL": r"^(\+?56|0)[2-9]\d{1}\d{7}$",  # Chile
    "CN": r"^((\+|00)86)?(1[3-9]|9[28])\d{9}$",  # China
    "CO": r"^(\+?57)?3(0(0|1|2|4|5)|1\d|2[0-4]|5(0|1))\d{7}$",  # Colombia
    "CD": r"^(\+?243|0)?(8|9)\d{8}$",  # Congo (DRC)
    "CR": r"^(\+506)?[2-8]\d{7}$",  # Costa Rica
    "HR": r"^\+385[1-9][0-9]{7,8}$",  # Croatia
    "CU": r"^(\+53|0053)?5\d{7}$",  # Cuba
    "CY": r"^(\+?357?)?(9(9|6)\d{6})$",  # Cyprus
    "CZ": r"^(\+?420)? ?[1-9][0-9]{2} ?[0-9]{3} ?[0-9]{3}$",  # Czech Republic
    "DK": r"^(\+?45)?\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}$",  # Denmark
    "DO": r"^(\+?1)?8[024]9\d{7}$",  # Dominican Republic
    "EC": r"^(\+?593|0)([2-7]|9[2-9])\d{7}$",  # Ecuador
    "EG": r"^((\+?20)|0)?1[0125]\d{8}$",  # Egypt
    "SV": r"^(\+?503)?[67]\d{7}$",  # El Salvador
    "EE": r"^(\+?372)?\s?(5|8[1-4])\s?([0-9]\s?){6,7}$",  # Estonia
    "FO": r"^(\+?298)?\s?\d{2}\s?\d{2}\s?\d{2}$",  # Faroe Islands
    "FJ": r"^(\+?679)?\s?\d{3}\s?\d{4}$",  # Fiji
    "FI": r"^(\+?358|0)\s?(4[0-6]|50)\s?(\d\s?){4,8}$",  # Finland
    "FR": r"^(\+?33|0)[67]\d{8}$",  # France
    "GF": r"^(\+?594|0|00594)[67]\d{8}$",  # French Guiana
    "PF": r"^(\+?689)?8[789]\d{6}$",  # French Polynesia
    "GE": r"^(\+?995)?(79\d{7}|5\d{8})$",  # Georgia
    "DE": r"^((\+49|0)1)(5[0-25-9]\d|6([23]|0\d?)|7([0-57-9]|6\d))\d{7,9}$",  # Germany
    "GH": r"^(\+233|0)(20|50|24|54|27|57|26|56|23|28|55|59)\d{7}$",  # Ghana
    "GR": r"^(\+?30|0)?6(8[5-9]|9(?![26])[0-9])\d{7}$",  # Greece
    "GL": r"^(\+?299)?\s?\d{2}\s?\d{2}\s?\d{2}$",  # Greenland
    "GP": r"^(\+?590|0|00590)[67]\d{8}$",  # Guadeloupe
    "GY": r"^(\+592|0)6\d{6}$",  # Guyana
    "HN": r"^(\+?504)?[9|8|3|2]\d{7}$",  # Honduras
    "HK": r"^(\+?852[-\s]?)?[456789]\d{3}[-\s]?\d{4}$",  # Hong Kong
    "HU": r"^(\+?36|06)(20|30|31|50|70)\d{7}$",  # Hungary
    "IS": r"^\+354[0-9]{7}$",  # Iceland
    "IN": r"^(\+?91|0)?[6789]\d{9}$",  # India
    "ID": r"^(\+?62|0)8(1[123456789]|2[1238]|3[1238]|5[12356789]|7[78]|9[56789]|8[123456789])([\s?|\d]{5,11})$",  # Indonesia
    "IR": r"^(\+98|0)?9\d{9}$",  # Iran
    "IQ": r"^(\+?964|0)?7[0-9]\d{8}$",  # Iraq
    "IE": r"^(\+?353|0)8[356789]\d{7}$",  # Ireland
    "IT": r"^(\+?39)?\s?3\d{2} ?\d{6,7}$",  # Italy
    "JM": r"^(\+?876)?\d{7}$",  # Jamaica
    "JP": r"^(\+81[ \-]?(\(0\))?|0)[6789]0[ \-]?\d{4}[ \-]?\d{4}$",  # Japan
    "JO": r"^(\+?962|0)?7[789]\d{7}$",  # Jordan
    "KZ": r"^(\+?7|8)?7\d{9}$",  # Kazakhstan
    "KE": r"^(\+?254|0)(7|1)\d{8}$",  # Kenya
    "KR": r"^((\+?82)[ \-]?)?0?1([0|1|6|7|8|9]{1})[ \-]?\d{3,4}[ \-]?\d{4}$",  # South Korea
    "XK": r"^\+383[1-9][0-9]{6,7}$",  # Kosovo
    "KW": r"^(\+?965)([569]\d{7}|41\d{6})$",  # Kuwait
    "KG": r"^(\+?7\s?\+?7|0)\s?\d{2}\s?\d{3}\s?\d{4}$",  # Kyrgyzstan
    "LV": r"^(\+?371)2\d{7}$",  # Latvia
    "LB": r"^(\+?961)?((3|81)\d{6}|7\d{7})$",  # Lebanon
    "LS": r"^(\+?266)(22|28|57|58|59|27|52)\d{6}$",  # Lesotho
    "LY": r"^((\+?218)|0)?(9[1-6]\d{7}|[1-8]\d{7,9})$",  # Libya
    "LI": r"^\+423[0-9]{3,12}$",  # Liechtenstein
    "LT": r"^(\+370|0)\d{8}$",  # Lithuania
    "LU": r"^(\+352)?((6\d1)\d{6})$",  # Luxembourg
    "MO": r"^(\+?853[-\s]?)?[6]\d{3}[-\s]?\d{4}$",  # Macao
    "MG": r"^((\+?261|0)(2|3)\d)?\d{7}$",  # Madagascar
    "MW": r"^(\+?265|0)(((77|88|31|99|98|21)\d{7})|((111|1)\d{6})|(32000\d{4}))$",  # Malawi
    "MY": r"^(\+?60|0)1(([0145](-|\s)?\d{7,8})|([236-9](-|\s)?\d{7}))$",  # Malaysia
    "MV": r"^(\+?960)?(7[2-9]|9[1-9])\d{5}$",  # Maldives
    "MT": r"^(\+?356|0)?(99|79|77|21|27|22|25)[0-9]{6}$",  # Malta
    "MQ": r"^(\+?596|0|00596)[67]\d{8}$",  # Martinique
    "MU": r"^(\+?230|0)?\d{8}$",  # Mauritius
    "MX": r"^(\+?52)?(1|01)?\d{10,11}$",  # Mexico
    "MD": r"^(\+?373|0)((6(0|1|2|6|7|8|9))|(7(6|7|8|9)))\d{6}$",  # Moldova
    "MC": r"^\+377[0-9]{8,9}$",  # Monaco
    "MN": r"^(\+|00|011)?976(77|81|88|91|94|95|96|99)\d{6}$",  # Mongolia
    "ME": r"^\+382[6-9][0-9]{6,7}$",  # Montenegro
    "MA": r"^(?:(?:\+|00)212|0)[5-7]\d{8}$",  # Morocco
    "MZ": r"^(\+?258)?8[234567]\d{7}$",  # Mozambique
    "MM": r"^(\+?959|09|9)(2[5-7]|3[1-2]|4[0-5]|6[6-9]|7[5-9]|9[6-9])[0-9]{7}$",  # Myanmar
    "NA": r"^(\+?264|0)(6|8)\d{7}$",  # Namibia
    "NP": r"^(\+?977)?9[78]\d{8}$",  # Nepal
    "NL": r"^(((\+|00)?31\(0\))|((\+|00)?31)|0)6{1}\d{8}$",  # Netherlands
    "NZ": r"^(\+?64|0)[28]\d{7,9}$",  # New Zealand
    "NI": r"^(\+?505)\d{7,8}$",  # Nicaragua
    "NG": r"^(\+?234|0)?[789]\d{9}$",  # Nigeria
    "MK": r"^\+389[2-9][0-9]{6,7}$",  # North Macedonia
    "NO": r"^(\+?47)?[49]\d{7}$",  # Norway
    "OM": r"^((\+|00)968)?(9[1-9])\d{6}$",  # Oman
    "PK": r"^((00|\+)?92|0)3[0-6]\d{8}$",  # Pakistan
    "PA": r"^(\+?507)\d{7,8}$",  # Panama
    "PG": r"^(\+?675|0)?(7\d|8[18])\d{6}$",  # Papua New Guinea
    "PY": r"^(\+?595|0)9[9876]\d{7}$",  # Paraguay
    "PE": r"^(\+?51)?9\d{8}$",  # Peru
    "PH": r"^(09|\+639)\d{9}$",  # Philippines
    "PL": r"^(\+?48)? ?([5-8]\d|45) ?\d{3} ?\d{2} ?\d{2}$",  # Poland
    "PT": r"^(\+?351)?9[1236]\d{7}$",  # Portugal
    "QA": r"^(\+974)?[3567]\d{7}$",  # Qatar
    "RE": r"^(\+?262|0|00262)[67]\d{8}$",  # Réunion
    "RO": r"^(\+?40|0)\s?7\d{2}(\/|\s|\.|\-)?\d{3}(\s|\.|\-)?\d{3}$",  # Romania
    "RU": r"^(\+?7|8)?9\d{9}$",  # Russia
    "RW": r"^(\+?250|0)?[7]\d{8}$",  # Rwanda
    "SM": r"^((\+378)|(0549)|(\+390549)|(\+3780549))?6\d{5,9}$",  # San Marino
    "SA": r"^(!?(\+?966)|0)?5\d{8}$",  # Saudi Arabia
    "RS": r"^(\+3816|06)[- \d]{5,9}$",  # Serbia
    "SL": r"^(\+?232|0)\d{8}$",  # Sierra Leone
    "SG": r"^(\+65)?[3689]\d{7}$",  # Singapore
    "SK": r"^(\+?421)? ?[1-9][0-9]{2} ?[0-9]{3} ?[0-9]{3}$",  # Slovakia
    "SI": r"^(\+386\s?|0)(\d{1}\s?\d{3}\s?\d{2}\s?\d{2}|\d{2}\s?\d{3}\s?\d{3})$",  # Slovenia
    "SO": r"^(\+?252|0)((6[0-9])\d{7}|(7[1-9])\d{7})$",  # Somalia
    "ZA": r"^(\+?27|0)\d{9}$",  # South Africa
    "SS": r"^(\+?211|0)(9[1257])\d{7}$",  # South Sudan
    "ES": r"^(\+?34)?[67]\d{8}$",  # Spain
    "LK": r"^(?:0|94|\+94)?(7(0|1|2|4|5|6|7|8)( |-)?)?\d{7}$",  # Sri Lanka
    "SD": r"^((\+?249)|0)?(9[012369]|1[012])\d{7}$",  # Sudan
    "SE": r"^(\+?46|0)[\s\-]?7[\s\-]?[02369]([\s\-]?\d){7}$",  # Sweden
    "CH": r"^(\+41|0)([1-9])\d{1,9}$",  # Switzerland
    "SY": r"^(!?(\+?963)|0)?9\d{8}$",  # Syria
    "TW": r"^(\+?886\-?|0)?9\d{8}$",  # Taiwan
    "TJ": r"^(\+?992)?[5][5]\d{7}$",  # Tajikistan
    "TZ": r"^(\+?255|0)?[67]\d{8}$",  # Tanzania
    "TH": r"^(\+66|66|0)\d{9}$",  # Thailand
    "TN": r"^(\+?216)?[2459]\d{7}$",  # Tunisia
    "TR": r"^(\+?90|0)?5\d{9}$",  # Turkey
    "TM": r"^(\+993|993|8)\d{8}$",  # Turkmenistan
    "UG": r"^(\+?256|0)?[7]\d{8}$",  # Uganda
    "UA": r"^(\+?38|8)?0\d{9}$",  # Ukraine
    "AE": r"^((\+?971)|0)?5[024568]\d{7}$",  # United Arab Emirates
    "GB": r"^(\+?44|0)7\d{9}$",  # United Kingdom
    "US": r"^(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$",  # United States
    "UY": r"^(\+598|0)9[1-9][\d]{6}$",  # Uruguay
    "UZ": r"^(\+?998)?(6[125-79]|7[1-69]|88|9\d)\d{7}$",  # Uzbekistan
    "VE": r"^(\+?58)?(2|4)\d{9}$",  # Venezuela
    "VN": r"^((\+?84)|0)((3([2-9]))|(5([25689]))|(7([0|6-9]))|(8([1-9]))|(9([0-9])))([0-9]{7})$",  # Vietnam
    "YE": r"^(((\+|00)9677|0?7)[0137]\d{7}|((\+|00)967|0)[1-7]\d{6})$",  # Yemen
    "ZM": r"^(\+?26)?09[567]\d{7}$",  # Zambia
    "ZW": r"^(\+263)[0-9]{9}$",  # Zimbabwe
}


# =============================================================================
# CHECKSUM ALGORITHMS
# =============================================================================

def luhn_checksum(number: str) -> int:
    """
    Calculate Luhn checksum digit.
    Used for credit cards, IMEI, some national IDs.
    """
    digits = [int(d) for d in number if d.isdigit()]
    odd_sum = sum(digits[-1::-2])
    even_sum = sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
    return (odd_sum + even_sum) % 10


def luhn_validate(number: str) -> bool:
    """Validate a number using Luhn algorithm."""
    return luhn_checksum(number) == 0


# Verhoeff algorithm tables
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum(number: str) -> int:
    """
    Calculate Verhoeff checksum digit.
    Used for Indian Aadhaar and some other IDs.
    More robust than Luhn - catches all single-digit and transposition errors.
    """
    c = 0
    for i, digit in enumerate(reversed(number)):
        if digit.isdigit():
            c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][int(digit)]]
    return VERHOEFF_INV[c]


def verhoeff_validate(number: str) -> bool:
    """Validate a number using Verhoeff algorithm."""
    c = 0
    for i, digit in enumerate(reversed(number)):
        if digit.isdigit():
            c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def mod11_checksum(number: str, weights: Optional[List[int]] = None) -> int:
    """
    Calculate Mod-11 checksum.
    Used for ISBNs, some national IDs (Norway, Denmark).

    Args:
        number: The number to calculate checksum for
        weights: Custom weights for each position (default: descending from length)
    """
    digits = [int(d) for d in number if d.isdigit()]

    if weights is None:
        # Default weights: descending from length
        weights = list(range(len(digits) + 1, 1, -1))

    total = sum(d * w for d, w in zip(digits, weights))
    remainder = total % 11

    return (11 - remainder) % 11


def mod11_validate(number: str, weights: Optional[List[int]] = None) -> bool:
    """
    Validate a number using Mod-11 algorithm.
    The last digit should equal the checksum of preceding digits.
    """
    if not number or len(number) < 2:
        return False

    check_digit = int(number[-1]) if number[-1].isdigit() else (10 if number[-1].upper() == 'X' else -1)
    if check_digit < 0:
        return False

    expected = mod11_checksum(number[:-1], weights)
    return check_digit == expected


def mod97_validate(number: str) -> bool:
    """
    Validate using ISO 13616 Mod-97 algorithm.
    Used for IBAN validation.
    """
    # Move first 4 chars to end
    rearranged = number[4:] + number[:4]

    # Convert letters to numbers (A=10, B=11, etc.)
    numeric = ""
    for char in rearranged.upper():
        if char.isalpha():
            numeric += str(ord(char) - 55)
        else:
            numeric += char

    return int(numeric) % 97 == 1


# =============================================================================
# IBAN VALIDATION
# =============================================================================

def validate_iban(iban: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate an IBAN using the python-stdnum library.

    Args:
        iban: The IBAN string to validate (with or without spaces)

    Returns:
        Tuple of (is_valid, country_code, metadata)
        - is_valid: True if the IBAN passes checksum validation
        - country_code: Two-letter country code (e.g., "DE", "GB")
        - metadata: Dict with bank info if available
    """
    try:
        from stdnum import iban as stdnum_iban
        from stdnum.exceptions import InvalidChecksum, InvalidFormat, InvalidLength

        # Clean the IBAN
        clean_iban = iban.replace(" ", "").replace("-", "").upper()

        # Validate
        if stdnum_iban.is_valid(clean_iban):
            country = clean_iban[:2]
            return True, country, {
                "compact": stdnum_iban.compact(clean_iban),
                "formatted": stdnum_iban.format(clean_iban),
            }
        return False, None, None

    except ImportError:
        # Fallback: basic validation without library
        return _basic_iban_validate(iban)
    except Exception:
        return False, None, None


def _basic_iban_validate(iban: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Basic IBAN validation using mod-97 algorithm without external library."""
    clean = iban.replace(" ", "").replace("-", "").upper()

    # Check basic format
    if len(clean) < 15 or len(clean) > 34:
        return False, None, None

    if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', clean):
        return False, None, None

    # ISO 13616 mod-97 validation
    country = clean[:2]
    rearranged = clean[4:] + clean[:4]

    # Convert letters to numbers (A=10, B=11, etc.)
    numeric = ""
    for char in rearranged:
        if char.isalpha():
            numeric += str(ord(char) - 55)
        else:
            numeric += char

    # Check if mod 97 equals 1
    if int(numeric) % 97 == 1:
        return True, country, {"compact": clean}

    return False, None, None


# =============================================================================
# BIC/SWIFT VALIDATION
# =============================================================================

def validate_bic(bic: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a BIC/SWIFT code using python-stdnum.

    Args:
        bic: The BIC/SWIFT code (8 or 11 characters)

    Returns:
        Tuple of (is_valid, country_code, metadata)
    """
    try:
        from stdnum import bic as stdnum_bic

        clean_bic = bic.replace(" ", "").upper()

        if stdnum_bic.is_valid(clean_bic):
            country = clean_bic[4:6]
            return True, country, {
                "compact": stdnum_bic.compact(clean_bic),
                "bank_code": clean_bic[:4],
                "country": country,
                "location": clean_bic[6:8],
                "branch": clean_bic[8:] if len(clean_bic) > 8 else "XXX",
            }
        return False, None, None

    except ImportError:
        # Fallback: basic format check
        clean = bic.replace(" ", "").upper()
        if re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', clean):
            return True, clean[4:6], {"compact": clean}
        return False, None, None
    except Exception:
        return False, None, None


# =============================================================================
# PHONE NUMBER VALIDATION
# =============================================================================

def validate_phone(
    phone: str,
    default_region: str = "US"
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a phone number using the phonenumbers library.

    Args:
        phone: The phone number string
        default_region: Default country code if not specified (e.g., "US", "GB")

    Returns:
        Tuple of (is_valid, country_code, metadata)
        - is_valid: True if the number is a valid phone number
        - country_code: Two-letter country code
        - metadata: Dict with formatted number, type, etc.
    """
    try:
        import phonenumbers
        from phonenumbers import NumberParseException, PhoneNumberType

        # Parse the phone number
        try:
            parsed = phonenumbers.parse(phone, default_region)
        except NumberParseException:
            return False, None, None

        # Validate
        if not phonenumbers.is_valid_number(parsed):
            return False, None, None

        # Get country code
        country = phonenumbers.region_code_for_number(parsed)

        # Get number type
        num_type = phonenumbers.number_type(parsed)
        type_names = {
            PhoneNumberType.MOBILE: "mobile",
            PhoneNumberType.FIXED_LINE: "fixed_line",
            PhoneNumberType.FIXED_LINE_OR_MOBILE: "fixed_or_mobile",
            PhoneNumberType.TOLL_FREE: "toll_free",
            PhoneNumberType.PREMIUM_RATE: "premium_rate",
            PhoneNumberType.VOIP: "voip",
            PhoneNumberType.PERSONAL_NUMBER: "personal",
            PhoneNumberType.PAGER: "pager",
        }

        return True, country, {
            "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "international": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
            "national": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
            "type": type_names.get(num_type, "unknown"),
            "country_code": parsed.country_code,
        }

    except ImportError:
        # Fallback: basic format check
        digits = re.sub(r'\D', '', phone)
        if 7 <= len(digits) <= 15:
            return True, None, {"digits": digits}
        return False, None, None
    except Exception:
        return False, None, None


# =============================================================================
# CREDIT CARD VALIDATION (LUHN ALGORITHM)
# =============================================================================

def validate_credit_card(card_number: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a credit card number using the Luhn algorithm.

    Args:
        card_number: The credit card number (with or without spaces/dashes)

    Returns:
        Tuple of (is_valid, card_type, metadata)
        - is_valid: True if passes Luhn checksum
        - card_type: Card type (visa, mastercard, amex, discover, etc.)
        - metadata: Dict with additional info
    """
    try:
        from stdnum import luhn

        # Clean the card number
        clean = re.sub(r'[\s\-]', '', card_number)

        if not clean.isdigit():
            return False, None, None

        # Validate with Luhn
        if not luhn.is_valid(clean):
            return False, None, None

        # Determine card type
        card_type = _get_card_type(clean)

        return True, card_type, {
            "digits": len(clean),
            "masked": f"**** **** **** {clean[-4:]}",
            "issuer": card_type,
        }

    except ImportError:
        # Fallback: manual Luhn implementation
        return _manual_luhn_validate(card_number)
    except Exception:
        return False, None, None


def _get_card_type(card_number: str) -> str:
    """Determine credit card type from number."""
    if card_number.startswith('4'):
        return "visa"
    elif card_number[:2] in ('51', '52', '53', '54', '55') or \
         (2221 <= int(card_number[:4]) <= 2720):
        return "mastercard"
    elif card_number[:2] in ('34', '37'):
        return "amex"
    elif card_number.startswith('6011') or \
         card_number[:3] in ('644', '645', '646', '647', '648', '649') or \
         card_number.startswith('65') or \
         (622126 <= int(card_number[:6]) <= 622925):
        return "discover"
    elif card_number[:4] in ('3528', '3529') or \
         (3530 <= int(card_number[:4]) <= 3589):
        return "jcb"
    elif card_number[:4] in ('6304', '6706', '6771', '6709'):
        return "maestro"
    elif card_number[:2] in ('36', '38') or card_number[:3] in ('300', '301', '302', '303', '304', '305'):
        return "diners"
    else:
        return "unknown"


def _manual_luhn_validate(card_number: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Manual Luhn algorithm implementation."""
    clean = re.sub(r'[\s\-]', '', card_number)

    if not clean.isdigit() or len(clean) < 13 or len(clean) > 19:
        return False, None, None

    # Luhn algorithm
    total = 0
    for i, digit in enumerate(reversed(clean)):
        d = int(digit)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    if total % 10 == 0:
        card_type = _get_card_type(clean)
        return True, card_type, {"digits": len(clean), "masked": f"**** {clean[-4:]}"}

    return False, None, None


# =============================================================================
# NATIONAL ID VALIDATION
# =============================================================================

# Mapping of country codes to stdnum module names
NATIONAL_ID_MODULES = {
    # Europe
    "DE": "de.idnr",      # German tax ID (Steuer-ID)
    "FR": "fr.nir",       # French INSEE/NIR
    "GB": "gb.nino",      # UK National Insurance Number
    "IT": "it.codicefiscale",  # Italian Codice Fiscale
    "ES": "es.dni",       # Spanish DNI
    "NL": "nl.bsn",       # Dutch BSN
    "BE": "be.nn",        # Belgian National Number
    "AT": "at.vnr",       # Austrian Versicherungsnummer
    "PL": "pl.pesel",     # Polish PESEL
    "SE": "se.personnummer",  # Swedish Personnummer
    "DK": "dk.cpr",       # Danish CPR
    "NO": "no.fodselsnummer",  # Norwegian Fødselsnummer
    "FI": "fi.hetu",      # Finnish HETU
    "CH": "ch.ssn",       # Swiss SSN
    "PT": "pt.nif",       # Portuguese NIF
    "IE": "ie.pps",       # Irish PPS

    # Americas
    "US": "us.ssn",       # US SSN
    "CA": "ca.sin",       # Canadian SIN
    "BR": "br.cpf",       # Brazilian CPF
    "AR": "ar.cuit",      # Argentine CUIT
    "CL": "cl.rut",       # Chilean RUT
    "MX": "mx.curp",      # Mexican CURP
    "CO": "co.nit",       # Colombian NIT

    # Asia-Pacific
    "AU": "au.tfn",       # Australian TFN
    "CN": "cn.ric",       # Chinese Resident ID
    "IN": "in.aadhaar",   # Indian Aadhaar (note: stdnum has basic support)
    "JP": "jp.cn",        # Japanese Corporate Number
    "KR": "kr.rrn",       # South Korean RRN
    "SG": "sg.nric",      # Singapore NRIC
    "MY": "my.nric",      # Malaysian NRIC
    "ID": "id.npwp",      # Indonesian NPWP

    # Middle East
    "IL": "il.idnr",      # Israeli ID
    "TR": "tr.tckimlik",  # Turkish TC Kimlik

    # Africa
    "ZA": "za.idnr",      # South African ID (custom implementation)
}


def validate_south_african_id(id_number: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate a South African ID number.

    Format: YYMMDD SSSS CAZ (13 digits)
    - YYMMDD: Date of birth
    - SSSS: Sequential number (0000-4999 female, 5000-9999 male)
    - C: Citizenship (0=SA citizen, 1=permanent resident, 2=refugee)
    - A: Random digit (formerly racial identifier)
    - Z: Luhn check digit

    Args:
        id_number: The 13-digit ID number

    Returns:
        Tuple of (is_valid, metadata)
    """
    clean = re.sub(r'[\s\-]', '', id_number)

    if len(clean) != 13 or not clean.isdigit():
        return False, None

    # Validate with Luhn algorithm
    try:
        from stdnum import luhn
        if not luhn.is_valid(clean):
            return False, None
    except ImportError:
        # Manual Luhn validation
        if not _manual_luhn_check(clean):
            return False, None

    # Extract components
    year = int(clean[0:2])
    month = int(clean[2:4])
    day = int(clean[4:6])
    sequence = int(clean[6:10])
    citizenship = int(clean[10])

    # Validate date components
    if month < 1 or month > 12:
        return False, None
    if day < 1 or day > 31:
        return False, None

    # Determine century (assume 1900s for > current year, 2000s otherwise)
    import datetime
    current_year = datetime.datetime.now().year % 100
    century = 1900 if year > current_year else 2000
    full_year = century + year

    # Determine gender
    gender = "female" if sequence < 5000 else "male"

    # Citizenship status
    citizenship_status = {
        0: "SA citizen",
        1: "permanent resident",
        2: "refugee"
    }.get(citizenship, "unknown")

    return True, {
        "date_of_birth": f"{full_year}-{month:02d}-{day:02d}",
        "gender": gender,
        "citizenship": citizenship_status,
        "formatted": f"{clean[:6]} {clean[6:10]} {clean[10:13]}",
    }


def _manual_luhn_check(number: str) -> bool:
    """Manual Luhn algorithm check for when stdnum is not available."""
    total = 0
    for i, digit in enumerate(reversed(number)):
        d = int(digit)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validate_national_id(
    id_number: str,
    country_code: str
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a national ID number using python-stdnum.

    Args:
        id_number: The ID number to validate
        country_code: Two-letter ISO country code (e.g., "US", "DE", "BR", "ZA")

    Returns:
        Tuple of (is_valid, country_code, metadata)
    """
    country_code = country_code.upper()

    # Special handling for South African IDs (custom implementation)
    if country_code == "ZA":
        is_valid, metadata = validate_south_african_id(id_number)
        return is_valid, "ZA" if is_valid else None, metadata

    if country_code not in NATIONAL_ID_MODULES:
        # Country not supported - return basic format check
        clean = re.sub(r'[\s\-\.]', '', id_number)
        if len(clean) >= 6:
            return True, country_code, {"format": "unknown", "validated": False}
        return False, None, None

    try:
        import importlib

        module_name = NATIONAL_ID_MODULES[country_code]
        module = importlib.import_module(f"stdnum.{module_name}")

        # Clean and validate
        clean = id_number.replace(" ", "").replace("-", "").replace(".", "")

        if hasattr(module, 'is_valid') and module.is_valid(clean):
            metadata = {"validated": True}

            # Get compact/formatted versions if available
            if hasattr(module, 'compact'):
                metadata["compact"] = module.compact(clean)
            if hasattr(module, 'format'):
                try:
                    metadata["formatted"] = module.format(clean)
                except Exception:
                    pass

            return True, country_code, metadata

        return False, None, None

    except ImportError:
        # Module not available
        return False, None, {"error": "stdnum not installed"}
    except Exception as e:
        return False, None, {"error": str(e)}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_detected_entity(
    entity_type: str,
    text: str,
    locale: Optional[str] = None
) -> Tuple[bool, float]:
    """
    Validate a detected PII entity and return adjusted confidence.

    Args:
        entity_type: The detected entity type (e.g., "IBAN", "PHONE_NUMBER")
        text: The detected text
        locale: Optional locale for context (e.g., "en-US", "de-DE")

    Returns:
        Tuple of (is_valid, confidence_adjustment)
        - is_valid: True if the entity passes validation
        - confidence_adjustment: Suggested confidence adjustment (-0.3 to +0.2)
    """
    # Extract country code from locale if provided
    country = None
    if locale and '-' in locale:
        country = locale.split('-')[1].upper()

    if entity_type == "IBAN":
        valid, _, _ = validate_iban(text)
        return valid, 0.15 if valid else -0.3

    elif entity_type == "IBAN_CODE":
        valid, _, _ = validate_iban(text)
        return valid, 0.15 if valid else -0.3

    elif entity_type in ("BIC", "SWIFT", "SWIFT_CODE"):
        valid, _, _ = validate_bic(text)
        return valid, 0.1 if valid else -0.2

    elif entity_type == "PHONE_NUMBER":
        valid, _, _ = validate_phone(text, default_region=country or "US")
        return valid, 0.1 if valid else -0.15

    elif entity_type == "CREDIT_CARD":
        valid, _, _ = validate_credit_card(text)
        return valid, 0.15 if valid else -0.3

    elif entity_type in ("NATIONAL_ID", "SSN", "NIF", "CPF", "PESEL"):
        if country:
            valid, _, _ = validate_national_id(text, country)
            return valid, 0.15 if valid else -0.2

    # Unknown entity type or no validation available
    return True, 0.0


def get_supported_countries() -> Dict[str, str]:
    """
    Get list of countries with national ID validation support.

    Returns:
        Dict mapping country code to ID type name
    """
    return {
        # Europe
        "DE": "Steuer-ID",
        "FR": "INSEE/NIR",
        "GB": "National Insurance Number",
        "IT": "Codice Fiscale",
        "ES": "DNI",
        "NL": "BSN",
        "BE": "National Number",
        "AT": "Versicherungsnummer",
        "PL": "PESEL",
        "SE": "Personnummer",
        "DK": "CPR",
        "NO": "Fødselsnummer",
        "FI": "HETU",
        "CH": "SSN",
        "PT": "NIF",
        "IE": "PPS",
        # Americas
        "US": "SSN",
        "CA": "SIN",
        "BR": "CPF",
        "AR": "CUIT",
        "CL": "RUT",
        "MX": "CURP",
        "CO": "NIT",
        # Asia-Pacific
        "AU": "TFN",
        "CN": "Resident ID",
        "IN": "Aadhaar",
        "KR": "RRN",
        "SG": "NRIC",
        "MY": "NRIC",
        "ID": "NPWP",
        "JP": "Corporate Number",
        # Middle East
        "IL": "ID Number",
        "TR": "TC Kimlik",
        # Africa
        "ZA": "South African ID",  # YYMMDD SSSS CAZ format with Luhn checksum
    }
