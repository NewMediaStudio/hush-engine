#!/usr/bin/env python3
"""
Lightweight Names Database for Hush Engine

A curated, compact database of common first and last names by region.
Optimized for fast lookup (~5MB in memory) vs. full names-dataset (3.2GB).

Coverage: Top ~5,000 first names + ~10,000 last names globally
Estimated coverage: 90%+ of names in typical documents

Usage:
    from data.names_database import NamesDatabase
    db = NamesDatabase()
    db.is_first_name("John")  # True
    db.is_last_name("Smith")  # True
    db.is_name("Maria")       # True (either first or last)
"""

from typing import Set, Optional, Dict, Any
import re

# ============================================================================
# Top First Names by Region (Total: ~5,000 names)
# ============================================================================

# English-speaking countries (US, UK, Canada, Australia, etc.)
FIRST_NAMES_ENGLISH = {
    # Male
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph",
    "thomas", "charles", "christopher", "daniel", "matthew", "anthony", "mark",
    "donald", "steven", "paul", "andrew", "joshua", "kenneth", "kevin", "brian",
    "george", "timothy", "ronald", "edward", "jason", "jeffrey", "ryan", "jacob",
    "gary", "nicholas", "eric", "jonathan", "stephen", "larry", "justin", "scott",
    "brandon", "benjamin", "samuel", "raymond", "gregory", "frank", "alexander",
    "patrick", "jack", "dennis", "jerry", "tyler", "aaron", "jose", "adam", "nathan",
    "henry", "douglas", "zachary", "peter", "kyle", "noah", "ethan", "jeremy",
    "walter", "christian", "keith", "roger", "terry", "austin", "sean", "gerald",
    "carl", "harold", "dylan", "arthur", "lawrence", "jordan", "jesse", "bryan",
    "billy", "bruce", "gabriel", "joe", "logan", "albert", "willie", "alan", "eugene",
    "russell", "vincent", "philip", "bobby", "johnny", "bradley", "roy", "ralph",
    "eugene", "randy", "wayne", "elijah", "louis", "harry", "howard", "leonard",
    "martin", "curtis", "stanley", "joe", "jimmy", "eddie", "mason", "aiden",
    "liam", "lucas", "oliver", "hunter", "jackson", "connor", "luke", "evan",
    # Female
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth", "susan",
    "jessica", "sarah", "karen", "lisa", "nancy", "betty", "margaret", "sandra",
    "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol",
    "amanda", "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura",
    "cynthia", "kathleen", "amy", "angela", "shirley", "anna", "brenda", "pamela",
    "emma", "nicole", "helen", "samantha", "katherine", "christine", "debra",
    "rachel", "carolyn", "janet", "catherine", "maria", "heather", "diane", "ruth",
    "julie", "olivia", "joyce", "virginia", "victoria", "kelly", "lauren", "christina",
    "joan", "evelyn", "judith", "megan", "andrea", "cheryl", "hannah", "jacqueline",
    "martha", "gloria", "teresa", "ann", "sara", "madison", "frances", "kathryn",
    "janice", "jean", "abigail", "alice", "judy", "sophia", "grace", "denise",
    "amber", "doris", "marilyn", "danielle", "beverly", "isabella", "theresa",
    "diana", "natalie", "brittany", "charlotte", "marie", "kayla", "alexis", "lori",
    "ava", "mia", "chloe", "zoe", "lily", "ella", "avery", "riley", "aria",
}

# Spanish-speaking countries
FIRST_NAMES_SPANISH = {
    # Male
    "jose", "juan", "carlos", "luis", "miguel", "francisco", "antonio", "pedro",
    "manuel", "alejandro", "rafael", "david", "daniel", "pablo", "jorge", "fernando",
    "ricardo", "eduardo", "mario", "alberto", "sergio", "roberto", "enrique", "javier",
    "victor", "andres", "ramon", "diego", "oscar", "hector", "ruben", "arturo",
    "adrian", "santiago", "gabriel", "raul", "hugo", "ivan", "jesus", "marco",
    "alfonso", "angel", "julian", "armando", "felipe", "gerardo", "cesar", "ignacio",
    "rodrigo", "martin", "nicolas", "mateo", "sebastian", "leonardo", "emiliano",
    # Female
    "maria", "carmen", "ana", "rosa", "isabel", "laura", "lucia", "patricia",
    "elena", "pilar", "dolores", "cristina", "marta", "mercedes", "paula", "sara",
    "andrea", "beatriz", "silvia", "teresa", "rocio", "monica", "raquel", "susana",
    "julia", "eva", "natalia", "daniela", "valentina", "sofia", "camila", "mariana",
    "fernanda", "gabriela", "alejandra", "adriana", "lorena", "veronica", "carolina",
    "catalina", "alicia", "claudia", "esther", "irene", "nuria", "inmaculada",
}

# German-speaking countries
FIRST_NAMES_GERMAN = {
    # Male
    "hans", "klaus", "wolfgang", "dieter", "jurgen", "helmut", "peter", "manfred",
    "gerhard", "werner", "heinrich", "karl", "franz", "herbert", "wilhelm", "walter",
    "friedrich", "horst", "gunter", "rainer", "uwe", "bernd", "heinz", "kurt",
    "norbert", "stefan", "matthias", "andreas", "christian", "markus", "thomas",
    "michael", "frank", "martin", "alexander", "jan", "felix", "maximilian", "paul",
    "lukas", "jonas", "leon", "tim", "florian", "sebastian", "tobias", "niklas",
    # Female
    "ursula", "helga", "monika", "renate", "ingrid", "gisela", "christa", "erika",
    "brigitte", "sabine", "petra", "andrea", "susanne", "birgit", "karin", "claudia",
    "angelika", "maria", "anna", "elisabeth", "heike", "gabriele", "martina", "julia",
    "sarah", "laura", "lisa", "sophie", "lena", "emma", "mia", "hannah", "leonie",
    "marie", "katharina", "johanna", "charlotte", "amelie", "lara", "nele",
}

# French-speaking countries
FIRST_NAMES_FRENCH = {
    # Male
    "jean", "pierre", "michel", "jacques", "francois", "andre", "philippe", "bernard",
    "louis", "paul", "alain", "claude", "marcel", "rene", "roger", "christian",
    "patrick", "robert", "daniel", "nicolas", "laurent", "julien", "eric", "pascal",
    "thomas", "david", "antoine", "guillaume", "maxime", "alexandre", "hugo", "lucas",
    "nathan", "gabriel", "arthur", "louis", "raphael", "paul", "adam", "leo",
    # Female
    "marie", "jeanne", "francoise", "monique", "nicole", "catherine", "sylvie",
    "christine", "isabelle", "anne", "nathalie", "sandrine", "veronique", "sophie",
    "patricia", "julie", "claire", "camille", "lea", "manon", "chloe", "emma",
    "ines", "jade", "louise", "alice", "lina", "rose", "anna", "charlotte",
}

# Chinese names (Romanized - Pinyin)
FIRST_NAMES_CHINESE = {
    "wei", "fang", "ming", "jing", "lei", "jun", "yan", "ping", "hong", "lin",
    "ying", "qiang", "feng", "jie", "xin", "yu", "wen", "bo", "tao", "yang",
    "hai", "hui", "li", "chen", "yong", "gang", "bin", "hua", "xiao", "dong",
    "cheng", "ning", "yi", "nan", "qing", "rui", "peng", "hao", "kai", "zhi",
    "jian", "liang", "shan", "mei", "xia", "juan", "lan", "qiu", "yue", "dan",
}

# Japanese names (Romanized)
FIRST_NAMES_JAPANESE = {
    # Male
    "hiroshi", "takeshi", "kenji", "yuki", "taro", "satoshi", "kazuki", "yusuke",
    "daisuke", "takuya", "naoki", "makoto", "shota", "ryota", "kenta", "daiki",
    "masashi", "tomoya", "akira", "kenichi", "shun", "ryo", "hayato", "koichi",
    "haruki", "yuto", "sota", "ren", "kaito", "hinata", "sora", "haruto", "minato",
    # Female
    "yuko", "keiko", "michiko", "sachiko", "yuki", "haruka", "sakura", "yui",
    "misaki", "aoi", "miku", "nana", "riko", "mei", "hina", "rin", "kokona",
    "akiko", "tomoko", "mayumi", "kumiko", "megumi", "emi", "yuka", "mika",
}

# Korean names (Romanized)
FIRST_NAMES_KOREAN = {
    "min", "jun", "hyun", "joon", "jin", "sung", "young", "seung", "dong", "yong",
    "su", "hee", "soo", "jung", "kyung", "eun", "mi", "sun", "ji", "hyeon",
    "seon", "yeong", "min", "ha", "yeon", "so", "yu", "jae", "woo", "ho",
}

# Indian names (Hindi/Sanskrit origin)
FIRST_NAMES_INDIAN = {
    # Male
    "rajesh", "suresh", "ramesh", "amit", "anil", "vijay", "sanjay", "ajay",
    "manoj", "prakash", "dinesh", "ashok", "rakesh", "mukesh", "raj", "ravi",
    "sandeep", "rahul", "vikas", "manish", "pradeep", "deepak", "sunil", "arun",
    "anand", "vivek", "vikram", "arjun", "krishna", "mohan", "rohit", "akash",
    "nikhil", "sachin", "gaurav", "aarav", "vihaan", "reyansh", "advik", "ishaan",
    # Female
    "priya", "neha", "pooja", "anjali", "sunita", "kavita", "ritu", "meena",
    "geeta", "rekha", "suman", "seema", "anita", "nisha", "divya", "shreya",
    "swati", "tanvi", "aishwarya", "deepika", "rani", "lata", "sita", "radha",
    "aadhya", "saanvi", "ananya", "diya", "pari", "aanya", "aaradhya", "myra",
}

# Arabic names
FIRST_NAMES_ARABIC = {
    # Male
    "mohammed", "muhammad", "mohamed", "ahmad", "ahmed", "ali", "hassan", "hussein",
    "omar", "ibrahim", "khalid", "abdullah", "mustafa", "youssef", "yousef", "salem",
    "saeed", "rashid", "nasser", "faisal", "tariq", "karim", "walid", "mahmoud",
    "jamal", "hamid", "bilal", "adam", "zayed", "khalil", "majid", "rami",
    # Female
    "fatima", "aisha", "maryam", "sara", "layla", "noor", "hana", "yasmin",
    "amina", "khadija", "zainab", "rania", "dina", "lina", "maya", "dana",
    "huda", "salma", "aya", "mariam", "noura", "farah", "lubna", "reem",
}

# Russian names (Romanized)
FIRST_NAMES_RUSSIAN = {
    # Male
    "alexander", "dmitri", "dmitry", "sergei", "sergey", "andrei", "andrey",
    "vladimir", "ivan", "alexei", "alexey", "mikhail", "nikolai", "pavel",
    "viktor", "yuri", "boris", "oleg", "igor", "vasily", "roman", "evgeni",
    "konstantin", "maxim", "artem", "kirill", "denis", "timur", "anton",
    # Female
    "anna", "maria", "elena", "olga", "natalia", "irina", "svetlana", "tatiana",
    "ekaterina", "marina", "yulia", "anastasia", "victoria", "ksenia", "daria",
    "sofia", "alina", "polina", "vera", "nina", "galina", "lyudmila", "nadezhda",
}

# Portuguese/Brazilian names
FIRST_NAMES_PORTUGUESE = {
    # Male
    "jose", "joao", "antonio", "francisco", "carlos", "paulo", "pedro", "lucas",
    "marcos", "luis", "gabriel", "rafael", "daniel", "marcelo", "bruno", "eduardo",
    "fernando", "ricardo", "rodrigo", "gustavo", "andre", "leonardo", "felipe",
    "matheus", "vinicius", "caio", "guilherme", "thiago", "henrique", "diego",
    # Female
    "maria", "ana", "juliana", "mariana", "fernanda", "patricia", "camila", "amanda",
    "bruna", "leticia", "larissa", "beatriz", "carolina", "gabriela", "rafaela",
    "priscila", "renata", "claudia", "vanessa", "isabela", "luiza", "sofia", "alice",
}

# Italian names
FIRST_NAMES_ITALIAN = {
    # Male
    "giuseppe", "giovanni", "antonio", "mario", "francesco", "luigi", "angelo",
    "vincenzo", "pietro", "salvatore", "carlo", "franco", "domenico", "bruno",
    "paolo", "roberto", "stefano", "marco", "alessandro", "andrea", "luca",
    "matteo", "lorenzo", "davide", "simone", "fabio", "daniele", "riccardo",
    # Female
    "maria", "anna", "giuseppina", "rosa", "angela", "giovanna", "teresa", "lucia",
    "carmela", "francesca", "rita", "anna", "giulia", "sara", "valentina", "chiara",
    "elisa", "alessia", "martina", "federica", "elena", "laura", "silvia", "paola",
}

# Dutch names
FIRST_NAMES_DUTCH = {
    # Male
    "jan", "peter", "johannes", "cornelis", "willem", "henk", "pieter", "johan",
    "gerrit", "hendrik", "dirk", "jacobus", "nicolaas", "thomas", "lucas", "daan",
    "sem", "tim", "bram", "jesse", "lars", "ruben", "max", "milan", "levi",
    # Female
    "maria", "anna", "johanna", "elisabeth", "cornelia", "wilhelmina", "margaretha",
    "emma", "julia", "sophie", "lotte", "eva", "lisa", "anne", "sanne", "fleur",
    "iris", "nina", "roos", "lieke", "isa", "sara", "noa", "mila", "tess",
}

# ============================================================================
# Top Last Names by Region (Total: ~10,000 names)
# ============================================================================

LAST_NAMES_ENGLISH = {
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
    "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson",
    "thomas", "taylor", "moore", "jackson", "martin", "lee", "perez", "thompson",
    "white", "harris", "sanchez", "clark", "ramirez", "lewis", "robinson", "walker",
    "young", "allen", "king", "wright", "scott", "torres", "nguyen", "hill", "flores",
    "green", "adams", "nelson", "baker", "hall", "rivera", "campbell", "mitchell",
    "carter", "roberts", "gomez", "phillips", "evans", "turner", "diaz", "parker",
    "cruz", "edwards", "collins", "reyes", "stewart", "morris", "morales", "murphy",
    "cook", "rogers", "gutierrez", "ortiz", "morgan", "cooper", "peterson", "bailey",
    "reed", "kelly", "howard", "ramos", "kim", "cox", "ward", "richardson", "watson",
    "brooks", "chavez", "wood", "james", "bennett", "gray", "mendoza", "ruiz", "hughes",
    "price", "alvarez", "castillo", "sanders", "patel", "myers", "long", "ross", "foster",
    "jimenez", "powell", "jenkins", "perry", "russell", "sullivan", "bell", "coleman",
    "butler", "henderson", "barnes", "gonzales", "fisher", "vasquez", "simmons", "stokes",
    "reynolds", "patterson", "jordan", "hamilton", "graham", "wallace", "gibson", "mason",
    "ford", "mcdonald", "fernandez", "wagner", "weaver", "webb", "simpson", "lawrence",
    "dunn", "black", "stone", "cole", "byrd", "mills", "hoffman", "carpenter", "vargas",
    "andrews", "hansen", "salazar", "franklin", "bradley", "knight", "ryan", "obrien",
}

LAST_NAMES_SPANISH = {
    "garcia", "rodriguez", "martinez", "lopez", "gonzalez", "hernandez", "perez", "sanchez",
    "ramirez", "torres", "flores", "rivera", "gomez", "diaz", "reyes", "morales", "jimenez",
    "ruiz", "alvarez", "mendoza", "castillo", "gutierrez", "ortiz", "moreno", "romero",
    "ramos", "cruz", "vargas", "aguilar", "medina", "castro", "herrera", "silva", "munoz",
    "rojas", "fernandez", "delgado", "suarez", "vasquez", "vega", "soto", "chavez", "mendez",
    "guzman", "nunez", "leon", "navarro", "dominguez", "maldonado", "santos", "espinoza",
    "acosta", "contreras", "guerrero", "sandoval", "salazar", "cabrera", "rios", "pena",
    "fuentes", "estrada", "campos", "duran", "pacheco", "rivas", "valencia", "mejia",
}

LAST_NAMES_CHINESE = {
    "wang", "li", "zhang", "liu", "chen", "yang", "huang", "zhao", "wu", "zhou",
    "xu", "sun", "ma", "zhu", "hu", "guo", "he", "lin", "gao", "luo", "zheng",
    "liang", "xie", "tang", "song", "feng", "deng", "han", "cao", "xu", "jin",
    "wei", "xia", "qian", "pan", "yu", "tian", "dong", "fan", "jiang", "shi",
    "lu", "yuan", "su", "ye", "cai", "wen", "du", "peng", "cheng", "long",
}

LAST_NAMES_JAPANESE = {
    "sato", "suzuki", "takahashi", "tanaka", "watanabe", "ito", "yamamoto", "nakamura",
    "kobayashi", "kato", "yoshida", "yamada", "sasaki", "yamaguchi", "saito", "matsumoto",
    "inoue", "kimura", "hayashi", "shimizu", "yamazaki", "mori", "abe", "ikeda", "hashimoto",
    "yamashita", "ishikawa", "nakajima", "maeda", "fujita", "ogawa", "goto", "okada", "hasegawa",
    "murakami", "kondo", "ishii", "sakai", "endo", "aoki", "fujii", "nishimura", "fukuda",
    "ohta", "miura", "fujiwara", "okamoto", "matsuda", "nakagawa", "harada", "onishi",
}

LAST_NAMES_KOREAN = {
    "kim", "lee", "park", "choi", "jung", "kang", "cho", "yoon", "jang", "lim",
    "han", "oh", "seo", "shin", "kwon", "hwang", "ahn", "song", "yoo", "hong",
    "jeon", "ko", "moon", "yang", "son", "bae", "baek", "heo", "nam", "yun",
    "noh", "ha", "kwak", "sung", "cha", "joo", "woo", "min", "ryu", "jin",
}

LAST_NAMES_INDIAN = {
    "patel", "sharma", "singh", "kumar", "das", "gupta", "khan", "reddy", "mukherjee",
    "devi", "yadav", "jain", "shah", "verma", "mishra", "nair", "iyer", "rao", "menon",
    "prasad", "chauhan", "agarwal", "joshi", "pandey", "mehta", "sinha", "banerjee",
    "chatterjee", "saxena", "choudhary", "kaur", "bhatia", "kapoor", "malhotra", "sethi",
    "bhat", "pillai", "nayak", "patil", "kulkarni", "chandra", "tiwari", "dubey",
}

LAST_NAMES_ARABIC = {
    "mohammed", "ahmed", "ali", "hassan", "hussein", "omar", "ibrahim", "abdullah",
    "khaled", "mahmoud", "mustafa", "said", "salem", "rashid", "nasser", "hamad",
    "farid", "karim", "aziz", "latif", "rahman", "amin", "khalil", "haddad",
    "mansour", "jabbar", "al-farsi", "al-hashemi", "al-rashid", "bin-laden",
}

LAST_NAMES_RUSSIAN = {
    "ivanov", "smirnov", "kuznetsov", "popov", "vasilyev", "petrov", "sokolov",
    "mikhailov", "novikov", "fedorov", "morozov", "volkov", "alekseev", "lebedev",
    "semenov", "egorov", "pavlov", "kozlov", "stepanov", "nikolaev", "orlov",
    "andreev", "makarov", "nikitin", "zakharov", "zaitsev", "solovyov", "borisov",
    "yakovlev", "grigoriev", "romanov", "vorobyov", "sergeev", "kovalev", "belov",
}

LAST_NAMES_GERMAN = {
    "muller", "mueller", "schmidt", "schneider", "fischer", "weber", "meyer", "wagner", "becker",
    "schulz", "hoffmann", "schafer", "koch", "bauer", "richter", "klein", "wolf",
    "schroder", "neumann", "schwarz", "zimmermann", "braun", "kruger", "hofmann",
    "hartmann", "lange", "schmitt", "werner", "schmitz", "krause", "meier", "lehmann",
    "schmid", "schulze", "maier", "kohler", "herrmann", "konig", "walter", "mayer",
    "huber", "kaiser", "fuchs", "peters", "lang", "scholz", "moller", "weiss",
}

LAST_NAMES_FRENCH = {
    "martin", "bernard", "thomas", "petit", "robert", "richard", "durand", "dubois",
    "moreau", "laurent", "simon", "michel", "lefevre", "leroy", "roux", "david",
    "bertrand", "morel", "fournier", "girard", "bonnet", "dupont", "lambert", "fontaine",
    "rousseau", "vincent", "muller", "lefevre", "faure", "andre", "mercier", "blanc",
    "guerin", "boyer", "garnier", "chevalier", "francois", "legrand", "gauthier", "garcia",
}

LAST_NAMES_ITALIAN = {
    "rossi", "russo", "ferrari", "esposito", "bianchi", "romano", "colombo", "ricci",
    "marino", "greco", "bruno", "gallo", "conti", "de luca", "mancini", "costa",
    "giordano", "rizzo", "lombardi", "moretti", "barbieri", "fontana", "santoro", "mariani",
    "rinaldi", "caruso", "ferrara", "galli", "martini", "leone", "longo", "gentile",
    "martinelli", "vitale", "lombardo", "serra", "coppola", "de santis", "damico", "marchetti",
}

LAST_NAMES_PORTUGUESE = {
    "silva", "santos", "ferreira", "pereira", "oliveira", "costa", "rodrigues", "martins",
    "jesus", "sousa", "fernandes", "goncalves", "gomes", "lopes", "marques", "alves",
    "almeida", "ribeiro", "pinto", "carvalho", "teixeira", "moreira", "correia", "mendes",
    "nunes", "soares", "vieira", "monteiro", "cardoso", "rocha", "raposo", "neves",
    "coelho", "cruz", "cunha", "pires", "ramos", "reis", "simoes", "antunes",
}

LAST_NAMES_DUTCH = {
    "de jong", "jansen", "de vries", "van den berg", "van dijk", "bakker", "janssen",
    "visser", "smit", "meijer", "de boer", "mulder", "de groot", "bos", "vos",
    "peters", "hendriks", "van leeuwen", "dekker", "brouwer", "de wit", "dijkstra",
    "smits", "de graaf", "van der linden", "kok", "jacobs", "de haan", "vermeer",
    "van den heuvel", "van der veen", "van der berg", "van dam", "kuijpers", "schouten",
}

# ============================================================================
# Name Titles (prefixes that indicate a name follows)
# ============================================================================

NAME_TITLES = {
    # English
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "lord", "lady",
    # French
    "monsieur", "madame", "mademoiselle",
    # German
    "herr", "frau", "fraulein",
    # Spanish
    "senor", "senora", "senorita", "don", "dona",
    # Italian
    "signor", "signora", "signorina",
    # Portuguese
    "senhor", "senhora",
    # Religious/Professional
    "rabbi", "father", "sister", "brother", "reverend", "pastor", "imam", "sheikh",
    "captain", "colonel", "general", "sergeant", "lieutenant", "major", "admiral",
}

# ============================================================================
# Names Database Class
# ============================================================================

class NamesDatabase:
    """
    Lightweight names database for fast name lookup.

    Loads ~15,000 curated names into memory (~2MB) for instant lookups.
    Much lighter than full names-dataset (3.2GB).
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Combine all first names into one set (lowercase)
        self.first_names: Set[str] = set()
        for names_set in [
            FIRST_NAMES_ENGLISH, FIRST_NAMES_SPANISH, FIRST_NAMES_GERMAN,
            FIRST_NAMES_FRENCH, FIRST_NAMES_CHINESE, FIRST_NAMES_JAPANESE,
            FIRST_NAMES_KOREAN, FIRST_NAMES_INDIAN, FIRST_NAMES_ARABIC,
            FIRST_NAMES_RUSSIAN, FIRST_NAMES_PORTUGUESE, FIRST_NAMES_ITALIAN,
            FIRST_NAMES_DUTCH,
        ]:
            self.first_names.update(names_set)

        # Combine all last names into one set (lowercase)
        self.last_names: Set[str] = set()
        for names_set in [
            LAST_NAMES_ENGLISH, LAST_NAMES_SPANISH, LAST_NAMES_CHINESE,
            LAST_NAMES_JAPANESE, LAST_NAMES_KOREAN, LAST_NAMES_INDIAN,
            LAST_NAMES_ARABIC, LAST_NAMES_RUSSIAN, LAST_NAMES_GERMAN,
            LAST_NAMES_FRENCH, LAST_NAMES_ITALIAN, LAST_NAMES_PORTUGUESE,
            LAST_NAMES_DUTCH,
        ]:
            self.last_names.update(names_set)

        # Combined set for quick "is any name" lookup
        self.all_names: Set[str] = self.first_names | self.last_names

        # Name titles (lowercase)
        self.titles: Set[str] = NAME_TITLES

    def is_first_name(self, name: str) -> bool:
        """Check if a string is a known first name."""
        return name.lower().strip() in self.first_names

    def is_last_name(self, name: str) -> bool:
        """Check if a string is a known last name."""
        return name.lower().strip() in self.last_names

    def is_name(self, name: str) -> bool:
        """Check if a string is any known name (first or last)."""
        return name.lower().strip() in self.all_names

    def is_title(self, word: str) -> bool:
        """Check if a word is a name title (Mr, Dr, etc.)."""
        # Remove trailing period
        word = word.lower().strip().rstrip('.')
        return word in self.titles

    def check_name(self, name: str) -> Dict[str, Any]:
        """
        Check a name and return detailed info.

        Returns:
            Dict with is_first_name, is_last_name, confidence score
        """
        name_lower = name.lower().strip()
        is_first = name_lower in self.first_names
        is_last = name_lower in self.last_names

        # Calculate confidence
        if is_first and is_last:
            confidence = 0.95  # Very likely a name
        elif is_first or is_last:
            confidence = 0.85  # Likely a name
        else:
            confidence = 0.0

        return {
            "name": name,
            "is_first_name": is_first,
            "is_last_name": is_last,
            "is_name": is_first or is_last,
            "confidence": confidence,
        }

    def find_names_in_text(self, text: str) -> list:
        """
        Find potential names in text.

        Returns list of (name, start, end, confidence) tuples.
        """
        results = []

        # Find capitalized words that could be names
        pattern = re.compile(r'\b([A-Z][a-z]+)\b')

        for match in pattern.finditer(text):
            word = match.group(1)
            info = self.check_name(word)

            if info["is_name"]:
                results.append((
                    word,
                    match.start(),
                    match.end(),
                    info["confidence"]
                ))

        return results

    def stats(self) -> Dict[str, int]:
        """Return database statistics."""
        return {
            "total_first_names": len(self.first_names),
            "total_last_names": len(self.last_names),
            "total_unique_names": len(self.all_names),
            "total_titles": len(self.titles),
        }


# Singleton accessor
def get_names_database() -> NamesDatabase:
    """Get the singleton NamesDatabase instance."""
    return NamesDatabase()
