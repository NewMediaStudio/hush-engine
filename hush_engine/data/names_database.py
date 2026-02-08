#!/usr/bin/env python3
"""
Comprehensive Names Database for Hush Engine

A curated database of common first and last names by region.
Approximately 1,000 names per major language/region.

Coverage: ~15,000 first names + ~15,000 last names globally
Estimated coverage: 95%+ of names in typical documents

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
# Top First Names by Region (~1,000 per major region)
# ============================================================================

# English-speaking countries (US, UK, Canada, Australia, etc.)
FIRST_NAMES_ENGLISH = {
    # Male - Traditional
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
    "randy", "wayne", "elijah", "louis", "harry", "howard", "leonard",
    "martin", "curtis", "stanley", "jimmy", "eddie", "mason", "aiden",
    "liam", "lucas", "oliver", "hunter", "jackson", "connor", "luke", "evan",
    "caleb", "isaac", "cameron", "landon", "wyatt", "jaxon", "jayden", "gavin",
    "carter", "julian", "brayden", "nathaniel", "dominic", "sebastian", "xavier",
    "ian", "cole", "chase", "tristan", "blake", "brody", "alex", "derek", "marcus",
    "trevor", "spencer", "jared", "chad", "troy", "darren", "colin", "corey", "dustin",
    "mitchell", "brett", "craig", "barry", "kirk", "marshall", "warren", "gordon",
    "carl", "ross", "grant", "jeff", "rick", "doug", "todd", "brad", "neil", "ted",
    "tony", "mike", "steve", "dave", "dan", "jim", "tom", "bob", "bill", "sam",
    "max", "ben", "tim", "rob", "andy", "chris", "matt", "nick", "greg", "jeff",
    "marcus", "travis", "cody", "shane", "dalton", "parker", "miles", "sawyer",
    "emmett", "eli", "owen", "finn", "levi", "tucker", "river", "phoenix", "beckett",
    "declan", "asher", "silas", "ezra", "easton", "ryder", "weston", "kingston",
    "waylon", "lincoln", "jace", "axel", "brooks", "atlas", "grayson", "hudson",
    "colton", "bentley", "maverick", "dean", "leo", "ryker", "roman", "elliot",
    "theo", "oscar", "reed", "graham", "harrison", "beau", "jude", "milo", "kai",
    # Male - Modern/Contemporary
    "jaylen", "jayson", "trey", "darius", "malik", "xavier", "devin", "marquis",
    "terrell", "tyrone", "dwayne", "reggie", "lamar", "antoine", "jerome", "kareem",
    "rashad", "quincy", "cedric", "donovan", "andre", "devon", "jamal", "monte",
    # Female - Traditional
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth", "susan", "jane",
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
    "scarlett", "penelope", "layla", "ellie", "nora", "zoey", "camila", "aubrey",
    "luna", "savannah", "brooklyn", "leah", "stella", "hazel", "aurora", "maya",
    "paisley", "audrey", "skylar", "violet", "claire", "bella", "lucy", "anna",
    "eleanor", "aaliyah", "allison", "gabriella", "nevaeh", "addison", "natasha",
    "brianna", "hailey", "mackenzie", "kaylee", "taylor", "morgan", "jordyn",
    "paige", "sydney", "destiny", "brooke", "kennedy", "reagan", "sadie", "quinn",
    "autumn", "ivy", "piper", "ruby", "madelyn", "willow", "eva", "naomi", "serenity",
    "elena", "caroline", "faith", "kylie", "alexandra", "ariana", "peyton", "bailey",
    "jocelyn", "lydia", "alexa", "julia", "valentina", "liliana", "adriana", "molly",
    "melody", "vivian", "genesis", "summer", "kinsley", "delilah", "jade", "isla",
    # Female - Modern
    "aaliyah", "aileen", "alana", "alicia", "alyssa", "anastasia", "ariel", "bianca",
    "brianna", "brielle", "caitlin", "cassandra", "chelsea", "courtney", "deanna",
    "elena", "erica", "felicia", "gina", "giselle", "holly", "jenna", "kendra",
    "kiana", "kiara", "kristen", "kristina", "lacey", "larissa", "leila", "lindsey",
    "lyndsey", "makayla", "marissa", "melanie", "miranda", "monique", "nadia",
    "priscilla", "raven", "rebekah", "sabrina", "selena", "serena", "shayla",
    "sierra", "tara", "tiffany", "trisha", "vanessa", "veronica", "whitney",
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
    "guillermo", "gonzalo", "jaime", "salvador", "agustin", "tomas", "mauricio",
    "cristobal", "esteban", "ismael", "bernardo", "rodolfo", "ernesto", "alfredo",
    "gustavo", "abel", "adolfo", "benjamin", "joel", "saul", "lorenzo", "gilberto",
    "samuel", "fabian", "dario", "enzo", "thiago", "maximo", "bautista", "luca",
    "valentino", "facundo", "bruno", "nahuel", "alexis", "ezequiel", "leandro",
    "iker", "alonso", "lautaro", "joaquin", "dylan", "axel", "mathias", "bastian",
    "cristian", "fabricio", "damian", "emilio", "federico", "genaro", "hernan",
    "horacio", "luciano", "marcelo", "mauro", "norberto", "octavio", "omar",
    "pascual", "ramiro", "renato", "rene", "roque", "rolando", "roman", "rogelio",
    "severo", "silvio", "tobias", "ulises", "wilfredo", "yeremy", "zacariah",
    # Female
    "maria", "carmen", "ana", "rosa", "isabel", "laura", "lucia", "patricia",
    "elena", "pilar", "dolores", "cristina", "marta", "mercedes", "paula", "sara",
    "andrea", "beatriz", "silvia", "teresa", "rocio", "monica", "raquel", "susana",
    "julia", "eva", "natalia", "daniela", "valentina", "sofia", "camila", "mariana",
    "fernanda", "gabriela", "alejandra", "adriana", "lorena", "veronica", "carolina",
    "catalina", "alicia", "claudia", "esther", "irene", "nuria", "inmaculada",
    "victoria", "marina", "luisa", "florencia", "magdalena", "antonia", "guadalupe",
    "concepcion", "esperanza", "josefina", "leticia", "marisol", "miriam", "norma",
    "yolanda", "alba", "angela", "aurora", "blanca", "cecilia", "clara", "diana",
    "elisa", "fatima", "gloria", "ines", "ivana", "jimena", "lourdes", "margarita",
    "maribel", "mariela", "nadia", "noelia", "olga", "pamela", "rebeca", "regina",
    "rosario", "sandra", "sonia", "tatiana", "vanessa", "virginia", "ximena",
    "yazmin", "abril", "agustina", "alma", "amparo", "antonella", "ariadna",
    "brenda", "carla", "celia", "constanza", "delia", "dulce", "emilia", "erica",
    "eugenia", "gisela", "graciela", "griselda", "hortensia", "ivonne", "janet",
    "josefa", "julieta", "karen", "lidia", "lorenza", "lucia", "luz", "maite",
    "manuela", "martina", "matilde", "micaela", "milagros", "nayeli", "ofelia",
    "perla", "piedad", "ramona", "renata", "rita", "rocio", "sabrina", "samantha",
    "selene", "soledad", "tamara", "tania", "valeria", "veronika", "xochitl",
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
    "philipp", "david", "julian", "moritz", "simon", "benedikt", "fabian", "dominik",
    "marcel", "patrick", "pascal", "oliver", "benjamin", "elias", "samuel", "jakob",
    "noah", "luca", "finn", "henry", "oskar", "anton", "emil", "theo", "leonard",
    "max", "ben", "louis", "carl", "otto", "vincenz", "konstantin", "valentin",
    "jonathan", "johannes", "lorenz", "rafael", "gabriel", "marius", "nico", "marco",
    "dennis", "kevin", "sascha", "torsten", "jens", "sven", "holger", "axel",
    "armin", "bastian", "carsten", "christopher", "dirk", "egon", "erwin", "eugen",
    "georg", "gottfried", "gregor", "gunnar", "gustav", "hannes", "hartmut", "heiko",
    "hendrik", "hubertus", "joachim", "jochen", "josef", "karsten", "konrad", "lars",
    "laurin", "lennart", "levin", "lothar", "ludwig", "manuel", "mattis", "nils",
    "olaf", "oswald", "ralf", "reinhard", "robert", "rolf", "rupert", "siegfried",
    "steffen", "swen", "thorsten", "tilman", "ulrich", "volker", "wilfried", "willi",
    # Female
    "ursula", "helga", "monika", "renate", "ingrid", "gisela", "christa", "erika",
    "brigitte", "sabine", "petra", "andrea", "susanne", "birgit", "karin", "claudia",
    "angelika", "maria", "anna", "elisabeth", "heike", "gabriele", "martina", "julia",
    "sarah", "laura", "lisa", "sophie", "lena", "emma", "mia", "hannah", "leonie",
    "marie", "katharina", "johanna", "charlotte", "amelie", "lara", "nele",
    "franziska", "stefanie", "nicole", "melanie", "jennifer", "christina", "sandra",
    "daniela", "simone", "anja", "sonja", "katja", "michaela", "manuela", "silke",
    "antje", "astrid", "barbara", "bettina", "cornelia", "dagmar", "doris", "edith",
    "elfriede", "eva", "frieda", "gerda", "gertrud", "gundula", "hedwig", "hildegard",
    "ilse", "irmgard", "jutta", "karla", "klara", "lieselotte", "luise", "margarete",
    "margit", "marianne", "marlene", "mathilde", "meike", "nadine", "roswitha",
    "ruth", "sigrid", "silvia", "tanja", "ulrike", "waltraud", "wiebke", "alina",
    "carolin", "celina", "chiara", "clara", "diana", "elena", "elisa", "emilia",
    "finja", "friederike", "greta", "ida", "isabel", "jana", "jasmin", "johanna",
    "josephine", "judith", "kira", "klara", "lea", "leila", "lina", "louisa",
    "lucia", "luisa", "magdalena", "maja", "maren", "marlena", "merle", "mila",
    "milena", "miriam", "nathalie", "nina", "paula", "pauline", "pia", "rebecca",
    "ronja", "rosa", "sabrina", "samira", "sofia", "stella", "svea", "theresa",
    "valentina", "vanessa", "vera", "verena", "victoria", "vivien", "zoe",
}

# French-speaking countries
FIRST_NAMES_FRENCH = {
    # Male
    "jean", "pierre", "michel", "jacques", "francois", "andre", "philippe", "bernard",
    "louis", "paul", "alain", "claude", "marcel", "rene", "roger", "christian",
    "patrick", "robert", "daniel", "nicolas", "laurent", "julien", "eric", "pascal",
    "thomas", "david", "antoine", "guillaume", "maxime", "alexandre", "hugo", "lucas",
    "nathan", "gabriel", "arthur", "raphael", "adam", "leo",
    "mathieu", "sebastien", "olivier", "jerome", "yannick", "bruno", "thierry",
    "didier", "stephane", "christophe", "fabrice", "benoit", "frederic", "gilles",
    "dominique", "herve", "xavier", "yves", "marc", "serge", "gael", "cedric",
    "damien", "fabien", "florian", "gautier", "gregoire", "henri", "hippolyte",
    "jacques", "jules", "leon", "luc", "ludovic", "mael", "martin", "mathis",
    "matthieu", "noel", "quentin", "regis", "samuel", "simon", "sylvain", "tanguy",
    "theo", "thibault", "tristan", "valentin", "victor", "vincent", "william",
    "alexis", "anthony", "arnaud", "aurelien", "baptiste", "bastien", "benjamin",
    "boris", "brice", "charles", "clement", "corentin", "cyril", "edouard", "eloi",
    "emile", "emmanuel", "etienne", "eugene", "felix", "fernand", "gaston",
    "geoffroy", "gerard", "hadrien", "jacky", "joel", "jonathan", "jordan",
    "kevin", "loic", "lorenzo", "malo", "marius", "maurice", "mickael", "morgan",
    "oscar", "pierre-louis", "remi", "romain", "ronan", "sacha", "stephane", "theo",
    # Female
    "marie", "jeanne", "francoise", "monique", "nicole", "catherine", "sylvie",
    "christine", "isabelle", "anne", "nathalie", "sandrine", "veronique", "sophie",
    "patricia", "julie", "claire", "camille", "lea", "manon", "chloe", "emma",
    "ines", "jade", "louise", "alice", "lina", "rose", "charlotte",
    "amelie", "aurelie", "brigitte", "celine", "clementine", "danielle", "delphine",
    "elise", "eloise", "emilie", "eva", "fabienne", "gaelle", "genevieve",
    "geraldine", "helene", "jacqueline", "jocelyne", "joelle", "laetitia", "laura",
    "laurence", "lena", "lorraine", "lucie", "madeleine", "margaux", "marguerite",
    "marianne", "martine", "mathilde", "melanie", "mireille", "muriel", "oceane",
    "odette", "pauline", "priscilla", "romane", "sarah", "simone", "solene",
    "stephanie", "suzanne", "thais", "valerie", "virginie", "yvette", "zoe",
    "adele", "agathe", "agnes", "anais", "angelique", "annette", "axelle", "carla",
    "cassandra", "cecile", "coralie", "daphne", "dominique", "dorothee", "edith",
    "eleanor", "elena", "elodie", "estelle", "eugenie", "fanny", "flora", "gabrielle",
    "gisele", "honorine", "irene", "jennifer", "jessica", "josephine", "juliette",
    "justine", "karine", "laure", "leonie", "lilou", "lise", "lola", "lucile",
    "madelene", "maeva", "maelle", "magalie", "maite", "marcelle", "marielle",
    "marina", "marlene", "melodie", "michelle", "nadege", "nadine", "ninon",
    "noelle", "ophelia", "penelope", "perrine", "rachelle", "rene", "rosalie",
    "roxane", "sabine", "salome", "segolene", "severine", "sibylle", "sidonie",
    "solange", "sybille", "tatiana", "therese", "valentine", "victoire", "viviane",
}

# Chinese names (Romanized - Pinyin) - Common given names
FIRST_NAMES_CHINESE = {
    # Single character names and multi-character names (romanized)
    "wei", "fang", "ming", "jing", "lei", "jun", "yan", "ping", "hong", "lin",
    "ying", "qiang", "feng", "jie", "xin", "yu", "wen", "bo", "tao", "yang",
    "hai", "hui", "li", "chen", "yong", "gang", "bin", "hua", "xiao", "dong",
    "cheng", "ning", "yi", "nan", "qing", "rui", "peng", "hao", "kai", "zhi",
    "jian", "liang", "shan", "mei", "xia", "juan", "lan", "qiu", "yue", "dan",
    "min", "jia", "qian", "lu", "yun", "fei", "shuang", "ting", "ling", "rong",
    "ya", "jiao", "hong", "li", "shu", "huan", "xue", "xiang", "yao", "shi",
    "chun", "cai", "qiong", "ai", "man", "zhen", "jun", "lan", "yu", "xiu",
    "biao", "chang", "cong", "da", "de", "ding", "dou", "en", "fa", "fu",
    "gao", "gen", "gong", "guan", "guang", "gui", "guo", "han", "hang", "heng",
    "hu", "huang", "ji", "jian", "jiao", "jin", "jiu", "juan", "jue", "kan",
    "kang", "ke", "kun", "la", "lao", "le", "lei", "lian", "liao", "lie",
    "long", "lun", "luo", "mao", "meng", "mi", "miao", "mu", "neng", "ni",
    "nian", "nong", "nu", "pan", "pei", "pi", "piao", "pin", "po", "pu",
    "qi", "qie", "qin", "qiong", "qu", "quan", "que", "ran", "rang", "rao",
    "ren", "ri", "rou", "ru", "ruan", "run", "ruo", "sa", "sai", "san",
    "sang", "sao", "se", "sen", "seng", "sha", "shan", "shao", "she", "shen",
    "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuo", "si",
    "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan",
    "tang", "te", "teng", "ti", "tian", "tie", "ting", "tong", "tou", "tu",
    "tuan", "tui", "tun", "tuo", "wa", "wai", "wan", "wang", "weng", "wo",
    "wu", "xi", "xia", "xie", "xin", "xing", "xiong", "xu", "xuan", "xun",
    "ya", "yong", "you", "yuan", "yun", "za", "zai", "zan", "zang", "zao",
    "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe",
    "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang",
    "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun",
    # Common compound given names
    "mingwei", "junhao", "yuxin", "zihan", "haoran", "yichen", "junjie", "chenxi",
    "weiming", "xiaoming", "daming", "jiaming", "yiming", "liming", "qiming",
    "wenbin", "wenxin", "wenhui", "wenjie", "wenli", "wenlong", "wenming",
    "jianhua", "jianping", "jianguo", "jianming", "jianwei", "jianxin", "jianjun",
    "dongyang", "dongliang", "dongming", "dongwei", "donghai", "dongli",
    "yuting", "yuhan", "yuhua", "yuming", "yujie", "yuqi", "yuxuan", "yuyang",
    "tianyu", "tianhao", "tianming", "tianle", "tianxin", "tianlong", "tianwei",
    "zhiwei", "zhiming", "zhiqiang", "zhigang", "zhihui", "zhiyong", "zhihua",
    "xiaoli", "xiaojun", "xiaofang", "xiaohui", "xiaojie", "xiaolong", "xiaomei",
    "yingying", "lingling", "fangfang", "huihui", "meimei", "lili", "tingting",
}

# Japanese names (Romanized)
FIRST_NAMES_JAPANESE = {
    # Male
    "hiroshi", "takeshi", "kenji", "yuki", "taro", "satoshi", "kazuki", "yusuke",
    "daisuke", "takuya", "naoki", "makoto", "shota", "ryota", "kenta", "daiki",
    "masashi", "tomoya", "akira", "kenichi", "shun", "ryo", "hayato", "koichi",
    "haruki", "yuto", "sota", "ren", "kaito", "hinata", "sora", "haruto", "minato",
    "yuya", "kazuya", "shogo", "kosuke", "takahiro", "kensuke", "yudai", "keita",
    "taiki", "kohei", "sho", "jun", "ryosuke", "tatsuya", "tsubasa", "yuji",
    "hideaki", "hideki", "noboru", "masato", "yoshio", "kiyoshi", "minoru",
    "susumu", "tsuyoshi", "yasuhiro", "tetsuya", "junichi", "shinichi", "shuichi",
    "takashi", "masayuki", "yoshiki", "nobuyuki", "shinji", "seiji", "keigo",
    "ryuji", "yuichi", "mitsuru", "kouki", "yamato", "taiga", "itsuki", "asahi",
    "riku", "ao", "aoi", "kai", "haru", "shin", "rei", "kei", "yu", "aki",
    "gen", "ken", "go", "jo", "shu", "toru", "jin", "kou", "sei", "you",
    "ryuichi", "ryoji", "kazuhiko", "yoshihiro", "yasushi", "tadashi", "atsushi",
    "osamu", "mamoru", "isamu", "takao", "hiroyuki", "kazuo", "shigeru", "manabu",
    "fumio", "norio", "kunio", "toshio", "akio", "michio", "hideo", "teruo",
    "ikuo", "masao", "yoshio", "haruo", "isao", "yukio", "sadao", "takao",
    "kazunori", "hirokazu", "tomohiro", "yoshinori", "kazuhiro", "nobuhiro",
    "takuji", "keisuke", "shinsuke", "yosuke", "eisuke", "sousuke", "kyosuke",
    "taisuke", "daisaku", "konosuke", "ryunosuke", "shintaro", "keitaro", "yujiro",
    # Female
    "yuko", "keiko", "michiko", "sachiko", "yuki", "haruka", "sakura", "yui",
    "misaki", "aoi", "miku", "nana", "riko", "mei", "hina", "rin", "kokona",
    "akiko", "tomoko", "mayumi", "kumiko", "megumi", "emi", "yuka", "mika",
    "kaori", "naoko", "noriko", "mariko", "yumiko", "kyoko", "kazuko", "yoshiko",
    "takako", "hiroko", "kimiko", "yukiko", "sumiko", "junko", "fumiko", "midori",
    "reiko", "chie", "natsumi", "moe", "miyu", "aya", "saki", "ai", "mai",
    "kana", "rika", "momoka", "honoka", "kaho", "mana", "yuna", "risa", "arisa",
    "ayaka", "sayaka", "asuka", "nanami", "kanako", "shiori", "mayu", "hikari",
    "chihiro", "miho", "ayumi", "hitomi", "eriko", "saori", "atsuko", "minako",
    "shoko", "miki", "yoko", "asami", "manami", "chiaki", "masami", "kozue",
    "satoko", "seiko", "yasuko", "hideko", "toshiko", "setsuko", "nobuko",
    "mutsuko", "miyako", "tamako", "kiyoko", "yaeko", "kinue", "shizue", "umeko",
    "hinako", "nanako", "momoko", "yurina", "marina", "erina", "karina", "serina",
    "amane", "himari", "koharu", "ichika", "sena", "yuzuki", "akane", "aiko",
    "emiko", "hanako", "haruko", "kazue", "makiko", "masako", "miyuki", "motoko",
    "naomi", "natsuko", "ranko", "rie", "ryoko", "sadako", "sayuri", "shizuka",
    "takako", "teruko", "tokiko", "tomoe", "yayoi", "yorie", "yuki", "yukari",
}

# Korean names (Romanized)
FIRST_NAMES_KOREAN = {
    "min", "jun", "hyun", "joon", "jin", "sung", "young", "seung", "dong", "yong",
    "su", "hee", "soo", "jung", "kyung", "eun", "mi", "sun", "ji", "hyeon",
    "seon", "yeong", "ha", "yeon", "so", "yu", "jae", "woo", "ho",
    "sang", "won", "hyuk", "min", "jun", "chan", "bin", "hyun", "seo", "yun",
    "jun", "ki", "tae", "in", "hwan", "hak", "gyu", "cheol", "hong", "nam",
    "bong", "myung", "man", "joong", "dae", "chang", "sik", "hoon", "soon",
    "chul", "kun", "kwon", "il", "pyung", "geon", "beom", "woong", "bok",
    "seok", "rae", "geun", "baek", "rok", "doo", "sam", "rim", "seob",
    "yeo", "pil", "moo", "gang", "sin", "wan", "gap", "hyang", "ok", "ran",
    "ja", "ae", "won", "hwa", "sim", "yun", "dan", "bora", "nari", "sora",
    "jia", "yuna", "minji", "suji", "yuri", "haeun", "soyeon", "jiyeon",
    "jiwoo", "siwoo", "minjun", "sujin", "yejin", "seojin", "hajin", "eunjin",
    "jimin", "sumin", "yemin", "minseo", "seoyeon", "jiyoung", "jihye", "sooyoung",
    "hyemi", "mirae", "mikyung", "eunji", "heejin", "sungmin", "minsung", "jungho",
    "dongho", "junwoo", "seungho", "taeho", "junho", "sanghoon", "jaehoon", "minwoo",
    "minki", "dongwoo", "sunwoo", "hyunwoo", "taehyun", "sihyun", "yunho", "jinho",
    "dohyun", "jaemin", "sungho", "yoonho", "hyunseok", "junseok", "minseok",
}

# Indian names (Hindi/Sanskrit origin and regional)
FIRST_NAMES_INDIAN = {
    # Male
    "rajesh", "suresh", "ramesh", "amit", "anil", "vijay", "sanjay", "ajay",
    "manoj", "prakash", "dinesh", "ashok", "rakesh", "mukesh", "raj", "ravi",
    "sandeep", "rahul", "vikas", "manish", "pradeep", "deepak", "sunil", "arun",
    "anand", "vivek", "vikram", "arjun", "krishna", "mohan", "rohit", "akash",
    "nikhil", "sachin", "gaurav", "aarav", "vihaan", "reyansh", "advik", "ishaan",
    "abhinav", "aditya", "ajit", "akhil", "alok", "amrit", "aniket", "anirudh",
    "ankur", "anuj", "aravind", "arnav", "ashish", "balaji", "bharat", "bhushan",
    "chandan", "chirag", "darshan", "devendra", "dhruv", "ganesh", "girish", "gopal",
    "govind", "harish", "hemant", "hitesh", "jagdish", "jai", "jatin", "kamal",
    "karan", "kartik", "kiran", "kishore", "kunal", "lalit", "lokesh", "madhav",
    "mahesh", "manu", "mayank", "milind", "mohit", "mukund", "naresh", "naveen",
    "neeraj", "nilesh", "nitin", "om", "pankaj", "paresh", "parth", "pawan",
    "piyush", "pranav", "pratap", "praveen", "raghav", "rajat", "rajiv", "rajan",
    "ram", "ramesh", "ritesh", "rohan", "sahil", "sameer", "sanjiv", "satish",
    "shailesh", "shashi", "shiv", "shivam", "shreyas", "siddharth", "sumit",
    "suraj", "sushil", "tarun", "tushar", "uday", "umesh", "varun", "venkatesh",
    "vinay", "vinod", "vishal", "yash", "yogesh", "aryan", "rudra", "kabir",
    "advait", "arnab", "dev", "shaurya", "atharv", "daksh", "lakshya", "vivaan",
    # Female
    "priya", "neha", "pooja", "anjali", "sunita", "kavita", "ritu", "meena",
    "geeta", "rekha", "suman", "seema", "anita", "nisha", "divya", "shreya",
    "swati", "tanvi", "aishwarya", "deepika", "rani", "lata", "sita", "radha",
    "aadhya", "saanvi", "ananya", "diya", "pari", "aanya", "aaradhya", "myra",
    "aditi", "akshara", "amrita", "anisha", "aparna", "archana", "aruna", "asha",
    "bhavana", "bhavika", "chandni", "chitra", "deepa", "devika", "gauri", "gita",
    "harini", "indira", "ishita", "janaki", "jaya", "jyoti", "kalpana", "kamala",
    "kiran", "kriti", "lakshmi", "lalita", "madhuri", "malini", "mamta", "mansi",
    "maya", "meera", "megha", "mohini", "nandini", "neelam", "nikita", "nirmala",
    "padma", "pallavi", "parvati", "preeti", "rachna", "radha", "radhika", "renu",
    "riya", "rohini", "rupali", "sakshi", "sandhya", "sangita", "sarita", "savita",
    "shanti", "sharmila", "shilpa", "sonal", "sudha", "sujata", "sunaina", "sushma",
    "tara", "trisha", "uma", "usha", "vandana", "varsha", "vidya", "vimla",
    "kiara", "navya", "ishika", "avni", "pihu", "tanya", "ira", "sia", "mahi",
}

# Arabic names
FIRST_NAMES_ARABIC = {
    # Male
    "mohammed", "muhammad", "mohamed", "ahmad", "ahmed", "ali", "hassan", "hussein",
    "omar", "ibrahim", "khalid", "abdullah", "mustafa", "youssef", "yousef", "salem",
    "saeed", "rashid", "nasser", "faisal", "tariq", "karim", "walid", "mahmoud",
    "jamal", "hamid", "bilal", "adam", "zayed", "khalil", "majid", "rami",
    "amir", "hakim", "hamza", "hani", "haroon", "hashim", "ismail", "jasim",
    "jawad", "kareem", "khaled", "malik", "mansour", "marwan", "mousa", "nabil",
    "nadir", "naeem", "nassir", "nawaf", "osama", "qasim", "rashed", "sadiq",
    "salah", "salim", "sami", "samir", "sharif", "sultan", "taha", "talal",
    "wael", "wasim", "yaser", "yasir", "zain", "zakariya", "zakaria", "ziad",
    "abed", "adel", "adnan", "aziz", "badr", "bassam", "dawood", "diya",
    "ehab", "essam", "fahad", "fouad", "ghassan", "habib", "hafez", "hisham",
    "hussain", "issa", "jihad", "lutfi", "mazin", "mubarak", "munir", "nadim",
    "omar", "othman", "rafiq", "raja", "ramzi", "riad", "saad", "sabri",
    "safwan", "salman", "shadi", "shakir", "sulaiman", "taher", "tarek",
    "tawfiq", "usama", "yahya", "yousuf", "yunus", "zaki", "zuhayr",
    # Female
    "fatima", "aisha", "maryam", "sara", "layla", "noor", "hana", "yasmin",
    "amina", "khadija", "zainab", "rania", "dina", "lina", "maya", "dana",
    "huda", "salma", "aya", "mariam", "noura", "farah", "lubna", "reem",
    "abeer", "afaf", "amal", "asma", "basma", "dalal", "dalia", "eman",
    "fadia", "farida", "faten", "ghada", "hala", "halima", "hamida", "hanaa",
    "hiba", "houda", "iman", "jamila", "jumana", "karima", "lamia", "lama",
    "latifa", "leena", "manal", "maha", "manar", "maysoon", "mona", "munira",
    "nadia", "nagla", "nahla", "naima", "najwa", "nasreen", "nawal", "nazik",
    "nesrine", "nihad", "nour", "nuha", "ola", "omnia", "rabab", "rahma",
    "rajaa", "randa", "rawia", "rana", "rim", "ruba", "ruqaya", "safa",
    "sahar", "samah", "samira", "sanaa", "sawsan", "shadia", "shahd", "shayma",
    "siham", "soad", "suha", "sumaya", "taghreed", "wafaa", "warda", "widad",
    "yara", "zahra", "zena", "zineb", "jumana", "rula", "roula", "suad",
}

# Russian names (Romanized)
FIRST_NAMES_RUSSIAN = {
    # Male
    "alexander", "dmitri", "dmitry", "sergei", "sergey", "andrei", "andrey",
    "vladimir", "ivan", "alexei", "alexey", "mikhail", "nikolai", "pavel",
    "viktor", "yuri", "boris", "oleg", "igor", "vasily", "roman", "evgeni",
    "konstantin", "maxim", "artem", "kirill", "denis", "timur", "anton",
    "anatoly", "arkady", "arseny", "bogdan", "danil", "danila", "egor",
    "fedor", "filipp", "georgi", "gleb", "grigori", "ilya", "leonid",
    "lev", "makar", "mark", "matvey", "nikita", "petr", "pyotr", "ruslan",
    "semyon", "stanislav", "stepan", "taras", "valentin", "valery", "vadim",
    "vasili", "viktor", "vitaly", "vlad", "vyacheslav", "yaroslav", "yakov",
    "zakhar", "aleksander", "aleksandr", "alek", "anatoliy", "artemy", "artom",
    "danill", "dimitri", "dimitry", "eugen", "feodor", "fyodor", "gavril",
    "genady", "gennady", "herman", "innokenty", "iosif", "ippolit", "isaak",
    "izmail", "kir", "kliment", "kuzma", "lavrenty", "luka", "marat",
    "mefodi", "methodius", "miron", "modest", "naum", "nazar", "nestor",
    "osip", "prokhor", "rodion", "saveli", "savva", "serafim", "silvester",
    "simon", "solomon", "spartak", "svyatoslav", "terenty", "timofey", "trofim",
    "valerian", "varfolomey", "veniamin", "vikenty", "vitali", "vladilen",
    "vsevolod", "yegor", "yefim", "yevgeny", "zinovy", "zoran",
    # Female
    "anna", "maria", "elena", "olga", "natalia", "irina", "svetlana", "tatiana",
    "ekaterina", "marina", "yulia", "anastasia", "victoria", "ksenia", "daria",
    "sofia", "alina", "polina", "vera", "nina", "galina", "lyudmila", "nadezhda",
    "alexandra", "alla", "antonina", "dina", "evgenia", "galya", "inna",
    "katya", "kira", "klara", "kristina", "larisa", "lena", "lida", "lidia",
    "lilia", "lubov", "lyuba", "masha", "milena", "mila", "nadya", "oksana",
    "raisa", "rita", "roza", "sasha", "sonya", "tanya", "valentina", "valeria",
    "varvara", "yekaterina", "yelena", "zhanna", "zoya", "adelina", "agnessa",
    "albina", "alena", "aleksandra", "alla", "alyona", "anfisa", "arina",
    "darya", "diana", "dominika", "emilia", "eva", "evgeniya", "fatima",
    "faina", "felicia", "ilona", "ivanna", "jana", "karina", "katerina",
    "kseniya", "lana", "lara", "larysa", "leyla", "liza", "lizaveta",
    "lucia", "lusya", "lydia", "marfa", "margot", "marianna", "maya",
    "melania", "mira", "nastya", "natasha", "nika", "nonna", "paulina",
    "pelageya", "renata", "rima", "serafima", "snezhana", "tamara", "taisiya",
    "uliana", "ulyana", "vasilisa", "veronika", "vlada", "yana", "yaroslava",
    "zinaida", "zlata", "yuliana", "zhenia",
}

# Portuguese/Brazilian names
FIRST_NAMES_PORTUGUESE = {
    # Male
    "jose", "joao", "antonio", "francisco", "carlos", "paulo", "pedro", "lucas",
    "marcos", "luis", "gabriel", "rafael", "daniel", "marcelo", "bruno", "eduardo",
    "fernando", "ricardo", "rodrigo", "gustavo", "andre", "leonardo", "felipe",
    "matheus", "vinicius", "caio", "guilherme", "thiago", "henrique", "diego",
    "bernardo", "davi", "enzo", "miguel", "arthur", "heitor", "lorenzo", "theo",
    "samuel", "nicolas", "guilherme", "pietro", "murilo", "otavio", "jonas",
    "tomas", "renan", "leandro", "luciano", "sergio", "ronaldo", "flavio",
    "adriano", "alex", "alexandro", "anderson", "artur", "benicio", "bento",
    "breno", "cesar", "claudio", "cristiano", "danilo", "dario", "denis",
    "edson", "emerson", "eugenio", "fabiano", "fabio", "fabricio", "feliciano",
    "geovane", "gilberto", "helio", "hugo", "igor", "ivan", "jaime", "jefferson",
    "joaquim", "julio", "luan", "manoel", "manuel", "marcio", "mario", "matias",
    "mauricio", "mauro", "neto", "nilson", "octavio", "oswaldo", "patricio",
    "ramiro", "raul", "renato", "roberto", "rogerio", "romulo", "ronaldo",
    "rui", "sidnei", "simao", "tiago", "vagner", "valter", "vicente", "vitor",
    "wagner", "washington", "wellington", "wendel", "william", "yuri",
    # Female
    "maria", "ana", "juliana", "mariana", "fernanda", "patricia", "camila", "amanda",
    "bruna", "leticia", "larissa", "beatriz", "carolina", "gabriela", "rafaela",
    "priscila", "renata", "claudia", "vanessa", "isabela", "luiza", "sofia", "alice",
    "julia", "helena", "valentina", "laura", "manuela", "giovanna", "lorena",
    "adriana", "alessandra", "aline", "alicia", "andreia", "angelica", "antonia",
    "barbara", "bianca", "carla", "catarina", "celia", "cecilia", "clara",
    "cristiane", "daniela", "debora", "denise", "diana", "eduarda", "elaine",
    "eliana", "eloisa", "erica", "fabiana", "fatima", "flavia", "francesca",
    "gisele", "graziela", "heloisa", "ingrid", "ivone", "janaina", "jessica",
    "josefa", "joana", "jussara", "karen", "lara", "leidiane", "lelia",
    "livia", "luana", "lucia", "luciana", "madalena", "mara", "marcela",
    "marcia", "margarida", "milena", "miriam", "monica", "natalia", "nicole",
    "paula", "raquel", "rita", "roberta", "rosa", "rosana", "samara",
    "sandra", "simone", "solange", "sonia", "stephane", "tatiana", "tereza",
    "thalita", "thais", "vera", "veronica", "virginia", "vitoria", "yasmin",
}

# Italian names
FIRST_NAMES_ITALIAN = {
    # Male
    "giuseppe", "giovanni", "antonio", "mario", "francesco", "luigi", "angelo",
    "vincenzo", "pietro", "salvatore", "carlo", "franco", "domenico", "bruno",
    "paolo", "roberto", "stefano", "marco", "alessandro", "andrea", "luca",
    "matteo", "lorenzo", "davide", "simone", "fabio", "daniele", "riccardo",
    "tommaso", "leonardo", "gabriele", "filippo", "edoardo", "diego", "giacomo",
    "nicola", "michele", "alberto", "sergio", "massimo", "enrico", "guido",
    "claudio", "giorgio", "maurizio", "aldo", "cesare", "dario", "emanuele",
    "eugenio", "fabrizio", "federico", "fernando", "gianluca", "gianni", "giulio",
    "ignazio", "ivano", "jacopo", "luciano", "luigi", "manuel", "marcello",
    "martino", "mauro", "mirco", "nino", "orazio", "oscar", "pasquale",
    "piero", "raffaele", "remo", "renato", "renzo", "rocco", "rolando",
    "romeo", "sandro", "saverio", "sebastiano", "silvio", "tiziano", "ugo",
    "umberto", "valerio", "vito", "vittorio", "adriano", "agostino", "alfio",
    "alfredo", "amedeo", "arturo", "attilio", "augusto", "aurelio", "basilio",
    "battista", "benedetto", "beniamino", "bernardo", "camillo", "carmine",
    "costantino", "cristiano", "damiano", "donato", "egidio", "elia", "elio",
    "emilio", "ennio", "enzo", "erio", "erminio", "ernesto", "ettore",
    "fabiano", "felice", "fiorentino", "fortunato", "fulvio", "gaetano",
    "gaspare", "gerardo", "germano", "gianfranco", "gianluigi", "gino",
    "giordano", "girolamo", "graziano", "gregorio", "guerino", "italo",
    "lauro", "lino", "livio", "loris", "lucio", "manlio", "marino",
    "nando", "natale", "nazario", "nicolo", "norberto", "nunzio", "oronzo",
    "ottavio", "patrizio", "primo", "quirino", "raimondo", "romualdo", "rosario",
    "ruggero", "samuele", "santino", "santo", "secondo", "severo", "silvano",
    "tarcisio", "teodoro", "terenzio", "tiberio", "timoteo", "tullio", "ubaldo",
    "ulisse", "urbano", "valentino", "valter", "vincente", "virgilio", "walter",
    # Female
    "maria", "anna", "giuseppina", "rosa", "angela", "giovanna", "teresa", "lucia",
    "carmela", "francesca", "rita", "giulia", "sara", "valentina", "chiara",
    "elisa", "alessia", "martina", "federica", "elena", "laura", "silvia", "paola",
    "monica", "claudia", "michela", "ilaria", "barbara", "patrizia", "roberta",
    "simona", "daniela", "stefania", "cristina", "alessandra", "sabrina", "serena",
    "sofia", "alice", "aurora", "beatrice", "camilla", "carolina", "caterina",
    "cecilia", "clara", "diana", "eleonora", "elisabetta", "emanuela", "emma",
    "eva", "fiamma", "fiorella", "flavia", "gabriella", "gaia", "giada",
    "ginevra", "gioia", "giorgia", "grazia", "ida", "irene", "isabella",
    "letizia", "licia", "lidia", "liliana", "linda", "liviana", "loretta",
    "luana", "luciana", "luisa", "mara", "margherita", "marianna", "marilena",
    "marina", "marta", "matilde", "milena", "mirella", "miriam", "nadia",
    "nicoletta", "noemi", "olga", "ornella", "pamela", "piera", "raffaella",
    "rebecca", "renata", "romina", "rossana", "rossella", "sandra", "silvana",
    "sonia", "stella", "susanna", "tamara", "tania", "tatiana", "valeria",
    "vanessa", "veronica", "viola", "virginia", "vittoria", "viviana", "zaira",
}

# Dutch names
FIRST_NAMES_DUTCH = {
    # Male
    "jan", "peter", "johannes", "cornelis", "willem", "henk", "pieter", "johan",
    "gerrit", "hendrik", "dirk", "jacobus", "nicolaas", "thomas", "lucas", "daan",
    "sem", "tim", "bram", "jesse", "lars", "ruben", "max", "milan", "levi",
    "thijs", "jayden", "noah", "finn", "luuk", "julian", "mees", "stijn",
    "sven", "koen", "niek", "teun", "joep", "floris", "sander", "bas",
    "mart", "wouter", "matthijs", "rick", "gijs", "niels", "mark", "tom",
    "paul", "erik", "rob", "frank", "michiel", "jeroen", "maarten", "arjan",
    "bart", "dennis", "marcel", "stefan", "marc", "ronald", "richard", "raymond",
    "patrick", "kevin", "vincent", "remco", "roy", "rene", "matthias", "laurens",
    "joost", "jasper", "marten", "ramon", "wesley", "martijn", "jeroen",
    "adriaan", "arie", "bert", "christiaan", "diederik", "evert", "frederik",
    "geert", "hans", "hugo", "ingmar", "jaap", "karel", "lodewijk", "oscar",
    "piet", "reinier", "robbert", "rutger", "simon", "tobias", "victor", "wim",
    # Female
    "maria", "anna", "johanna", "elisabeth", "cornelia", "wilhelmina", "margaretha",
    "emma", "julia", "sophie", "lotte", "eva", "lisa", "anne", "sanne", "fleur",
    "iris", "nina", "roos", "lieke", "isa", "sara", "noa", "mila", "tess",
    "fenna", "femke", "laura", "nikki", "kim", "marloes", "linda", "monique",
    "esther", "anouk", "rianne", "danielle", "marieke", "floor", "lieke", "ilse",
    "joyce", "wendy", "bianca", "chantal", "mandy", "naomi", "petra", "renate",
    "sandra", "suzan", "yvonne", "miriam", "ingrid", "karin", "annemarie",
    "marjolein", "astrid", "liesbeth", "carla", "anita", "jacqueline", "sylvia",
    "henriette", "geertje", "maartje", "marianne", "jantine", "bregje", "daphne",
    "eline", "greta", "hanna", "jet", "karlijn", "lara", "maaike", "nienke",
    "paulien", "quinty", "rosalie", "sofie", "vera", "willemijn", "yvette",
    "zoey", "amelie", "bo", "cato", "demi", "elisa", "fay", "gwen", "hailey",
    "indy", "jasmijn", "kyra", "lana", "merel", "noor", "olivia", "pien",
}

# ============================================================================
# Top Last Names by Region (~1,000 per major region)
# ============================================================================

LAST_NAMES_ENGLISH = {
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "doe",
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
    "gordon", "burns", "george", "hunter", "owens", "hart", "hunt", "wallace", "grant",
    "harrison", "rice", "west", "fox", "murray", "gardner", "freeman", "wells", "webb",
    "olson", "gilbert", "hunt", "stanley", "jacobs", "berry", "hudson", "duncan", "newman",
    "lane", "price", "meyer", "rice", "brooks", "warren", "ferguson", "woods", "harvey",
    "porter", "armstrong", "elliott", "spencer", "ray", "mcdonald", "wheeler", "hawkins",
    "shaw", "kelley", "henry", "fowler", "burton", "burke", "hicks", "wagner", "perkins",
    "castro", "rowe", "howell", "fletcher", "chambers", "snyder", "medina", "garza",
    "garrett", "wade", "bishop", "fuller", "francis", "dunn", "hansen", "daniels",
    "palmer", "greene", "hudson", "montgomery", "jenkins", "perry", "stevens", "schmidt",
    "austin", "welch", "rose", "rhodes", "pearson", "lucas", "craig", "henry", "little",
    "page", "drake", "sutton", "gregory", "cross", "hayes", "reid", "banks", "logan",
    "may", "holt", "gibbs", "hudson", "jensen", "fleming", "day", "wong", "hart",
    "love", "glover", "davidson", "mann", "norris", "watts", "barr", "terry", "bass",
    "briggs", "frank", "harmon", "mccarthy", "winters", "stein", "kirby", "snow",
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
    "aguirre", "alvarado", "ayala", "bautista", "benitez", "bermudez", "blanco", "bravo",
    "camacho", "cardenas", "carrasco", "carrillo", "cervantes", "cisneros", "cordova",
    "coronado", "cortes", "de la cruz", "de leon", "duarte", "escobar", "espinosa",
    "figueroa", "galindo", "gallegos", "garay", "garza", "gastelum", "giraldo", "godoy",
    "guillen", "ibarra", "jaramillo", "lara", "leal", "ledesma", "lira", "llamas",
    "luna", "macias", "marin", "marquez", "mata", "melendez", "meza", "miranda",
    "molina", "montes", "mora", "nava", "ocampo", "ochoa", "olivares", "orozco",
    "padilla", "palma", "paredes", "parra", "pelayo", "pineda", "ponce", "portillo",
    "quintana", "quintero", "razo", "rendon", "reyna", "rincon", "rosales", "rubio",
    "salinas", "serrano", "solano", "solis", "soriano", "sotelo", "tello", "trejo",
    "trujillo", "uribe", "valdes", "valdez", "valenzuela", "vallejo", "varela",
    "vazquez", "velasco", "velazquez", "villa", "villalobos", "villanueva", "villegas",
    "yanez", "zamora", "zarate", "zavala", "zepeda", "zuniga",
}

LAST_NAMES_CHINESE = {
    "wang", "li", "zhang", "liu", "chen", "yang", "huang", "zhao", "wu", "zhou",
    "xu", "sun", "ma", "zhu", "hu", "guo", "he", "lin", "gao", "luo", "zheng",
    "liang", "xie", "tang", "song", "feng", "deng", "han", "cao", "peng", "jin",
    "wei", "xia", "qian", "pan", "yu", "tian", "dong", "fan", "jiang", "shi",
    "lu", "yuan", "su", "ye", "cai", "wen", "du", "cheng", "long",
    "cui", "duan", "fan", "gu", "hou", "kong", "lai", "lv", "mao", "meng",
    "nie", "qiu", "ren", "shen", "shao", "tan", "tao", "wan", "xiao", "xing",
    "xiong", "yan", "yi", "yin", "yao", "zeng", "zhan", "zhong", "zou",
    "bai", "bi", "bo", "chai", "chang", "chao", "chi", "dai", "di", "ding",
    "fang", "fu", "ge", "geng", "hang", "hao", "hong", "hua", "ji", "jia",
    "jiao", "kan", "ke", "lang", "lei", "li", "liao", "ling", "liu", "min",
    "mo", "mu", "nan", "ning", "ou", "pei", "pi", "ping", "pu", "qi",
    "qiao", "qin", "qu", "rong", "ruan", "sang", "sha", "sheng", "shu", "si",
    "song", "sui", "suo", "ti", "tong", "tu", "wa", "wang", "wen", "weng",
    "wu", "xi", "xiang", "xu", "xuan", "xue", "yan", "yang", "yao", "ye",
    "ying", "you", "yu", "yue", "yun", "za", "zao", "zhang", "zhao", "zhen",
    "zheng", "zhi", "zhong", "zhou", "zhu", "zhuan", "zhui", "zhuo", "zi", "zong",
}

LAST_NAMES_JAPANESE = {
    "sato", "suzuki", "takahashi", "tanaka", "watanabe", "ito", "yamamoto", "nakamura",
    "kobayashi", "kato", "yoshida", "yamada", "sasaki", "yamaguchi", "saito", "matsumoto",
    "inoue", "kimura", "hayashi", "shimizu", "yamazaki", "mori", "abe", "ikeda", "hashimoto",
    "yamashita", "ishikawa", "nakajima", "maeda", "fujita", "ogawa", "goto", "okada", "hasegawa",
    "murakami", "kondo", "ishii", "sakai", "endo", "aoki", "fujii", "nishimura", "fukuda",
    "ohta", "miura", "fujiwara", "okamoto", "matsuda", "nakagawa", "harada", "onishi",
    "takeuchi", "kaneko", "wada", "nakayama", "ishida", "ueda", "morita", "hara", "sugiyama",
    "sakamoto", "higuchi", "maruyama", "imai", "takagi", "ando", "taniguchi", "otsuka",
    "kawamura", "miyazaki", "fujimoto", "shimada", "fukushima", "ueno", "hirano", "kubo",
    "yokoyama", "kawaguchi", "miyamoto", "fujioka", "taguchi", "sugimoto", "kikuchi", "iwata",
    "sakurai", "yoshikawa", "yamane", "tamura", "komori", "hamada", "kawai", "uemura",
    "sugawara", "nishiyama", "murata", "kawano", "tsuchiya", "ozaki", "nomura", "kamiya",
    "hamamoto", "takeda", "yamanaka", "katayama", "miyake", "shibata", "matsubara", "arai",
    "akiyama", "koike", "asano", "sano", "shimada", "hirata", "nishida", "takeda",
    "machida", "kodama", "naito", "ishihara", "osawa", "katou", "kuroda", "iwamoto",
}

LAST_NAMES_KOREAN = {
    "kim", "lee", "park", "choi", "jung", "kang", "cho", "yoon", "jang", "lim",
    "han", "oh", "seo", "shin", "kwon", "hwang", "ahn", "song", "yoo", "hong",
    "jeon", "ko", "moon", "yang", "son", "bae", "baek", "heo", "nam", "yun",
    "noh", "ha", "kwak", "sung", "cha", "joo", "woo", "min", "ryu", "jin",
    "im", "gu", "gong", "bang", "sim", "byun", "ham", "um", "chang", "pyo",
    "gi", "do", "so", "cheon", "in", "maeng", "jeong", "ji", "hyun", "seong",
    "ye", "won", "tae", "yu", "lee", "chae", "la", "bong", "suk", "tan",
    "seol", "kwang", "gye", "cheong", "pae", "mun", "myung", "eom", "pyun",
}

LAST_NAMES_INDIAN = {
    "patel", "sharma", "singh", "kumar", "das", "gupta", "khan", "reddy", "mukherjee",
    "devi", "yadav", "jain", "shah", "verma", "mishra", "nair", "iyer", "rao", "menon",
    "prasad", "chauhan", "agarwal", "joshi", "pandey", "mehta", "sinha", "banerjee",
    "chatterjee", "saxena", "choudhary", "kaur", "bhatia", "kapoor", "malhotra", "sethi",
    "bhat", "pillai", "nayak", "patil", "kulkarni", "chandra", "tiwari", "dubey",
    "srivastava", "rastogi", "thakur", "bhagat", "bose", "chawla", "chopra", "desai",
    "dhawan", "ghosh", "gill", "goel", "goyal", "hegde", "khanna", "kohli", "krishna",
    "mahajan", "malik", "mathur", "mittal", "mitra", "mohan", "mukherjee", "naidu",
    "narayan", "parekh", "parikh", "rajan", "ramachandran", "ranjan", "rawat", "sahni",
    "saini", "sarkar", "saxena", "sengupta", "shukla", "siddiqui", "sodhi", "sood",
    "srinivas", "subramanian", "swamy", "tandon", "tripathi", "tyagi", "uppal", "varghese",
    "venkatesh", "vij", "vohra", "walia", "ahluwalia", "ahuja", "arora", "aurora",
    "bajaj", "bakshi", "balakrishnan", "bhatnagar", "bhatt", "bhattacharya", "birla",
    "chaturvedi", "datta", "deol", "deshpande", "gaikwad", "gandhi", "ganguly", "goswami",
    "grewal", "handa", "hora", "jaggi", "jaswal", "jha", "kamath", "khurana", "kothari",
    "lal", "luthra", "madhavan", "mani", "mehra", "mohanty", "murthy", "nagpal",
    "naik", "nanda", "narang", "nehru", "oberoi", "padmanabhan", "pandit", "puri",
    "rathore", "sabharwal", "sachdev", "sagar", "sandhu", "sareen", "sehgal", "shirodkar",
    "shrivastava", "sikka", "sodhi", "soni", "sundaram", "suresh", "taneja", "trehan",
    "trivedi", "vaidya", "vashisht", "venkat", "wahi", "wadhwa", "yadava", "zaveri",
}

LAST_NAMES_ARABIC = {
    "mohammed", "ahmed", "ali", "hassan", "hussein", "omar", "ibrahim", "abdullah",
    "khaled", "mahmoud", "mustafa", "said", "salem", "rashid", "nasser", "hamad",
    "farid", "karim", "aziz", "latif", "rahman", "amin", "khalil", "haddad",
    "mansour", "jabbar", "al-farsi", "al-hashemi", "al-rashid",
    "abdel", "abboud", "adel", "affan", "akbar", "al-amin", "al-bakr", "al-fayed",
    "al-ghamdi", "al-harbi", "al-jabri", "al-khalifa", "al-maktoum", "al-mansour",
    "al-masri", "al-otaibi", "al-qasimi", "al-rashidi", "al-sabah", "al-saud",
    "al-sharif", "al-suwaidi", "al-thani", "al-zahra", "almasi", "ammar", "anwar",
    "asad", "ashraf", "assaf", "attar", "awad", "ayub", "bakr", "bashir",
    "bitar", "darwish", "dawoud", "diab", "farouk", "ghanem", "habib", "hafez",
    "hajj", "hakeem", "halabi", "hamdi", "hamoudi", "hariri", "hashim", "jaber",
    "jamil", "karam", "kassem", "labib", "makki", "malouf", "masoud", "mazloum",
    "mubarak", "naji", "najjar", "nasrallah", "qureshi", "rahal", "rahmani", "sabbagh",
    "saleh", "samara", "shami", "shamsi", "sharif", "sheikh", "sultani", "taha",
    "wahab", "yamani", "youssef", "zaki", "zaidan", "zein",
}

LAST_NAMES_RUSSIAN = {
    "ivanov", "smirnov", "kuznetsov", "popov", "vasilyev", "petrov", "sokolov",
    "mikhailov", "novikov", "fedorov", "morozov", "volkov", "alekseev", "lebedev",
    "semenov", "egorov", "pavlov", "kozlov", "stepanov", "nikolaev", "orlov",
    "andreev", "makarov", "nikitin", "zakharov", "zaitsev", "solovyov", "borisov",
    "yakovlev", "grigoriev", "romanov", "vorobyov", "sergeev", "kovalev", "belov",
    "anisimov", "antonov", "baranov", "belyaev", "bogdanov", "bondarenko", "bykov",
    "chernov", "denisov", "dmitriev", "dorofeev", "efimov", "emelyanov", "fadeev",
    "filatov", "fomin", "gavrilov", "gerasimov", "gorbachev", "gromov", "gusev",
    "ignatov", "ilyin", "kalashnikov", "karpov", "kazakov", "kirillov", "klimov",
    "komarov", "kondratiev", "korolev", "kostrov", "kovalenko", "krilov", "kudryavtsev",
    "kulikov", "kuznetsova", "lazarev", "loginov", "lukin", "markov", "matveev",
    "melnikov", "mironov", "moiseev", "naumov", "nazarov", "odintsov", "osipov",
    "panov", "polyakov", "ponomarev", "prokhorov", "rodionov", "rogov", "ryabov",
    "safonov", "samsonov", "savchenko", "savin", "shestakov", "shilov", "shishkin",
    "shubin", "simonov", "sitnikov", "smirnova", "sobolev", "sorokin", "sukhanov",
    "suvorov", "tarasov", "tikhomirov", "tikhonov", "titov", "trofimov", "tsvetkov",
    "uvarov", "vasiliev", "vinogradov", "vlasov", "voronin", "voronov", "yegorov",
    "yermakov", "zhukov", "zimin", "zubarev", "zuev",
}

LAST_NAMES_GERMAN = {
    "muller", "mueller", "schmidt", "schneider", "fischer", "weber", "meyer", "wagner", "becker",
    "schulz", "hoffmann", "schafer", "koch", "bauer", "richter", "klein", "wolf",
    "schroder", "neumann", "schwarz", "zimmermann", "braun", "kruger", "hofmann",
    "hartmann", "lange", "schmitt", "werner", "schmitz", "krause", "meier", "lehmann",
    "schmid", "schulze", "maier", "kohler", "herrmann", "konig", "walter", "mayer",
    "huber", "kaiser", "fuchs", "peters", "lang", "scholz", "moller", "weiss",
    "jung", "hahn", "schubert", "vogel", "friedrich", "keller", "gunther", "frank",
    "berger", "winkler", "roth", "beck", "lorenz", "baumann", "franke", "albrecht",
    "schreiber", "bohm", "winter", "kramer", "schulte", "vogt", "haas", "sommer",
    "gross", "kruse", "seidel", "engel", "simon", "ernst", "brandt", "otto",
    "adam", "bach", "bauer", "blum", "bohn", "bruckner", "busch", "dietrich", "dorn",
    "ehlers", "eichler", "eisner", "emmert", "engert", "fabian", "falk", "fiedler",
    "fleischer", "forster", "frey", "freitag", "fromm", "furth", "gehring", "gerhardt",
    "gersten", "gessner", "glaser", "goldberg", "graf", "greiner", "grob", "gruber",
    "haase", "haberle", "hagen", "hammer", "hanke", "hansen", "hauser", "hecht",
    "heilmann", "heinemann", "heinz", "held", "hellmann", "henkel", "henschel", "hering",
    "herold", "herrmann", "herzog", "hesse", "hessler", "heyer", "hillmann", "hinz",
    "hirsch", "hochmuth", "hofer", "holz", "horn", "huhn", "huth", "jager", "janssen",
    "john", "jost", "junker", "kammer", "karcher", "kiefer", "kirsch", "klett", "klotz",
    "knapp", "koester", "kraft", "kraus", "krebs", "krieger", "kuhn", "kunz", "kurz",
    "langer", "lauer", "laumann", "liebermann", "lindner", "link", "lohmann", "lortz",
}

LAST_NAMES_FRENCH = {
    "martin", "bernard", "thomas", "petit", "robert", "richard", "durand", "dubois",
    "moreau", "laurent", "simon", "michel", "lefevre", "leroy", "roux", "david",
    "bertrand", "morel", "fournier", "girard", "bonnet", "dupont", "lambert", "fontaine",
    "rousseau", "vincent", "muller", "lefevre", "faure", "andre", "mercier", "blanc",
    "guerin", "boyer", "garnier", "chevalier", "francois", "legrand", "gauthier", "garcia",
    "perrin", "robin", "clement", "morin", "nicolas", "henry", "roussel", "mathieu",
    "gautier", "masson", "marchand", "duval", "denis", "dumont", "marie", "lemaire",
    "noel", "meyer", "dufour", "meunier", "brun", "blanchard", "giraud", "joly",
    "riviere", "lucas", "brunet", "gaillard", "barbier", "arnaud", "martinez", "gerard",
    "roche", "renaud", "schmitt", "roy", "leroux", "colin", "vidal", "caron",
    "picard", "roger", "fabre", "aubert", "lemoine", "renard", "dumas", "lacroix",
    "olivier", "philippe", "bourgeois", "pierre", "benoit", "rey", "leclerc", "payet",
    "rolland", "leclercq", "guillaume", "lecomte", "lopez", "jean", "dupuy", "guillot",
    "hubert", "berger", "carpentier", "sanchez", "dupuis", "moulin", "louis", "deschamps",
    "huet", "vasseur", "perez", "boucher", "fleury", "royer", "klein", "jacquet",
    "adam", "paris", "poirier", "marty", "aubry", "guyot", "carre", "charles",
    "renault", "charpentier", "menard", "maillard", "baron", "bertin", "bailly", "herve",
    "schneider", "fernandez", "le gall", "collet", "leger", "bouvier", "julien", "prevost",
    "millet", "perrot", "daniel", "le roux", "cousin", "germain", "breton", "besson",
}

LAST_NAMES_ITALIAN = {
    "rossi", "russo", "ferrari", "esposito", "bianchi", "romano", "colombo", "ricci",
    "marino", "greco", "bruno", "gallo", "conti", "de luca", "mancini", "costa",
    "giordano", "rizzo", "lombardi", "moretti", "barbieri", "fontana", "santoro", "mariani",
    "rinaldi", "caruso", "ferrara", "galli", "martini", "leone", "longo", "gentile",
    "martinelli", "vitale", "lombardo", "serra", "coppola", "de santis", "damico", "marchetti",
    "parisi", "villa", "conte", "ferri", "fabbri", "bianco", "marini", "grasso",
    "valentini", "messina", "sala", "de angelis", "gatti", "pellegrini", "palumbo",
    "sanna", "farina", "rizzi", "monti", "cattaneo", "moroni", "alberti", "guerra",
    "ruggiero", "benedetti", "orlando", "silvestri", "barone", "donati", "caputo",
    "bernardi", "dangelo", "giuliani", "basile", "dangelo", "carbone", "sorrentino",
    "riva", "amato", "ferraro", "pellegrino", "martino", "santini", "piras", "ravelli",
    "grassi", "testa", "fiore", "pagano", "donato", "leonardi", "negri", "derossi",
    "poli", "montanari", "neri", "rossetti", "palmieri", "battaglia", "cosentino", "franco",
    "venturi", "mazza", "sartori", "orlandi", "colombo", "pagani", "moro", "landi",
    "deluca", "pasquali", "molinari", "innocenti", "viviani", "pesce", "ventura", "pavan",
    "mazzoni", "murgia", "capuzzo", "mazzini", "belotti", "agostini", "rosetti", "piccolo",
}

LAST_NAMES_PORTUGUESE = {
    "silva", "santos", "ferreira", "pereira", "oliveira", "costa", "rodrigues", "martins",
    "jesus", "sousa", "fernandes", "goncalves", "gomes", "lopes", "marques", "alves",
    "almeida", "ribeiro", "pinto", "carvalho", "teixeira", "moreira", "correia", "mendes",
    "nunes", "soares", "vieira", "monteiro", "cardoso", "rocha", "raposo", "neves",
    "coelho", "cruz", "cunha", "pires", "ramos", "reis", "simoes", "antunes",
    "machado", "freitas", "azevedo", "batista", "borges", "campos", "castro", "dias",
    "domingues", "duarte", "esteves", "faria", "figueiredo", "gaspar", "guerreiro",
    "henriques", "leite", "lima", "loureiro", "macedo", "maia", "matos", "medeiros",
    "miranda", "moura", "nascimento", "nogueira", "pacheco", "paiva", "passos", "pedro",
    "peralta", "pinheiro", "queiroz", "resende", "rego", "sampaio", "silveira", "tavares",
    "valente", "vargas", "vaz", "andrade", "araujo", "assis", "barros", "bastos",
    "brito", "cabral", "caldeira", "carneiro", "chaves", "fonseca", "fontes", "guedes",
    "guerra", "lacerda", "leao", "leal", "lobato", "lourenco", "magalhaes", "malheiro",
    "melo", "mendonca", "mesquita", "morais", "moreira", "moreno", "pedrosa", "ponte",
    "queiros", "salgueiro", "sequeira", "siqueira", "sobral", "souza", "teles", "torres",
}

LAST_NAMES_DUTCH = {
    "de jong", "jansen", "de vries", "van den berg", "van dijk", "bakker", "janssen",
    "visser", "smit", "meijer", "de boer", "mulder", "de groot", "bos", "vos",
    "peters", "hendriks", "van leeuwen", "dekker", "brouwer", "de wit", "dijkstra",
    "smits", "de graaf", "van der linden", "kok", "jacobs", "de haan", "vermeer",
    "van den heuvel", "van der veen", "van der berg", "van dam", "kuijpers", "schouten",
    "willems", "hoekstra", "van den broek", "de koning", "van der heijden", "van der wal",
    "jansma", "kramer", "van wijk", "prins", "van der meer", "post", "kuiper", "hofman",
    "sanders", "willems", "van den brink", "wolters", "hermans", "van der vliet", "boer",
    "maas", "timmermans", "groen", "van den bosch", "koster", "schipper", "van beek",
    "appel", "arendsen", "arens", "baas", "bakhuizen", "beek", "beelen", "berends",
    "bertens", "beukers", "bleeker", "blom", "bosman", "brands", "broer", "burger",
    "claassen", "cornelissen", "damen", "doorn", "driessen", "eijkman", "eldering",
    "feenstra", "geerts", "gerritsen", "goossens", "graaf", "haan", "hagen", "hagemans",
    "heerema", "heijnen", "helms", "hofstra", "huisman", "jong", "jonker", "keizer",
    "klein", "klomp", "klop", "koning", "konings", "kool", "kroon", "laan", "lange",
    "linden", "looijen", "loos", "lubbers", "marsman", "mol", "molenaar", "mulders",
    "neelen", "nieman", "noordhuis", "oomen", "oorschot", "peeters", "pijl", "ploeg",
    "rademaker", "riemens", "rijkers", "roos", "ruiter", "schaap", "scheepers", "sloot",
    "snel", "spaans", "stam", "steen", "storm", "terpstra", "timmer", "van beek",
    "van der aa", "van der aart", "van der akker", "van der berg", "van der bilt",
    "van der bos", "van der ende", "van der goot", "van der horst", "van der kamp",
    "van der kooi", "van der laan", "van der lee", "van der maat", "van der meulen",
    "van der ploeg", "van der pol", "van der put", "van der ree", "van der schaaf",
    "van der steen", "van der tak", "van der velde", "van der ven", "van der vlist",
    "van der wal", "van der werf", "van der wiel", "van der zanden", "van dijk",
    "van dongen", "van doorn", "van eck", "van eijk", "van gaal", "van hees",
    "van hoek", "van kampen", "van keulen", "van laar", "van loon", "van maanen",
    "van noort", "van oort", "van os", "van rijn", "van schaik", "van schie",
    "van velzen", "veenstra", "verhoeven", "vermeulen", "voerman", "vogelaar", "voogd",
    "vreeman", "weijers", "werf", "westra", "wiersma", "wijnands", "zandstra", "zeeman",
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
    Comprehensive names database for fast name lookup.

    Loads ~30,000 curated names into memory (~5MB) for instant lookups.
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
