#!/usr/bin/env python3
"""
Curated names organized by locale (ISO 639-1 language codes).

These are manually maintained. After editing, regenerate the unified file:
    python3 tools/ingest_popular_names.py --local
"""

# ============================================================================
# Curated First Names by Locale
# These are manually maintained. After editing, regenerate the unified file:
#   python3 tools/ingest_popular_names.py --local
# ============================================================================

FIRST_NAMES_EN = {
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

FIRST_NAMES_ES = {
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
    "alicia", "natalia", "julia", "clara", "valentina", "camila", "daniela", "sofia",
    "mariana", "fernanda", "guadalupe", "alejandra", "gabriela", "catalina", "valeria",
    "florencia", "martina", "agustina", "milagros", "abril", "julieta", "celeste",
    "carolina", "constanza", "esperanza", "ines", "lourdes", "magdalena", "manuela",
    "micaela", "miranda", "nadia", "paloma", "renata", "rosario", "soledad",
    "ximena", "zara", "bianca", "delfina", "emilia", "felicitas", "graciela",
    "helena", "irene", "josefina", "karen", "lorena", "macarena", "nerea",
}

FIRST_NAMES_DE = {
    # Male
    "alexander", "andreas", "bernd", "christian", "daniel", "david", "dennis",
    "dirk", "erik", "florian", "frank", "fritz", "georg", "hans", "heinrich",
    "helmut", "jan", "jochen", "johann", "jonas", "julian", "jurgen", "karl",
    "klaus", "lars", "lukas", "manfred", "markus", "matthias", "max", "michael",
    "moritz", "niklas", "oliver", "patrick", "paul", "peter", "philipp", "ralf",
    "rainer", "reinhard", "rolf", "sascha", "sebastian", "stefan", "stephan",
    "thomas", "tim", "tobias", "uwe", "walter", "werner", "wilhelm", "wolfgang",
    # Female
    "alexandra", "andrea", "angelika", "anja", "anna", "annette", "barbara",
    "birgit", "brigitte", "charlotte", "claudia", "dagmar", "daniela", "doris",
    "elke", "emma", "eva", "franziska", "gabriele", "gisela", "greta", "hanna",
    "heidi", "helga", "hildegard", "ines", "ingrid", "jana", "julia", "karin",
    "katharina", "kristin", "lena", "leonie", "lina", "lisa", "luise", "maja",
    "maria", "marie", "marlene", "martina", "melanie", "mia", "monika", "nadine",
    "petra", "renate", "sabine", "sandra", "sarah", "silke", "simone", "sofia",
    "sophie", "stefanie", "susanne", "svenja", "tanja", "ulrike", "ursula",
    "vanessa", "vera",
    # Swiss/Austrian
    "beat", "christoph", "dominik", "fabian", "felix", "gabriel", "hannes",
    "jakob", "laurenz", "linus", "lorenz", "rafael", "samuel", "simon", "valentin",
    "adrian", "bernhard", "bruno", "dieter", "erich", "gerhard", "gottfried",
    "gunter", "gustav", "horst", "konrad", "leopold", "ludwig", "norbert", "otto",
    "roland", "siegfried", "theodor", "ulrich",
    "adelheid", "anneliese", "astrid", "christa", "elfriede", "frieda",
    "gertrude", "hannelore", "irmgard", "lieselotte", "margarete", "rosemarie",
    "traude", "waltraud",
    # Additional German
    "benedikt", "clemens", "ferdinand", "gregor", "ignaz", "kilian", "xaver",
    "annika", "carina", "fiona", "jasmin", "jennifer", "katja", "kerstin",
    "larissa", "laura", "lea", "linda", "manuela", "marina", "natascha",
    "nicole", "sabrina", "stephanie", "tatiana", "theresa", "verena",
    # Modern German
    "elias", "emilia", "finn", "finja", "greta", "leon", "lilly", "louis",
    "mila", "noah", "sophie",
}

FIRST_NAMES_FR = {
    # Male
    "jean", "pierre", "louis", "jacques", "philippe", "francois", "alain",
    "michel", "bernard", "claude", "andre", "rene", "marcel", "georges", "paul",
    "henri", "yves", "nicolas", "antoine", "thierry", "christophe", "eric",
    "laurent", "olivier", "stephane", "frederic", "pascal", "sebastien", "guillaume",
    "vincent", "mathieu", "julien", "hugo", "maxime", "alexandre", "thomas",
    "clement", "lucas", "theo", "gabriel", "raphael", "arthur", "noah", "leo",
    "adam", "nathan", "ethan", "samuel", "liam", "jules", "victor", "romain",
    "damien", "fabien", "xavier", "cedric", "cyrille", "gauthier", "tanguy",
    "adrien", "quentin", "baptiste", "loic", "sylvain", "arnaud", "benoit",
    "emile", "fernand", "gaston", "gustave", "leon", "lucien", "marius",
    "raymond", "roger", "serge",
    # Female
    "marie", "jeanne", "francoise", "catherine", "isabelle", "monique", "sylvie",
    "nathalie", "christine", "anne", "brigitte", "nicole", "veronique", "sandrine",
    "sophie", "caroline", "julie", "camille", "emma", "lea", "chloe", "manon",
    "ines", "jade", "louise", "alice", "lina", "rose", "anna", "mila", "julia",
    "charlotte", "amandine", "aurelie", "celine", "claire", "delphine", "elodie",
    "florence", "laure", "lucie", "margaux", "marine", "mathilde", "pauline",
    "valerie", "virginie",
    # Belgian / Swiss French
    "brice", "gilles", "patrice", "didier", "fabrice", "herve", "lionel",
    "regis", "thibault", "axel", "bastien", "corentin", "mael", "titouan",
    "constance", "gaelle", "maelys", "oceane", "solene",
    # Additional French
    "auguste", "baptiste", "cesar", "edgar", "felix", "gregoire", "hugues",
    "isidore", "leonard", "martial", "narcisse", "prosper", "raoul", "theophile",
    "adelaide", "apolline", "bernadette", "colette", "edmee", "gisele",
    "henriette", "josephine", "madeleine", "odette", "simone", "therese",
    # Modern French
    "agathe", "capucine", "clementine", "eloise", "juliette", "leonie",
    "romane", "victoire",
}

FIRST_NAMES_ZH = {
    # Male - Common Chinese given names (romanized/Pinyin)
    "wei", "fang", "lei", "jun", "jian", "ming", "qiang", "jie", "tao", "hao",
    "gang", "ping", "chao", "kai", "lin", "bin", "long", "peng", "bo", "cheng",
    "yong", "feng", "guang", "zhi", "xin", "wen", "yi", "da", "hua", "shi",
    "hong", "guo", "liang", "xiang", "yu", "hai", "dong", "nan", "zhong", "de",
    "jia", "xiao", "sheng", "fu", "shan", "chang", "bao", "ting", "rui", "chen",
    # Female - Common Chinese given names
    "li", "xia", "yan", "fang", "jing", "ying", "na", "lan", "mei", "yun",
    "hui", "xiu", "qin", "ling", "hong", "min", "juan", "yue", "rong", "qing",
    "zhen", "xue", "dan", "lian", "fen", "ya", "yao", "lu", "ting", "jiao",
    "cui", "shan", "zhu", "ai", "man", "lian", "miao", "han", "xiao", "yun",
    # Modern Chinese given names (popular for young generation)
    "zihan", "yichen", "ziyu", "haoyu", "yuxuan", "minghao", "junhao", "haoran",
    "zixuan", "yufei", "xiaohan", "sihan", "yutong", "mengqi", "yuxi", "xinyi",
    "jiahao", "tianyu", "wenjie", "zhiyuan", "siqi", "yiran", "jiayi", "xiaoxiao",
    "ruoxi", "anqi", "wanyi", "yilin", "ziqi", "jiaxin", "yuanyuan", "lingling",
    "yiyi", "xuanxuan", "peipei", "wenya", "yuhan", "meiqi", "yiwen", "siyuan",
    # Cantonese/Hong Kong names
    "wai", "wing", "tsz", "cheuk", "lok", "yat", "hei", "hin", "chi", "shing",
    "kit", "man", "yin", "pak", "ka", "lai", "yee", "mei", "fai", "kam",
    # Taiwanese names (additional)
    "yuwen", "jiaxuan", "yuxin", "ziting", "yijun", "shihan", "yunxuan",
    "yiru", "peichen", "yating", "weichen", "junwei", "zhihao", "yikai",
    # Malay-Chinese names (common in Singapore/Malaysia)
    "ah", "beng", "choon", "eng", "gek", "hock", "keng", "leng", "seng", "teck",
    "wah", "yew", "kwok", "siew", "khim", "bee", "lian", "neo", "poh", "swee",
    "chee", "huat", "kah", "kok", "meng", "peng", "see", "tai", "wee", "yeow",
}

FIRST_NAMES_JA = {
    # Male - Traditional
    "akira", "daisuke", "haruki", "hideki", "hiroshi", "ichiro", "isamu", "jiro",
    "kenji", "koji", "makoto", "masao", "masashi", "noboru", "osamu", "ryota",
    "satoshi", "shin", "shingo", "shinji", "shun", "tadashi", "takashi", "takeshi",
    "takumi", "taro", "tetsuya", "tomohiro", "toru", "yasuhiro", "yoshio", "yuki",
    "yusuke", "yuji", "yukio", "wataru", "kazuki", "kenta", "kota", "naoki",
    "sho", "sota", "ryo", "ren", "daiki", "yuto", "kaito", "haruto", "minato",
    "souta", "ryusei", "hayato", "yuuma", "kouki", "taiga", "asahi", "riku",
    # Female - Traditional
    "yoko", "keiko", "sachiko", "hiroko", "michiko", "junko", "yoshiko", "tomoko",
    "ayako", "kazuko", "mariko", "noriko", "reiko", "misako", "masako", "kyoko",
    "takako", "chieko", "kumiko", "akiko", "haruko", "fumiko", "shizuko", "yasuko",
    "sumiko", "teruko", "toshiko", "atsuko", "ikuko", "kuniko", "mitsuko",
    # Female - Modern
    "sakura", "hana", "yui", "aoi", "rin", "mio", "saki", "mei", "haruka",
    "nana", "risa", "yuna", "mana", "erika", "asuka", "kaede", "sayaka",
    "minami", "ayaka", "chihiro", "honoka", "kana", "koharu", "nanami",
    "riko", "sora", "yuka", "akane", "hinata", "himari", "ema", "rin",
    "mei", "mitsuki", "miyu", "hina", "ichika", "tsumugi", "runa",
    # Common family-use names (both genders)
    "hikaru", "kaoru", "nao", "shinobu", "makoto",
    # Additional common names
    "ai", "ami", "aya", "chika", "eri", "haru", "kaho", "kanon", "maki",
    "mari", "miku", "natsuki", "rena", "rika", "rio", "sayo", "suzuki",
    "yuri", "ayumi", "megumi", "midori", "momoko", "naomi", "shiori",
}

FIRST_NAMES_KO = {
    # Male
    "minho", "jinho", "junhyuk", "hyunwoo", "seungjun", "dongwon", "jiwon",
    "minjun", "hajun", "siwoo", "juwon", "yejun", "dohyun", "hajin", "jihoon",
    "seojun", "eunwoo", "joonyoung", "sanghoon", "kyungho", "youngjin", "sunwoo",
    "taehyung", "jimin", "jungkook", "seokjin", "namjoon", "yoongi", "hoseok",
    "wonjin", "gunwoo", "sihyun", "hyunbin", "jaewon", "kyungmin", "dongha",
    "junho", "sungjin", "yongmin", "chanwoo", "jaehyun", "woosik", "doojoon",
    # Female
    "jiyeon", "soyeon", "minji", "yuna", "hayeon", "seoyeon", "chaeyoung",
    "dahyun", "sana", "nayeon", "tzuyu", "jeongyeon", "momo", "jihyo",
    "eunji", "subin", "jieun", "yerin", "haeun", "sieun", "jiwoo", "yujin",
    "soojin", "seoyun", "minseo", "hayoon", "jooeun", "soyoung", "nara",
    "yewon", "arin", "hyejin", "sunhee", "eunbi", "boram", "somin",
    # Common names (unisex)
    "youngho", "minsoo", "jisoo", "jiyoung", "minah", "sooyoung",
    # Common surname-derived given names
    "joon", "hyun", "min", "jin", "woo", "ho", "young", "kyung", "sung",
    "hee", "soo", "eun", "yeon", "ji", "ha",
    # Additional common Korean names
    "taemin", "jonghyun", "kibum", "chanyeol", "baekhyun", "sehun", "kai",
    "doyoung", "taeyong", "mark", "jisung", "haechan", "jeno", "renjun",
}

FIRST_NAMES_HI = {
    # Male
    "aarav", "vihaan", "aditya", "vivaan", "arjun", "reyansh", "mohammed",
    "sai", "arnav", "dhruv", "kabir", "ritvik", "anirudh", "atharva", "pranav",
    "rudra", "shaurya", "ansh", "krishna", "ishaan", "darsh", "veer", "ayaan",
    "virat", "sarthak", "parth", "harsh", "dev", "agastya", "aadhya",
    "rahul", "amit", "vijay", "raj", "suresh", "rajesh", "vikram", "sanjay",
    "deepak", "manoj", "ashok", "ramesh", "sunil", "arun", "dinesh", "vinod",
    "pramod", "anand", "rakesh", "ajay", "pradeep", "naresh", "rohit",
    "ravi", "mohan", "gopal", "kiran", "nitin", "pankaj", "sachin",
    "gaurav", "vishal", "manish", "abhishek", "sandeep", "mukesh",
    # Female
    "priya", "neha", "pooja", "sunita", "anita", "rekha", "kavita", "meena",
    "seema", "rita", "nisha", "rashmi", "geeta", "savita", "shobha", "sudha",
    "asha", "lata", "usha", "kamala", "lakshmi", "sita", "radha", "ganga",
    "maya", "pushpa", "sarla", "manju", "padma", "leela", "kusum", "kanta",
    "sarita", "shanti", "parvati", "aruna", "malini", "yamuna", "sharada",
    "aanya", "ananya", "aadhya", "aaradhya", "diya", "isha", "kiara",
    "myra", "navya", "pari", "riya", "saanvi", "sara", "tara", "zara",
    "anika", "avni", "gauri", "jaya", "kriti", "mahima", "nandini", "pallavi",
    "renuka", "shruti", "tanvi", "varsha",
    # South Indian
    "venkatesh", "srinivas", "ramana", "krishnamurthy", "subramaniam",
    "lakshman", "balasubramanian", "chandrasekhar", "raghunath",
    "meenakshi", "lalitha", "vasundhara", "jayalakshmi",
    # Bengali
    "subhash", "debashish", "aniket", "sourav", "partha", "dipankar",
    "sutapa", "moumita", "debjani", "anindita",
    # Punjabi
    "gurpreet", "harpreet", "jaspreet", "manpreet", "amarjeet",
    "simran", "navjot", "jasleen", "harleen",
}

FIRST_NAMES_AR = {
    # Male
    "mohammed", "ahmad", "ali", "hassan", "hussein", "omar", "khalid", "ibrahim",
    "mustafa", "abdulrahman", "abdullah", "tariq", "youssef", "nasser", "faisal",
    "salim", "hamza", "bilal", "zaid", "ammar", "adnan", "samir", "rashid",
    "jamal", "kareem", "waleed", "majid", "hani", "amin", "bashar",
    "ismail", "idris", "sulaiman", "dawud", "harun", "musa", "isa", "yusuf",
    "marwan", "riad", "bassam", "saad", "talal", "nabil", "naeem", "sami",
    "wael", "imad", "ziad", "fadel", "maher", "munir", "tamer", "ashraf",
    "badr", "fahad", "ghazi", "habib", "laith", "muhannad", "osman", "rafiq",
    "shadi", "wahid", "yasir", "zakariya", "anas", "ayoub", "hisham", "jihad",
    "mazen", "nizar", "qasim", "reda", "sultan", "tawfiq", "usama", "yaser",
    # Female
    "fatima", "aisha", "maryam", "khadija", "sara", "layla", "noor", "hana",
    "rania", "dina", "lina", "samira", "amina", "yasmin", "zahra", "salma",
    "dalal", "jamila", "lubna", "maha", "nawal", "reem", "suha", "wafa",
    "abeer", "eman", "ghada", "huda", "najla", "rabab", "sana", "tahani",
    "afaf", "bahia", "dunia", "farida", "hanadi", "inaam", "latifa",
    "mai", "nahla", "qamra", "raghad", "sabah", "thana", "umm", "widad",
    # Additional Arabic names
    "abed", "fouad", "ghassan", "haitham", "kamal", "mahmoud", "nadim",
    "ramzi", "safwan", "taleb", "wisam", "zuhair",
    "abla", "bushra", "deena", "elham", "faiza", "ghalia", "hayat", "inas",
    "jumana", "khulood", "lamia", "maysa", "nada", "rawia", "siham", "tamara",
}

FIRST_NAMES_RU = {
    # Male
    "alexander", "alexei", "andrei", "anton", "artem", "boris", "denis", "dmitri",
    "evgeni", "fyodor", "gennadi", "grigori", "igor", "ilya", "ivan", "kirill",
    "konstantin", "leonid", "maxim", "mikhail", "nikita", "nikolai", "oleg",
    "pavel", "roman", "sergei", "stanislav", "timur", "vadim", "valentin",
    "valeri", "vasili", "victor", "vitali", "vladimir", "yaroslav", "yuri",
    "pyotr", "georgi", "arseni", "bogdan", "daniil", "egor", "fedor",
    "gleb", "matvei", "ruslan", "semyon", "stepan", "taras", "zakhar",
    "anatoli", "arkadi", "efim", "gavriil", "innokenti", "lev", "mark",
    # Female
    "alexandra", "alina", "anastasia", "anna", "daria", "ekaterina", "elena",
    "evgenia", "galina", "irina", "kristina", "larisa", "lidia", "lyudmila",
    "maria", "marina", "nadezhda", "natalia", "nina", "olga", "oksana",
    "polina", "sofia", "svetlana", "tamara", "tatiana", "valentina", "vera",
    "victoria", "yulia", "zhanna", "zinaida",
    "alisa", "arina", "diana", "elizaveta", "kira", "ksenia", "milana",
    "varvara", "vasilisa", "veronika",
    # Slavic diminutives (commonly used as given names)
    "sasha", "misha", "kolya", "dima", "vanya", "volodya", "petya", "kostya",
    "grisha", "lyosha", "zhenya", "senya", "tanya", "katya", "masha",
    "natasha", "lena", "sveta", "nadya", "olya", "liza", "dasha", "anya",
    "sonya", "valya",
}

FIRST_NAMES_PT = {
    # Male
    "joao", "jose", "antonio", "francisco", "pedro", "paulo", "carlos", "lucas",
    "rafael", "gabriel", "daniel", "felipe", "guilherme", "bruno", "marcos",
    "rodrigo", "andre", "gustavo", "leonardo", "thiago", "matheus", "vinicius",
    "ricardo", "marcelo", "fernando", "jorge", "renato", "alexandre", "diogo",
    "tiago", "bernardo", "henrique", "enzo", "davi", "miguel", "arthur",
    "heitor", "samuel", "theo", "lorenzo", "levi", "benicio", "caio",
    "fabricio", "flavio", "geraldo", "helio", "jairo", "leandro", "mauro",
    "nelson", "otavio", "reginaldo", "sergio", "valdir", "wagner",
    # Female
    "maria", "ana", "patricia", "fernanda", "juliana", "adriana", "camila",
    "bruna", "amanda", "larissa", "leticia", "gabriela", "rafaela", "mariana",
    "beatriz", "carolina", "vanessa", "daniela", "priscila", "monique",
    "helena", "alice", "laura", "valentina", "sophia", "manuela", "luiza",
    "cecilia", "isadora", "lorena", "vitoria", "aurora", "clarice", "elisa",
    "flavia", "gisele", "ines", "joana", "katia", "luana", "miriam",
    "nadia", "raquel", "simone", "tatiana", "vera",
}

FIRST_NAMES_IT = {
    # Male
    "marco", "giuseppe", "giovanni", "antonio", "francesco", "andrea", "luca",
    "alessandro", "lorenzo", "matteo", "davide", "simone", "stefano", "roberto",
    "giorgio", "michele", "leonardo", "riccardo", "pietro", "filippo", "gabriele",
    "emanuele", "nicolo", "tommaso", "edoardo", "giacomo", "salvatore", "vincenzo",
    "raffaele", "pasquale", "domenico", "carlo", "mario", "luigi", "sergio",
    "massimo", "enrico", "fabio", "paolo", "claudio", "bruno", "guido",
    "alberto", "cesare", "dante", "elia", "federico", "gianluca", "hugo",
    "ignazio", "jacopo", "leone", "manolo", "nino", "orlando", "piero",
    "renato", "tiziano", "umberto", "valentino", "alfredo", "amedeo",
    "benedetto", "camillo", "dario", "emilio", "fabrizio", "gianfranco",
    "ivo", "lamberto", "maurizio", "nunzio", "ottavio", "primo", "remo",
    "silvio", "tullio", "ugo", "valerio",
    # Female
    "giulia", "francesca", "chiara", "sara", "valentina", "anna", "maria",
    "elena", "alessia", "silvia", "federica", "martina", "ilaria", "elisa",
    "roberta", "daniela", "paola", "monica", "lucia", "simona", "aurora",
    "sofia", "ginevra", "beatrice", "emma", "alice", "bianca", "greta",
    "vittoria", "camilla", "arianna", "caterina", "eleonora", "ludovica",
    "matilde", "noemi", "rebecca", "serena", "veronica",
    "adriana", "carlotta", "diana", "emanuela", "fiamma", "giorgia",
    "isabella", "laura", "margherita", "nicoletta", "ornella", "patrizia",
    "rosa", "stella", "teresa",
    # Additional Italian
    "claudio", "flavio", "franco", "gianni", "gino", "lino", "livio",
    "lucio", "marcello", "massimiliano", "mirko", "nicola", "rocco",
    "ruggero", "sandro", "vittorio",
    "allegra", "antonella", "carmela", "concetta", "cosima", "donatella",
    "elisabetta", "fabiana", "filomena", "grazia", "immacolata", "loretta",
    "maddalena", "marilena", "natascia", "oriana", "romina", "rossana",
    "sabrina", "tiziana",
}

FIRST_NAMES_NL = {
    # Male
    "jan", "pieter", "cornelis", "willem", "hendrik", "johannes", "gerrit",
    "jacobus", "dirk", "petrus", "theodorus", "adrianus", "antonius", "franciscus",
    "daan", "sem", "liam", "lucas", "bram", "noah", "finn", "levi", "luuk",
    "milan", "jesse", "jayden", "ruben", "max", "hugo", "thomas", "tim",
    "lars", "thijs", "stijn", "niels", "sander", "rick", "bas", "koen",
    "jeroen", "bart", "wouter", "maarten", "stefan", "michiel", "arjan",
    "joris", "floris", "cas", "gijs", "tijn", "mees", "jens",
    # Female
    "maria", "cornelia", "anna", "johanna", "elisabeth", "hendrika", "wilhelmina",
    "emma", "julia", "sophie", "lotte", "anna", "lisa", "eva", "sanne",
    "fleur", "iris", "anouk", "noor", "sarah", "britt", "naomi", "mila",
    "amber", "lieke", "anne", "roos", "fenna", "femke", "kim", "linda",
    "marieke", "nienke", "petra", "renate", "simone", "wendy",
    "demi", "evi", "jade", "luna", "nina", "olivia", "sara", "tessa", "yara",
    "isa", "lana", "lynn", "maud", "vera", "zoe",
    # Belgian Dutch
    "wout", "robbe", "xander", "nout", "lotte", "fien", "elle", "roos",
}

# ============================================================================
# Curated Last Names by Locale
# ============================================================================

LAST_NAMES_EN = {
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson",
    "anderson", "thomas", "taylor", "moore", "jackson", "martin", "lee",
    "perez", "thompson", "white", "harris", "sanchez", "clark", "ramirez",
    "lewis", "robinson", "walker", "young", "allen", "king", "wright",
    "scott", "torres", "nguyen", "hill", "flores", "green", "adams",
    "nelson", "baker", "hall", "rivera", "campbell", "mitchell", "carter",
    "roberts", "gomez", "phillips", "evans", "turner", "diaz", "parker",
    "cruz", "edwards", "collins", "reyes", "stewart", "morris", "morales",
    "murphy", "cook", "rogers", "gutierrez", "ortiz", "morgan", "cooper",
    "peterson", "bailey", "reed", "kelly", "howard", "ramos", "kim", "cox",
    "ward", "richardson", "watson", "brooks", "chavez", "wood", "james",
    "bennett", "gray", "mendoza", "ruiz", "hughes", "price", "alvarez",
    "castillo", "sanders", "patel", "myers", "long", "ross", "foster",
    "jimenez", "powell", "jenkins", "perry", "russell", "sullivan", "bell",
    "coleman", "butler", "henderson", "barnes", "gonzales", "fisher",
    "vasquez", "simmons", "griffin", "mcdonald", "hayes",
    # British
    "o'brien", "o'connor", "o'sullivan", "mccarthy", "macdonald",
    "mcdougall", "macleod", "campbell", "mackenzie", "sinclair",
    "fraser", "hamilton", "stewart", "wallace", "douglas", "graham",
    "robertson", "davidson", "johnston", "duncan", "ferguson", "cameron",
    "murray", "gordon", "ross", "reid", "crawford", "kerr",
    # Additional English
    "armstrong", "barber", "bishop", "blake", "bolton", "bowen", "brady",
    "brennan", "briggs", "bruce", "burke", "burns", "burton", "cannon",
    "carpenter", "casey", "chambers", "chandler", "chapman", "chase",
    "christensen", "clarkson", "cole", "conner", "cooke", "costa", "crane",
    "cross", "dalton", "dawson", "dean", "delaney", "donovan", "drake",
    "dunn", "elliott", "emerson", "erickson", "farmer", "finn", "fletcher",
    "ford", "fox", "gallagher", "garrett", "gibson", "goodwin", "grant",
    "hansen", "harding", "harper", "hart", "harvey", "hawkins", "haynes",
    "howell", "hunt", "hyde", "ingram", "irwin",
}

LAST_NAMES_ES = {
    "garcia", "rodriguez", "martinez", "lopez", "gonzalez", "hernandez",
    "perez", "sanchez", "ramirez", "torres", "flores", "rivera", "gomez",
    "diaz", "reyes", "morales", "jimenez", "ruiz", "alvarez", "mendoza",
    "castillo", "romero", "herrera", "medina", "aguilar", "vargas", "castro",
    "cruz", "ortiz", "gutierrez", "moreno", "munoz", "rojas", "navarro",
    "ramos", "dominguez", "molina", "vega", "silva", "velasquez", "contreras",
    "guerrero", "fernandez", "sandoval", "espinoza", "soto", "nunez", "figueroa",
    "delgado", "cardenas", "cabrera", "acosta", "fuentes", "leon", "valdez",
    "salazar", "campos", "maldonado", "estrada", "rios", "padilla", "lara",
    "trujillo", "miranda", "pineda", "bautista", "bravo", "camacho",
    "cervantes", "cisneros", "cordova", "de la cruz", "escobar", "gallegos",
    "ibarra", "jaramillo", "lugo", "marin", "nava", "ochoa", "palacios",
    "quintero", "rosales", "serrano", "tapia", "urbina", "valenzuela",
    "villanueva", "zamora", "arellano", "benitez", "bustamante", "carrillo",
    "de leon", "enriquez", "guzman", "hurtado", "lozano", "macias", "mejia",
    "montes", "orozco", "pena", "quiroz", "rangel", "salas", "trevino",
    "urdaneta", "vasquez", "zuniga",
}

LAST_NAMES_ZH = {
    "wang", "li", "zhang", "liu", "chen", "yang", "zhao", "huang", "zhou",
    "wu", "xu", "sun", "hu", "zhu", "gao", "lin", "he", "guo", "ma", "luo",
    "liang", "song", "zheng", "xie", "han", "tang", "feng", "deng", "cao",
    "peng", "zeng", "xiao", "tian", "dong", "pan", "yuan", "yu", "jiang",
    "cai", "jia", "wei", "du", "ye", "cheng", "su", "lv", "ding", "shen",
    "ren", "lu", "yao", "tan", "fan", "xia", "wan", "qian", "shi",
    # Additional Chinese surnames
    "an", "bai", "bian", "chang", "chi", "cui", "dai", "fang", "fu", "ge",
    "gong", "gu", "hao", "hou", "ji", "jiao", "jin", "kang", "kong", "lang",
    "lei", "lian", "meng", "miao", "min", "mu", "ni", "ning", "qin", "qiu",
    "shan", "shao", "tao", "wan", "wen", "xiang", "xin", "xiong", "yan",
    "yang", "yi", "yin", "you", "yue", "zhan", "zhong", "zou", "zuo",
    # Cantonese/HK surnames
    "chan", "cheung", "chow", "fung", "ho", "kwok", "lai", "lam", "lee",
    "leung", "ng", "tam", "tse", "wong", "yip", "yuen", "cheng", "hui",
    "lo", "mak", "siu", "tang", "tsang", "yeung",
    # Hokkien/Teochew surnames (Singapore/Malaysia)
    "tan", "lim", "ong", "koh", "teo", "goh", "chua", "sim", "ang", "tay",
    "foo", "heng", "low", "neo", "poh", "seah", "toh", "yeo",
}

LAST_NAMES_JA = {
    "sato", "suzuki", "takahashi", "tanaka", "watanabe", "ito", "yamamoto",
    "nakamura", "kobayashi", "kato", "yoshida", "yamada", "sasaki", "yamaguchi",
    "matsumoto", "inoue", "kimura", "hayashi", "shimizu", "yamazaki",
    "mori", "abe", "ikeda", "hashimoto", "yamashita", "ishikawa", "nakajima",
    "maeda", "fujita", "ogawa", "goto", "okada", "hasegawa", "murakami",
    "kondo", "ishii", "saito", "sakamoto", "endo", "aoki", "fujii", "nishimura",
    "fukuda", "ota", "miura", "fujiwara", "okamoto", "matsuda", "nakagawa",
    "harada", "ono", "tamura", "takeuchi", "kaneko", "wada", "nakayama",
    "ishida", "ueda", "morita", "hara", "hirata", "miyamoto",
    # Additional Japanese surnames
    "adachi", "arai", "asano", "baba", "chiba", "doi", "eguchi", "fuji",
    "hamada", "hattori", "higa", "honda", "hosokawa", "ichikawa", "iwata",
    "kawaguchi", "kawamoto", "kitamura", "kubo", "kuroda", "matsui",
    "minami", "miyazaki", "mochizuki", "nagata", "nagai", "nishida",
    "nomura", "oda", "oishi", "okazaki", "onishi", "osawa", "sakai",
    "sawada", "shibata", "sugimoto", "takeda", "terada", "tsuchida",
    "tsuda", "ueno", "umeda", "yamane", "yokoyama", "yoshimura",
}

LAST_NAMES_KO = {
    "kim", "lee", "park", "choi", "jung", "kang", "cho", "yoon", "jang",
    "lim", "han", "oh", "seo", "shin", "kwon", "hwang", "ahn", "song",
    "ryu", "hong", "jeon", "go", "bae", "moon", "yang", "ha", "nam",
    "shim", "noh", "kwak", "woo", "sung", "cha", "byun",
    # Additional Korean surnames
    "baek", "min", "yoo", "heo", "jin", "eom", "gu", "doh", "ban",
    "geum", "gi", "son", "won", "ye", "cheon", "gong", "ma", "sa",
    "tae", "tam", "yeom", "pi", "wang", "tang",
    # Romanization variants
    "yi", "rhee", "ri", "pak", "chung", "chang", "whang", "lyu",
}

LAST_NAMES_HI = {
    # North Indian
    "sharma", "verma", "singh", "kumar", "gupta", "jain", "agarwal", "mehta",
    "patel", "shah", "mishra", "pandey", "tiwari", "dubey", "srivastava",
    "rastogi", "saxena", "kapoor", "khanna", "malhotra", "arora", "bhatia",
    "chopra", "dhawan", "grover", "luthra", "mehra", "nanda", "oberoi",
    "sachdeva", "tandon", "vohra", "walia",
    # South Indian
    "reddy", "naidu", "rao", "pillai", "nair", "menon", "iyer", "iyengar",
    "krishnan", "subramaniam", "murugan", "anand", "venkatesh", "chandra",
    "sundaram", "raman", "swaminathan", "balasubramanian", "gopal", "natarajan",
    # Bengali
    "banerjee", "chatterjee", "mukherjee", "das", "bose", "ghosh", "sen",
    "roy", "dutta", "sarkar", "chakraborty", "bhattacharya", "ganguly", "mitra",
    "barua", "saha",
    # Marathi/Gujarati
    "desai", "modi", "trivedi", "bhatt", "dave", "parikh", "shukla",
    "thakkar", "dalal", "doshi", "gandhi", "johari", "kothari",
    # Sikh/Punjabi
    "sidhu", "bajwa", "brar", "dhillon", "gill", "grewal", "hundal",
    "kalra", "randhawa", "sandhu", "saini", "virdi",
    # Additional common Indian surnames
    "abraham", "alexander", "cherian", "george", "jacob", "john", "joseph",
    "kurian", "mathew", "philip", "thomas", "varghese",
    "deol", "hegde", "kamat", "kulkarni", "patil", "shetty",
}

LAST_NAMES_AR = {
    "al-ahmad", "al-ali", "al-bakr", "al-darwish", "al-fahad", "al-ghamdi",
    "al-habib", "al-harbi", "al-hassan", "al-hussein", "al-ibrahim",
    "al-jaber", "al-khalil", "al-mahmoud", "al-nasser", "al-omar",
    "al-qahtani", "al-rashid", "al-saleh", "al-shamsi", "al-tamimi",
    "al-yousef", "al-zahrani", "al-dosari", "al-malki", "al-otaibi",
    "al-shehri", "al-anazi", "al-mutairi", "al-subaie",
    # Without al- prefix
    "hassan", "hussein", "ibrahim", "khalil", "mahmoud", "mohammed",
    "nasser", "omar", "rashid", "saleh", "said",
    # North African
    "benali", "benmoussa", "bouaziz", "bousaid", "hadj", "haddad",
    "khelil", "mansouri", "medjdoub", "meziane", "rahmani", "saidi",
    "taleb", "ziani",
    # Levantine
    "abboud", "aoun", "boutros", "daher", "gemayel", "haddad", "karam",
    "khoury", "makdisi", "mouawad", "nasrallah", "sabbagh", "sleiman",
    "tannous", "youssef",
    # Egyptian
    "abdel-fattah", "abdel-nour", "el-masri", "el-sayed", "farouk",
    "hosni", "mubarak", "sadat", "shafik", "soliman",
    # Iraqi/Gulf
    "al-azzawi", "al-baghdadi", "al-dulaimi", "al-hakim", "al-janabi",
    "al-jubouri", "al-kazemi", "al-kubaisi", "al-saadi", "al-tikrit",
}

LAST_NAMES_RU = {
    "ivanov", "smirnov", "kuznetsov", "popov", "vasiliev", "petrov",
    "sokolov", "mikhailov", "novikov", "fyodorov", "morozov", "volkov",
    "alekseev", "lebedev", "semyonov", "egorov", "pavlov", "kozlov",
    "stepanov", "nikolaev", "orlov", "andreev", "makarov", "nikitin",
    "zakharov", "zaitsev", "solovyov", "borisov", "yakovlev", "grigoriev",
    "romanov", "vorobyov", "sergeev", "kuzmin", "frolov", "alexandrov",
    "dmitriev", "korolev", "gusev", "kiselev", "ilyin", "maximov",
    "polyakov", "sorokin", "vinogradov", "kovalev", "belov", "medvedev",
    "antonov", "tarasov", "zhukov", "baranov", "filippov", "komarov",
    "davydov", "belyaev", "gerasimov", "bogdanov", "osipov",
    # Ukrainian
    "kovalenko", "bondarenko", "shevchenko", "kravchenko", "tkachenko",
    "oliynyk", "lysenko", "melnyk", "savchenko", "marchenko",
    # Additional
    "abramov", "afanasiev", "belousov", "chernov", "demin", "efimov",
    "gavrilov", "ignatov", "kalinin", "logunov", "moiseev",
}

LAST_NAMES_DE = {
    "muller", "schmidt", "schneider", "fischer", "weber", "meyer", "wagner",
    "becker", "schulz", "hoffmann", "schafer", "koch", "bauer", "richter",
    "klein", "wolf", "schroder", "neumann", "schwarz", "zimmermann",
    "braun", "kruger", "hofmann", "hartmann", "lange", "schmitt", "werner",
    "schmitz", "krause", "meier", "lehmann", "schmid", "schulze", "maier",
    "kohler", "herrmann", "konig", "walter", "mayer", "huber", "kaiser",
    "fuchs", "peters", "lang", "scholz", "moller", "weiss", "jung",
    "hahn", "schubert", "vogel", "friedrich", "keller", "gunther", "frank",
    "berger", "winkler", "roth", "beck", "lorenz", "baumann", "franke",
    "albrecht", "schuster", "simon", "ludwig", "bohm", "winter",
    "kraus", "martin", "schumacher", "kroger", "schreiber", "brandt",
    "horn", "dietrich", "haas", "schumann", "vogt", "otto", "sommer",
    "stein", "jager", "grosse", "engel", "ernst", "kohl", "kraft",
    "bruckner", "lindner", "pfeiffer", "seidel", "stark", "strauss",
    "thiel", "unger", "wendt", "wulf", "blum", "ebert", "falk", "geiger",
    "hamann", "heinz", "kirchner", "kolb", "lenz", "menzel", "nikolaus",
    "opitz", "rauch", "seifert", "stahl", "trautmann", "voigt", "wirth",
}

LAST_NAMES_FR = {
    "martin", "bernard", "thomas", "petit", "robert", "richard", "durand",
    "dubois", "moreau", "laurent", "simon", "michel", "lefevre", "leroy",
    "roux", "david", "bertrand", "morel", "fournier", "girard", "bonnet",
    "dupont", "lambert", "fontaine", "rousseau", "vincent", "muller", "lefevre",
    "faure", "andre", "mercier", "blanc", "guerin", "boyer", "garnier",
    "chevalier", "francois", "legrand", "gauthier", "garcia", "perrin",
    "robin", "clement", "morin", "nicolas", "henry", "roussel", "mathieu",
    "gautier", "masson", "marchand", "duval", "denis", "dumont", "marie",
    "lemaire", "noel", "meyer", "dufour", "meunier", "brun", "blanchard",
    "giraud", "joly", "riviere", "lucas", "brunet", "gaillard", "barbier",
    "arnaud", "martinez", "gerard", "roche", "renault", "schmitt",
    # Belgian French
    "janssen", "peeters", "claes", "willems", "goossens", "maes",
    # Additional French surnames
    "adam", "aubert", "bastien", "beaumont", "bertin", "blondel", "bouvier",
    "breton", "caron", "charpentier", "collet", "cousin", "delorme",
    "descamps", "etienne", "ferrand", "fleury", "gagne", "guilbert",
    "hamon", "jacquet", "laporte", "leblanc", "leclerc", "lemoine", "maillard",
    "marion", "navarro", "oliveira", "pasquier", "perrot", "picard",
    "prevost", "renard", "royer", "sauvage", "tessier", "vidal",
}

LAST_NAMES_IT = {
    "rossi", "russo", "ferrari", "esposito", "bianchi", "romano", "colombo",
    "ricci", "marino", "greco", "bruno", "gallo", "conti", "de luca",
    "mancini", "costa", "giordano", "rizzo", "lombardi", "moretti",
    "barbieri", "fontana", "santoro", "mariani", "rinaldi", "caruso",
    "ferrara", "galli", "martini", "leone", "longo", "gentile", "martinelli",
    "vitale", "lombardo", "serra", "coppola", "de santis", "d'angelo",
    "marchetti", "parisi", "villa", "conte", "ferraro", "ferri", "fabbri",
    "bianco", "marini", "grasso", "valentini", "messina", "sala",
    # Additional Italian
    "amato", "basile", "bellini", "bernardi", "berti", "bonetti", "caputo",
    "cattaneo", "damico", "de rosa", "donati", "farina", "fiore", "grassi",
    "guerra", "lazzari", "mazza", "monti", "neri", "orlando", "pagano",
    "palmieri", "pellegrini", "piazza", "riva", "ruggiero", "sanna",
    "silvestri", "sorrentino", "testa", "vitali",
}

LAST_NAMES_PT = {
    "silva", "santos", "oliveira", "souza", "pereira", "costa", "rodrigues",
    "almeida", "nascimento", "lima", "araujo", "fernandes", "carvalho",
    "gomes", "martins", "rocha", "ribeiro", "alves", "monteiro", "mendes",
    "barros", "freitas", "barbosa", "pinto", "moura", "cavalcanti",
    "cardoso", "vieira", "correia", "cunha", "dias", "teixeira", "campos",
    "nunes", "soares", "moreira", "batista", "lopes", "marques", "machado",
    "melo", "ferreira", "azevedo", "borges", "castro", "fonseca",
    # Additional Portuguese surnames
    "abreu", "aguiar", "andrade", "antunes", "baptista", "braga", "cabral",
    "carneiro", "coelho", "domingues", "duarte", "esteves", "faria",
    "figueiredo", "gaspar", "henriques", "leal", "magalhaes", "matos",
    "miranda", "morais", "nogueira", "pacheco", "pires", "ramos",
    "reis", "sa", "sampaio", "simoes", "tavares", "vaz",
}

LAST_NAMES_NL = {
    "de jong", "jansen", "de vries", "van den berg", "van dijk", "bakker",
    "janssen", "visser", "smit", "meijer", "de boer", "mulder", "de groot",
    "bos", "vos", "peters", "hendriks", "van leeuwen", "dekker", "brouwer",
    "de wit", "dijkstra", "smeets", "de graaf", "van der linden", "kok",
    "jacobs", "de haan", "vermeer", "van den heuvel", "van der veen",
    "van der berg", "van dam", "kuijpers", "schouten", "willems", "hoekstra",
    "van den broek", "de koning", "van der heijden", "van der wal",
    "jansma", "kramer", "van wijk", "prins", "van der meer", "post",
    "kuiper", "hofman", "sanders", "willems", "van den brink", "wolters",
    "hermans", "van der vliet", "boer", "maas", "timmermans", "groen",
    "van den bosch", "koster", "schipper", "van beek",
    "smits",
    "appel", "arendsen", "arens", "baas", "bakhuizen", "beek", "beelen",
    "berends", "bertens", "beukers", "bleeker", "blom", "bosman", "brands",
    "broer", "burger", "claassen", "cornelissen", "damen", "doorn",
    "driessen", "eijkman", "eldering", "feenstra", "geerts", "gerritsen",
    "goossens", "graaf", "haan", "hagen", "hagemans", "heerema", "heijnen",
    "helms", "hofstra", "huisman", "jong", "jonker", "keizer", "klein",
    "klomp", "klop", "koning", "konings", "kool", "kroon", "laan", "lange",
    "linden", "looijen", "loos", "lubbers", "marsman", "mol", "molenaar",
    "mulders", "neelen", "nieman", "noordhuis", "oomen", "oorschot",
    "peeters", "pijl", "ploeg", "rademaker", "riemens", "rijkers", "roos",
    "ruiter", "schaap", "scheepers", "sloot", "snel", "spaans", "stam",
    "steen", "storm", "terpstra", "timmer", "van beek", "van der aa",
    "van der aart", "van der akker", "van der berg", "van der bilt",
    "van der bos", "van der ende", "van der goot", "van der horst",
    "van der kamp", "van der kooi", "van der laan", "van der lee",
    "van der maat", "van der meulen", "van der ploeg", "van der pol",
    "van der put", "van der ree", "van der schaaf", "van der steen",
    "van der tak", "van der velde", "van der ven", "van der vlist",
    "van der wal", "van der werf", "van der wiel", "van der zanden",
    "van dijk", "van dongen", "van doorn", "van eck", "van eijk",
    "van gaal", "van hees", "van hoek", "van kampen", "van keulen",
    "van laar", "van loon", "van maanen", "van noort", "van oort",
    "van os", "van rijn", "van schaik", "van schie", "van velzen",
    "veenstra", "verhoeven", "vermeulen", "voerman", "vogelaar", "voogd",
    "vreeman", "weijers", "werf", "westra", "wiersma", "wijnands",
    "zandstra", "zeeman",
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
# Curated names mapped to locale codes (for ingestion script)
# ============================================================================

CURATED_FIRST_NAMES = {
    "en": FIRST_NAMES_EN, "es": FIRST_NAMES_ES, "de": FIRST_NAMES_DE,
    "fr": FIRST_NAMES_FR, "zh": FIRST_NAMES_ZH, "ja": FIRST_NAMES_JA,
    "ko": FIRST_NAMES_KO, "hi": FIRST_NAMES_HI, "ar": FIRST_NAMES_AR,
    "ru": FIRST_NAMES_RU, "pt": FIRST_NAMES_PT, "it": FIRST_NAMES_IT,
    "nl": FIRST_NAMES_NL,
}

CURATED_LAST_NAMES = {
    "en": LAST_NAMES_EN, "es": LAST_NAMES_ES, "zh": LAST_NAMES_ZH,
    "ja": LAST_NAMES_JA, "ko": LAST_NAMES_KO, "hi": LAST_NAMES_HI,
    "ar": LAST_NAMES_AR, "ru": LAST_NAMES_RU, "de": LAST_NAMES_DE,
    "fr": LAST_NAMES_FR, "it": LAST_NAMES_IT, "pt": LAST_NAMES_PT,
    "nl": LAST_NAMES_NL,
}
