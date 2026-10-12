# SPDX-License-Identifier: Apache-2.0
"""Calibration supplement (v3): instruction following, chat, writing and multilingual conversations, the domains our
first calibration set was thin on (prompts.py: 42% code tokens, chat 4.5%, writing 1.2%, multilingual 5%).

Why: build v2.1 lost IFEval prompt-strict against native (86.9 vs 89.5, case control worst) while Mia's 2.9 bpw pack
kept it; the bit allocation weighted experts by a code-heavy panel. This set gives the allocator the experts that
format-, case- and language-constrained replies use.

Held out from IFEval by construction: every topic and every constraint sentence below is written for this file, no
IFEval prompt is used, and where IFEval has a distinctive convention (its fixed yes / no answer strings, its six-
asterisk separator, its double-angle-bracket title) this set uses a different one. check_overlap() reports n-gram
overlap with a prompt list (run it against IFEval's before generating). Seeded and deterministic, like prompts.py;
every item lands in the "cal" split (the published KLD panel is unchanged).

  python3 prompts_instruct.py [ifeval_prompts.json]      counts, and the overlap report when a prompt list is given
"""
from __future__ import annotations

import random
import re

SEED = 20261010

# (task, keywords that fit it: lowercase, so they combine with any case constraint)
TASKS = [
    ("Write a short guide to repotting a houseplant that has outgrown its container.", ["roots", "soil", "drainage"]),
    ("Explain how a bicycle derailleur moves the chain between gears.", ["chain", "cable", "sprocket"]),
    ("Describe a relaxed two-day visit to Porto for someone who loves food.", ["river", "pastry", "tram"]),
    ("Write a product announcement for a solar-powered phone charger.", ["battery", "sunlight", "portable"]),
    ("Give advice to a new manager running their first one-on-one meetings.", ["listen", "goals", "feedback"]),
    ("Write a short story about a night-shift baker who finds a stray dog.", ["flour", "dawn", "oven"]),
    ("Explain why the sky turns red and orange at sunset.", ["scattering", "wavelength", "atmosphere"]),
    ("Write a notice from a town council to residents about a new recycling schedule.", ["bins", "schedule", "plastic"]),
    ("Summarize the pros and cons of electric scooters for city commuting.", ["battery", "traffic", "safety"]),
    ("Write a pitch for a mobile app that lets neighbors swap books.", ["library", "community", "swap"]),
    ("Explain the castling move in chess to a beginner.", ["king", "rook", "check"]),
    ("Write a blog introduction about learning to cook at forty.", ["kitchen", "recipe", "patience"]),
    ("Describe the water cycle for a classroom poster.", ["evaporation", "clouds", "rain"]),
    ("Write a farewell message to a colleague who is moving abroad.", ["journey", "team", "memories"]),
    ("Explain how compound interest works, with one small example.", ["principal", "rate", "growth"]),
    ("Review an imaginary science fiction film about terraforming Mars.", ["dust", "colony", "oxygen"]),
    ("Give tips for photographing the night sky with a phone.", ["exposure", "tripod", "stars"]),
    ("Write a job posting for a part-time bicycle mechanic.", ["repair", "customers", "tools"]),
    ("Explain what a credit score is and how a person can raise theirs.", ["payments", "debt", "history"]),
    ("Write a poem about an empty train station at midnight.", ["platform", "lantern", "whistle"]),
    ("Describe how honeybees tell each other where flowers are.", ["dance", "hive", "nectar"]),
    ("Write instructions for brewing pour-over coffee at home.", ["grind", "bloom", "kettle"]),
    ("Argue for and against a four-day work week.", ["productivity", "rest", "wages"]),
    ("Write a welcome note for new members of a weekend hiking club.", ["trail", "boots", "safety"]),
    ("Explain how vaccines prepare the immune system.", ["antibodies", "cells", "memory"]),
    ("Write a short fable about a tortoise who learns to ask for help.", ["shell", "river", "friend"]),
    ("Describe how the printing press changed the spread of ideas.", ["ink", "books", "type"]),
    ("Give a packing list for a week of camping beside a lake.", ["tent", "water", "stove"]),
    ("Explain how noise-cancelling headphones work.", ["microphone", "wave", "phase"]),
    ("Write an opening speech for a school science fair.", ["curiosity", "experiment", "future"]),
    ("Write a complaint email to an airline about a lost suitcase.", ["suitcase", "flight", "refund"]),
    ("Explain why some deserts are cold.", ["altitude", "rain", "temperature"]),
    ("Describe a city of the future where private cars are banned.", ["bicycles", "parks", "quiet"]),
    ("Write a guide to growing vegetables on a small balcony.", ["pots", "sunlight", "tomatoes"]),
    ("Explain how a refrigerator keeps food cold.", ["compressor", "refrigerant", "heat"]),
    ("Write a thank-you note to a teacher who made math fun.", ["patience", "numbers", "grateful"]),
    ("Summarize the plot of an invented mystery novel set on a cruise ship.", ["detective", "cabin", "storm"]),
    ("Give advice to someone moving to a cold climate for the first time.", ["layers", "heating", "snow"]),
    ("Explain what inflation is and why central banks watch it.", ["prices", "interest", "money"]),
    ("Write a dialogue between a robot and a gardener arguing about weeds.", ["sensor", "roots", "weeds"]),
    ("Describe the life cycle of a monarch butterfly.", ["caterpillar", "milkweed", "migration"]),
    ("Write a marketing email for a neighborhood yoga studio's open day.", ["stretch", "breathing", "class"]),
    ("Explain how to change a flat tire on a car.", ["jack", "spare", "bolts"]),
    ("Write an encouraging message for a friend training for a marathon.", ["miles", "pace", "finish"]),
    ("Describe the benefits of learning a musical instrument as an adult.", ["practice", "memory", "joy"]),
    ("Write a short horror story set in an empty shopping mall after closing time.", ["escalator", "lights", "echo"]),
    ("Explain how seats are allocated after a parliamentary election.", ["ballots", "seats", "votes"]),
    ("Give a recipe for a simple lentil soup.", ["onion", "cumin", "simmer"]),
    ("Write a press release about a small town opening its first public library.", ["books", "mayor", "readers"]),
    ("Explain the main theories about why cats purr.", ["vibration", "comfort", "healing"]),
    ("Write advice for a teenager starting a first part-time job.", ["schedule", "manager", "savings"]),
    ("Plan a rainy-day at home for a family of four.", ["games", "blankets", "snacks"]),
    ("Explain what a black hole is.", ["gravity", "light", "horizon"]),
    ("Write a short biography of an invented inventor who built a clockwork bird.", ["gears", "workshop", "flight"]),
    ("Compare tea and coffee as morning drinks.", ["caffeine", "flavor", "ritual"]),
    ("Write a set of house rules for a shared apartment.", ["dishes", "guests", "quiet"]),
    ("Explain how the moon causes tides.", ["gravity", "ocean", "orbit"]),
    ("Write a toast for a couple's tenth wedding anniversary.", ["decade", "laughter", "love"]),
    ("Give tips for making a small bedroom feel bigger.", ["mirrors", "light", "storage"]),
    ("Explain why recycling aluminum cans saves so much energy.", ["energy", "metal", "smelting"]),
    ("Write a scene where two strangers share an umbrella at a bus stop.", ["rain", "bus", "conversation"]),
    ("Describe a day in the life of a beekeeper.", ["hives", "smoke", "honey"]),
    ("Write instructions for teaching a puppy to sit.", ["treat", "command", "reward"]),
    ("Explain how a 3D printer builds an object.", ["layers", "filament", "nozzle"]),
    ("Write a short essay on why public parks matter to cities.", ["trees", "community", "health"]),
    ("Give advice for a job interview held over video call.", ["camera", "lighting", "questions"]),
    ("Write a postcard from a traveler visiting Iceland in winter.", ["glaciers", "geysers", "wind"]),
    ("Explain the difference between a virus and a bacterium.", ["cells", "antibiotics", "infection"]),
    ("Write an apology to a friend for missing their birthday dinner.", ["sorry", "dinner", "friendship"]),
    ("Describe how to organize a neighborhood clean-up day.", ["volunteers", "gloves", "bags"]),
    ("Write a short guide to keeping a sourdough starter alive.", ["flour", "jar", "feeding"]),
    ("Explain how GPS figures out where a phone is.", ["satellites", "signals", "time"]),
    ("Write a product review of an imaginary pair of waterproof hiking boots.", ["comfort", "grip", "laces"]),
    ("Describe what makes a good team meeting.", ["agenda", "decisions", "time"]),
    ("Write a short myth explaining why the sea is salty.", ["giant", "salt", "waves"]),
    ("Explain how to read a topographic map.", ["contour", "elevation", "scale"]),
]

YES_NO = [
    "Is it a good idea to learn two new languages at the same time?",
    "Should a small bakery accept payments in cryptocurrency?",
    "Would a mid-sized city benefit from free public buses?",
    "Can a houseplant live in a room with no windows?",
    "Should a beginner buy a used bicycle rather than a new one?",
    "Is a paper journal still worth keeping when phones have note apps?",
    "Would running every single day be good for most people?",
    "Is it reasonable to quit a stable job to start a food truck?",
]
CHOICES = [("Absolutely.", "Not at all.", "It depends."), ("Yes, clearly.", "No, clearly not.", "Hard to say."),
           ("I would say yes.", "I would say no.", "I am not sure.")]

LANGS = ["Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Korean", "Chinese", "Thai", "Vietnamese",
         "Hindi", "Russian", "Indonesian", "Turkish", "Polish", "Dutch"]
END_PHRASES = ["That is all for now.", "Let me know if anything is unclear.", "Thanks for reading.",
               "Hope this helps you get started.", "Until next time."]
FORBID = ["very", "really", "basically", "simply", "just", "actually", "great", "important"]
LETTERS = "bcdfgkmprstw"


def _constraints(rng: random.Random, kws: list[str]):
    """(name, tags, text) candidates for one task; two constraints conflict when they share a tag or a CONFLICT pair."""
    n = rng.randint
    k = rng.choice(kws)
    others = [w for w in kws if w != k]
    return [
        ("caps_all", {"case"}, rng.choice([
            "Write your whole reply in English using capital letters only, every word of it.",
            "Use uppercase letters for the entire answer, from the first word to the last.",
            "Every letter of your response has to be a capital letter. No lowercase at all.",
            "Reply in ALL CAPS, the whole way through, headings and lists included."])),
        ("lower_all", {"case"}, rng.choice([
            "Write the entire reply in lowercase letters. Not a single capital letter anywhere, not even at the start "
            "of sentences or in names.",
            "keep every letter lowercase in your answer, including the first word of each sentence.",
            "Use only lowercase letters throughout your response; capitals are not allowed."])),
        ("caps_words", {"case"}, rng.choice([
            f"Stress at least {n(3, 10)} key words by writing each of them fully in capital letters.",
            f"Use no more than {n(2, 5)} words written entirely in capitals.",
            f"Somewhere in the text, put {n(4, 12)} or more words in ALL CAPS for emphasis."])),
        ("no_comma", {"punct"}, rng.choice([
            "Do not use a single comma anywhere in your reply.",
            "Avoid commas completely; use other punctuation instead.",
            "Your answer must contain zero commas."])),
        ("words_max", {"length"}, f"Keep the whole answer under {rng.choice([40, 60, 80, 100, 150])} words."),
        ("words_min", {"length"}, f"Make the answer at least {rng.choice([250, 300, 400])} words long."),
        ("sentences", {"length"}, rng.choice([f"Use exactly {n(3, 8)} sentences.",
                                              f"Answer in fewer than {n(4, 9)} sentences."])),
        ("paragraphs", {"length", "structure"},
         f"Write exactly {n(2, 5)} paragraphs, separated from each other by a line containing only ***."),
        ("nth_word", {"length", "structure"}, (lambda p: (
            f"Write {p} paragraphs separated by blank lines, and begin paragraph {n(2, p)} with the word "
            f"\"{rng.choice(['meanwhile', 'still', 'finally', 'today', 'besides'])}\"."))(n(3, 5))),
        ("bullets", {"structure"}, rng.choice([
            f"Present the answer as exactly {n(3, 7)} bullet points, each starting with \"- \".",
            f"Use a markdown list of exactly {n(3, 6)} items and nothing else."])),
        ("highlight", {"highlight"}, f"Highlight at least {n(2, 4)} phrases with markdown italics, like *this*."),
        ("sections", {"structure"}, (lambda m, w: (
            f"Split the answer into {m} sections and start each with \"{w} 1\", \"{w} 2\" and so on."))(
            n(2, 4), rng.choice(["Part", "Step", "Chapter", "Point"]))),
        ("json", {"json", "structure"}, rng.choice([
            "Format the entire output as valid JSON.",
            "Return only a JSON object; you may use markdown code fences around it."])),
        ("title", {"title"}, rng.choice([
            "Open with a title set inside double curly braces, for example {{a title}}.",
            "Give the piece a title on the first line, inside square brackets like [The Title], then a blank line."])),
        ("placeholders", {"placeholder"},
         f"Include at least {n(2, 4)} placeholders in angle brackets that a reader would fill in, such as <name>."),
        ("postscript", {"ending"}, rng.choice(["Close with an afterthought line that opens with P.S.",
                                               "Finish with a short note beginning with \"P.S.\""])),
        ("end_phrase", {"ending"},
         f"End your reply with the exact sentence \"{rng.choice(END_PHRASES)}\" and write nothing after it."),
        ("quote_wrap", {"wrap"}, "Put your entire reply inside one pair of double quotation marks."),
        ("keywords", {"keyword"}, f"Make sure the words \"{others[0]}\" and \"{others[1]}\" both appear."),
        ("keyword_freq", {"keyword"}, rng.choice([f"Use the word \"{k}\" at least {n(3, 5)} times.",
                                                  f"Use the word \"{k}\" no more than once."])),
        ("forbidden", {"keyword"}, (lambda a, b: f"Do not use the words \"{a}\" or \"{b}\".")(*rng.sample(FORBID, 2))),
        ("letter_freq", {"letter"},
         f"The letter \"{rng.choice(LETTERS)}\" should appear at least {n(8, 20)} times in your answer."),
        ("language", {"language"}, f"Write the whole reply in {rng.choice(LANGS)} and never switch to English."),
        ("repeat", {"repeat"}, "First repeat my request word for word, without changing anything, then give your answer."),
        ("two", {"two"}, "Give two different versions of the answer, separated by a line of six hyphens: ------"),
    ]


CONFLICT = [({"json"}, {"structure", "wrap", "ending", "two", "title", "placeholder", "highlight", "length"}),
            ({"language"}, {"case", "keyword", "letter"}),
            ({"two"}, {"structure", "wrap", "ending", "repeat", "length"}),
            ({"repeat"}, {"wrap", "case"}),
            ({"wrap"}, {"ending", "title"})]


def _compatible(chosen, cand):
    for _, tags, _ in chosen:
        if tags & cand[1]:
            return False
        for a, b in CONFLICT:
            if (tags & a and cand[1] & b) or (tags & b and cand[1] & a):
                return False
    return True


# weights of constraint names when drawing: case control is where build v2.1 lost, so it is drawn most often
WEIGHT = {"caps_all": 6, "lower_all": 5, "caps_words": 4, "language": 3}

CHAT = [
    "My team keeps missing sprint goals. What are three things I could try next sprint?",
    "I want to read more but I fall asleep after two pages. Any ideas?",
    "How do I politely leave a group chat that has become too noisy?",
    "What is a reasonable weekly budget for groceries for two people who cook most meals?",
    "Can you help me plan a surprise picnic for my partner on a tight budget?",
    "How should I prepare my car for a long road trip in summer?",
    "I get anxious before phone calls. How can I make them easier?",
    "What are some low-cost hobbies that get me off screens in the evening?",
    "How do I choose between a laptop with more memory and one with a better screen?",
    "My neighbor's dog barks all day. How do I bring it up without starting a feud?",
    "What should I know before adopting a senior cat?",
    "Help me think through whether to take a gap year before university.",
    "How do I keep basil alive on a windowsill?",
    "I keep losing my keys. Give me a system that actually works.",
    "What's a good way to learn to touch type as an adult?",
    "How can I make my small kitchen easier to cook in?",
    "I need a gift idea for a coworker who likes hiking and board games.",
    "How do I start running again after a knee injury, carefully?",
    "What are signs that a job posting might be a scam?",
    "Give me a simple weekly cleaning schedule for a two-bedroom flat.",
    "How do I explain to my parents what I do as a data engineer?",
    "How do I ask for feedback from my manager without sounding insecure?",
    "What should a first-aid kit for a family car contain?",
    "How do I stop overthinking messages before I send them?",
    "What is a good routine for studying a new language for 30 minutes a day?",
    "I'm hosting dinner for six and one guest is vegan. What should I cook?",
    "How do I tell if a used smartphone has been well looked after?",
    "Help me write a short message to reconnect with an old friend.",
    "What should I consider before moving to a city where I know nobody?",
    "How can I sleep better when my schedule changes every week?",
    "What's the easiest way to start saving for retirement in my twenties?",
    "How do I handle a client who keeps changing requirements?",
    "Recommend a few board games for a group that does not usually play games.",
    "My houseplants get fungus gnats. How do I get rid of them?",
    "How do I make a toddler's bedtime less of a battle?",
    "What questions should I ask a landlord before signing a lease?",
    "How do I get comfortable speaking up in meetings?",
    "Suggest a cheap and healthy lunch I can prepare for the whole work week.",
    "How do I decide which streaming subscriptions to cancel?",
    "My bike chain keeps slipping. What should I check first?",
    "What should I pack in a carry-on for a three-day work trip?",
    "How do I teach my kid to ride a bike without training wheels?",
    "How can I be a better listener in arguments with my partner?",
    "Help me choose a name for a weekend pottery class.",
    "What do I need to start bird watching?",
    "How can I make my resume stand out for an entry-level design job?",
    "What should I do in the first week at a new job?",
    "How do I keep a long-distance friendship going?",
]

WRITING = [
    "Write a two-paragraph opening for a fantasy novel set in a city built on the back of a giant turtle.",
    "Write a wedding invitation for an outdoor ceremony in an orchard.",
    "Rewrite this sentence in five different tones (formal, playful, angry, poetic, neutral): 'The meeting moved to Friday.'",
    "Write a eulogy for a beloved family dog, warm and a little funny.",
    "Write the text for a safety sign at a swimming pool. Use capital letters for the warnings.",
    "Convert this headline to title case and explain any rules you applied: 'the quiet return of the city tram'.",
    "Write a short newsletter announcing a community garden's harvest festival.",
    "Write a bedtime story about a cloud that is afraid of heights.",
    "Write a LinkedIn post celebrating a team that shipped a hard project, without sounding corporate.",
    "Write a letter to your future self to be opened in ten years.",
    "Write a short sonnet about a bicycle commute in the rain.",
    "Write an obituary-style piece for a retired lighthouse, in a gentle tone.",
    "Write a short screenplay scene: a chef discovers the restaurant critic is their estranged sister.",
    "Draft a polite but firm message asking a client to pay an overdue invoice.",
    "Write a recipe introduction for a family stew, like a food blogger, but keep it under 150 words.",
    "Write a free-verse poem about the first snowfall in a city.",
    "Write an advertisement script for a radio spot selling handmade candles.",
    "Write a short speech a captain gives before a championship game.",
    "Write a children's riddle poem whose answer is 'a shadow'.",
    "Write a formal complaint to a city about potholes on a residential street.",
    "Turn these bullet points into a friendly email: office closed Monday, pizza Friday, new parking rules.",
    "Write a monologue of an old clock tower watching the town change.",
    "Write a short travel essay about getting lost in a market in Marrakesh.",
    "Write a product description for a reusable water bottle that tracks how much you drink.",
    "Write a cover letter paragraph for a librarian applying to a university library.",
    "Write a motivational poster text for a school gym, in uppercase.",
    "Write a short ghost story told by a museum night guard.",
    "Write a thank-you speech for an award given to a volunteer firefighter.",
    "Write an apology statement from a bakery that sold cakes with the wrong labels.",
    "Write a short dialogue between two astronauts arguing about what to name a new planet.",
    "Rewrite this paragraph so a ten-year-old can follow it: 'Photosynthesis converts light energy into chemical "
    "energy stored in glucose, releasing oxygen as a by-product.'",
    "Write a song chorus about leaving a small hometown.",
    "Write a short op-ed arguing that every school should teach cooking.",
    "Write a set of five funny but useful tips for first-time campers.",
    "Write a postcard from a cat on vacation to the dog who stayed home.",
    "Write a short adventure story in second person set in a jungle temple.",
    "Write an announcement, all in capital letters, that a train service is cancelled because of snow.",
    "Write the welcome text for a museum exhibition about the history of maps.",
    "Write a haiku for each of the four seasons.",
    "Write a personal statement opening for a nursing school application.",
    "Write a fairy tale where the dragon is the one who needs rescuing.",
    "Write a mock weather forecast for a planet with two suns.",
    "Write a heartfelt letter from a grandparent to a grandchild starting university.",
    "Write a satirical news brief about a town that elects a goat as mayor.",
    "Write meeting minutes for a fictional neighborhood association meeting about a new playground.",
    "Write a short scene told only through text messages between two siblings planning a party.",
    "Write a persuasive paragraph for keeping the town library open on Sundays.",
    "Write instructions for a board game you invent, in under 200 words.",
]

MULTI = [
    ("es", "¿Qué consejos darías a alguien que empieza a correr por primera vez?"),
    ("es", "Escribe un cuento corto sobre un pescador y una tormenta inesperada."),
    ("es", "Resume las ventajas y desventajas de trabajar desde casa en una lista breve."),
    ("es", "Redacta un mensaje de felicitación para una amiga que acaba de graduarse."),
    ("fr", "Peux-tu m'expliquer simplement comment fonctionne une éolienne ?"),
    ("fr", "Écris une lettre à un voisin pour l'inviter à une fête de quartier."),
    ("fr", "Donne-moi trois idées de repas rapides pour un soir de semaine."),
    ("fr", "Raconte une petite histoire sur un chat qui vit dans une librairie."),
    ("de", "Wie bereite ich mich am besten auf ein Vorstellungsgespräch vor?"),
    ("de", "Schreibe eine kurze Geschichte über einen Leuchtturmwärter, der einen Brief findet."),
    ("de", "Erkläre den Unterschied zwischen Wetter und Klima für ein Kind."),
    ("de", "Formuliere eine höfliche E-Mail, um einen Termin zu verschieben."),
    ("it", "Come si prepara un buon risotto ai funghi?"),
    ("it", "Scrivi una poesia breve sul mare d'estate."),
    ("it", "Quali sono tre consigli per imparare una nuova lingua da adulti?"),
    ("pt", "Quais são as vantagens de andar de bicicleta na cidade?"),
    ("pt", "Escreva uma mensagem de agradecimento para um professor."),
    ("pt", "Explique como funciona a energia solar de forma simples."),
    ("ja", "初めて一人暮らしをする人へのアドバイスを五つ教えてください。"),
    ("ja", "雨の日の駅を舞台にした短い物語を書いてください。"),
    ("ja", "なぜ空は青いのか、子どもにも分かるように説明してください。"),
    ("ja", "上司に有給休暇を申請する丁寧なメールを書いてください。"),
    ("ko", "처음 캠핑을 가는 사람에게 필요한 준비물을 알려 주세요."),
    ("ko", "비 오는 날에 어울리는 짧은 시를 써 주세요."),
    ("ko", "건강한 아침 식사 아이디어를 세 가지 추천해 주세요."),
    ("zh", "请给第一次养猫的人一些实用建议。"),
    ("zh", "写一篇关于城市夜晚的短文，大约两百字。"),
    ("zh", "解释一下为什么树叶在秋天会变色。"),
    ("zh", "帮我写一封邮件，向客户说明项目会延期一周。"),
    ("zh", "列出五个在家锻炼身体的简单方法。"),
    ("th", "ช่วยแนะนำวิธีประหยัดเงินสำหรับนักศึกษาหน่อย"),
    ("th", "เขียนนิทานสั้นๆ เกี่ยวกับช้างที่อยากบินได้"),
    ("th", "อธิบายว่าทำไมต้องดื่มน้ำให้เพียงพอในแต่ละวัน"),
    ("vi", "Hãy cho tôi vài lời khuyên để ngủ ngon hơn."),
    ("vi", "Viết một đoạn văn ngắn tả cảnh chợ quê buổi sáng."),
    ("vi", "Giải thích ngắn gọn vì sao cần tái chế rác thải nhựa."),
    ("hi", "पहली बार नौकरी शुरू करने वाले व्यक्ति के लिए कुछ सुझाव दीजिए।"),
    ("hi", "बारिश के मौसम पर एक छोटी कविता लिखिए।"),
    ("hi", "समझाइए कि पेड़ लगाना पर्यावरण के लिए क्यों ज़रूरी है।"),
    ("ru", "Какие книги ты бы посоветовал человеку, который хочет начать читать больше?"),
    ("ru", "Напиши короткое письмо другу, который переехал в другой город."),
    ("ru", "Объясни простыми словами, как работает солнечная батарея."),
    ("ar", "ما هي أفضل الطرق لتعلم لغة جديدة بسرعة؟"),
    ("ar", "اكتب قصة قصيرة عن طفل يجد خريطة كنز في حديقة جده."),
    ("id", "Berikan tips untuk menjaga kesehatan saat bekerja di depan komputer seharian."),
    ("id", "Tulis pesan ucapan selamat ulang tahun untuk ibu."),
    ("tr", "İlk kez yurt dışına seyahat edecek biri için öneriler yazar mısın?"),
    ("tr", "Sonbahar hakkında kısa bir şiir yaz."),
    ("pl", "Jakie są najlepsze sposoby na oszczędzanie energii w domu?"),
    ("pl", "Napisz krótkie opowiadanie o psie, który zgubił się w lesie."),
    ("nl", "Wat zijn goede tips om beter te leren plannen?"),
    ("nl", "Schrijf een korte uitnodiging voor een verjaardagsfeest in de tuin."),
    ("uk", "Порадь, як підготуватися до першого марафону."),
    ("uk", "Напиши короткий вірш про зимовий ранок."),
    ("sv", "Hur kan man göra sin lägenhet mer energisnål?"),
    ("he", "תן לי שלושה רעיונות לארוחת ערב מהירה ובריאה."),
    ("ms", "Terangkan secara ringkas mengapa tidur yang cukup penting untuk pelajar."),
    ("el", "Γράψε ένα σύντομο μήνυμα για να ευχαριστήσεις έναν φίλο για τη βοήθειά του."),
    ("ne", "नयाँ ठाउँमा घुम्न जाँदा ध्यान दिनुपर्ने कुराहरू के के हुन्?"),
    ("fa", "چند پیشنهاد برای مدیریت بهتر زمان در محل کار بنویس."),
]

FOLLOWUPS = ["Now rewrite your answer in capital letters only.",
             "Now rewrite it entirely in lowercase.",
             "Make it shorter: half the length, same points.",
             "Turn your answer into a numbered list.",
             "Rewrite it so that it has no commas at all.",
             "Now give the same answer in Spanish."]


def build_prompt_set(long_text: str | None = None):
    rng = random.Random(SEED)
    items = []

    def add(domain, text):
        items.append({"domain": domain, "messages": [{"role": "user", "content": text}], "tools": None})

    # instruction following: every task with one to three compatible constraints, three draws per task
    for task, kws in TASKS:
        for _ in range(3):
            cands = _constraints(rng, kws)
            chosen = []
            n_c = rng.choice([1, 1, 2, 2, 2, 3])
            while len(chosen) < n_c:
                pool = [c for c in cands if _compatible(chosen, c)]
                if not pool:
                    break
                chosen.append(rng.choices(pool, weights=[WEIGHT.get(c[0], 1) for c in pool])[0])
            text = task + " " + " ".join(c[2] for c in chosen)
            add("instruct", text)
    for q in YES_NO:
        a, b, c = rng.choice(CHOICES)
        add("instruct", f"{q} Reply with exactly one of these and nothing else: \"{a}\", \"{b}\" or \"{c}\"")
    for q in CHAT:
        add("chat", q)
    for q in WRITING:
        add("writing", q)
    for lang, q in MULTI:
        add(f"multilingual_{lang}", q)

    for i, it in enumerate(items):
        it["id"] = f"i{i:04d}"
        r = rng.random()
        if r < 0.60:                                  # IFEval runs with thinking off; chat use is mostly off too
            it["thinking"], it["effort"] = False, None
        elif r < 0.80:
            it["thinking"], it["effort"] = True, "low"
        else:
            it["thinking"], it["effort"] = True, "high"
        it["followup"] = rng.random() < 0.20
        it["split"] = "cal"
    return items


def check_overlap(items, prompts: list[str], n: int = 8):
    """Prompts sharing any n-word sequence with a prompt list (lowercased, punctuation stripped)."""
    def grams(t):
        w = re.sub(r"[^\w\s]", " ", t.lower()).split()
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    ref = set()
    for p in prompts:
        ref |= grams(p)
    hits = []
    for it in items:
        g = grams(it["messages"][0]["content"]) & ref
        if g:
            hits.append((it["id"], sorted(g)[:3]))
    return hits


if __name__ == "__main__":
    import collections
    import json
    import sys
    s = build_prompt_set()
    c = collections.Counter(x["domain"].split("_")[0] for x in s)
    print(len(s), "prompts", dict(c), "thinking", sum(x["thinking"] for x in s), "followups",
          sum(x["followup"] for x in s))
    names = collections.Counter(w for x in s if x["domain"] == "instruct"
                                for w in ("caps", "lowercase", "comma", "JSON", "whole reply in")
                                if w in x["messages"][0]["content"])
    print("instruct prompts mentioning:", dict(names))
    if len(sys.argv) > 1:
        ref = json.load(open(sys.argv[1]))
        ref = list(ref.values()) if isinstance(ref, dict) else ref
        for n in (8, 6):
            hits = check_overlap(s, ref, n)
            print(f"{n}-gram overlap with {len(ref)} reference prompts: {len(hits)} prompts")
            for h in hits[:10]:
                print("  ", h)
