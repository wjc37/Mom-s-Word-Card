"""Assign Mom's vocabulary into 50 meaning-based clusters."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

VOCAB_PATH = Path(__file__).parent / "vocabulary.json"
FALLBACK = "__unassigned__"

# Exactly 50 meaning-based clusters (Korean label + Korean/English keywords)
MEANING_CLUSTERS: list[dict[str, object]] = [
    {"name": "감정 · 기분", "ko": ["감정", "기분", "기쁨", "슬픔", "분노", "행복", "우울", "불안", "두려", "공포", "짜증", "흥분", "외로", "그리", "불만", "분개", "억울", "고충", "항의", "투덜", "불평"], "en": ["emotion", "mood", "feel", "happy", "sad", "anger", "fear", "anxiety", "joy", "grief", "dread", "frantic", "elated", "melanch", "grievance", "resent", "complain", "grumble"]},
    {"name": "성격 · 태도", "ko": ["성격", "성질", "태도", "겸손", "거만", "오만", "유순", "완고", "고집", "소심", "대담", "용기", "겁", "온순", "건방"], "en": ["personality", "character", "attitude", "humble", "arrogant", "stubborn", "shy", "bold", "meek", "timid", "cocky", "sassy", "docile"]},
    {"name": "비난 · 모욕", "ko": ["비난", "모욕", "경멸", "멸시", "조롱", "비하", "욕", "헐뜯", "꾸짖", "책망", "비방", "모독", "독설"], "en": ["critic", "insult", "contempt", "scorn", "mock", "deride", "rebuke", "denounce", "vilify", "disdain", "berate", "ridicule", "vitriol"]},
    {"name": "칭찬 · 존경", "ko": ["칭찬", "존경", "경외", "감탄", "영예", "명성", "인정", "감사", "고마", "찬양"], "en": ["praise", "admire", "respect", "honor", "acclaim", "revere", "grateful", "kudos", "prestig"]},
    {"name": "말하기 · 표현", "ko": ["말", "발음", "표현", "설명", "담화", "연설", "발언", "구사", "암시", "완곡", "장황", "간결"], "en": ["speak", "talk", "rhetoric", "articulate", "utter", "phrase", "pronounc", "eloquen", "verbal", "paraphrase", "enunciate"]},
    {"name": "생각 · 판단", "ko": ["생각", "판단", "추론", "추측", "믿", "의심", "확신", "주장", "견해", "논리", "이해", "깨닫", "수학", "비율", "통계", "증명", "가설"], "en": ["think", "believe", "reason", "deduc", "specul", "opinion", "judg", "convict", "assert", "contempl", "comprehend", "discern", "math", "ratio", "logic", "theorem", "hypothesis"]},
    {"name": "기억 · 지각", "ko": ["기억", "지각", "인지", "자각", "망각", "기억 상실", "직관", "감지"], "en": ["memory", "remember", "recall", "perceiv", "cogni", "amnesia", "forget", "aware", "notion", "sense"]},
    {"name": "법률 · 범죄", "ko": ["법", "범죄", "죄", "재판", "기소", "유죄", "무죄", "형사", "교도", "감옥", "탄핵", "소송", "증거", "변호"], "en": ["legal", "law", "crime", "criminal", "court", "trial", "guilty", "felony", "prison", "prosec", "attorney", "verdict", "indict", "lawsuit"]},
    {"name": "정치 · 권력", "ko": ["정치", "권력", "정부", "독재", "입법", "선거", "시위", "반란", "혁명", "민주", "보수", "진보", "의회", "국회"], "en": ["politic", "government", "dictator", "legislat", "democra", "rebell", "insurrect", "regime", "authority", "power", "oligarch", "senate", "congress"]},
    {"name": "의학 · 질병", "ko": ["병", "질병", "증상", "치료", "의사", "환자", "진단", "염증", "감염", "발작", "마비", "치매", "암", "수술"], "en": ["disease", "medical", "symptom", "doctor", "patient", "diagnos", "infect", "seizure", "paraly", "dementia", "patholog", "clinic", "therapy"]},
    {"name": "건강 · 몸 상태", "ko": ["건강", "피로", "통증", "아픔", "상처", "회복", "체질", "수면", "숙취", "변비", "기침", "콧물"], "en": ["health", "pain", "ache", "hurt", "injur", "recover", "fatigue", "sleep", "hangover", "cough", "sick", "ill", "nausea", "vomit"]},
    {"name": "신체 · 해부", "ko": ["신체", "몸", "팔", "다리", "머리", "심장", "폐", "위", "간", "뼈", "피부", "근육", "자궁", "방광"], "en": ["body", "limb", "brain", "heart", "lung", "stomach", "liver", "bone", "skin", "muscle", "bladder", "uterus", "anatom"]},
    {"name": "음식 · 식사", "ko": ["음식", "식사", "요리", "맛", "먹", "마시", "배", "식욕", "레시피", "재료", "술", "음주", "취", "담배", "흡연"], "en": ["food", "eat", "meal", "cook", "cuisine", "taste", "drink", "appetite", "recipe", "ingredient", "beverage", "snack", "alcohol", "drunk", "booze", "beer", "wine", "smoke", "cigarette"]},
    {"name": "일 · 직장", "ko": ["일", "직장", "업무", "근무", "회사", "사무", "채용", "해고", "퇴직", "면접", "경력"], "en": ["work", "job", "office", "employ", "career", "business", "task", "duty", "workplace", "hire", "fired", "resume"]},
    {"name": "경제 · 돈", "ko": ["돈", "경제", "가격", "비용", "재산", "부", "가난", "세금", "투자", "금융", "무역", "적자"], "en": ["money", "economic", "price", "cost", "wealth", "poor", "tax", "invest", "finance", "trade", "budget", "profit", "debt", "deficit"]},
    {"name": "교육 · 학습", "ko": ["교육", "학습", "공부", "학교", "학생", "교사", "수업", "시험", "지식", "숙련"], "en": ["education", "learn", "study", "school", "student", "teacher", "class", "exam", "knowledge", "skill", "train", "tutor"]},
    {"name": "가족 · 관계", "ko": ["가족", "부모", "아이", "부부", "결혼", "이혼", "친구", "관계", "애정", "사랑", "연애", "성", "섹스", "성적", "유혹"], "en": ["family", "parent", "child", "marriage", "divorce", "friend", "relationship", "love", "romance", "spouse", "dating", "sex", "sexual", "seduc", "lust"]},
    {"name": "교통 · 운전", "ko": ["운전", "교통", "차", "도로", "사고", "충돌", "주차", "운행", "차선", "브레이크"], "en": ["drive", "traffic", "car", "vehicle", "road", "accident", "collision", "lane", "brake", "highway", "interstate", "park"]},
    {"name": "여행 · 이동", "ko": ["여행", "이동", "도착", "출발", "항공", "호텔", "관광", "길", "방향"], "en": ["travel", "trip", "journey", "arrive", "depart", "flight", "hotel", "tourist", "route", "wander", "roam"]},
    {"name": "자연 · 날씨", "ko": ["날씨", "비", "눈", "바람", "구름", "자연", "동물", "식물", "숲", "바다", "산", "개", "고양", "새", "곤충", "꽃", "나무"], "en": ["weather", "rain", "snow", "wind", "cloud", "nature", "animal", "plant", "forest", "ocean", "mountain", "storm", "dog", "cat", "bird", "insect", "flower", "tree"]},
    {"name": "시간 · 변화", "ko": ["시간", "변화", "증가", "감소", "성장", "발전", "진행", "지속", "일시", "영원", "갑자기"], "en": ["time", "change", "grow", "increase", "decrease", "develop", "progress", "duration", "sudden", "gradual", "evolve"]},
    {"name": "크기 · 정도", "ko": ["크기", "정도", "양", "많", "적", "충분", "과도", "극단", "상당", "약간"], "en": ["size", "amount", "large", "small", "enough", "excess", "extreme", "moderate", "slight", "vast", "tiny", "immense"]},
    {"name": "강함 · 능력", "ko": ["강", "능력", "능숙", "실력", "기술", "재능", "영재", "무능", "역량", "숙련"], "en": ["strong", "ability", "skill", "competent", "talent", "proficient", "capable", "incompetent", "expert", "prodigy", "prowess"]},
    {"name": "약함 · 결함", "ko": ["약", "부족", "결함", "흠", "실패", "오류", "잘못", "허술", "엉성", "결점"], "en": ["weak", "lack", "flaw", "defect", "fail", "error", "mistake", "shortcoming", "inadequ", "flimsy", "mediocre"]},
    {"name": "성공 · 성취", "ko": ["성공", "성취", "달성", "승리", "이기", "해내", "완수", "성과", "열망", "염원", "포부", "야망", "갈망", "목표"], "en": ["success", "achieve", "accompl", "victory", "win", "triumph", "attain", "fulfill", "excel", "aspire", "yearn", "ambition", "desire", "crave", "strive"]},
    {"name": "실패 · 좌절", "ko": ["실패", "좌절", "포기", "실망", "낙담", "절망", "곤경", "궁지", "위기"], "en": ["fail", "defeat", "give up", "disappoint", "despair", "frustrat", "predicament", "crisis", "setback", "futility"]},
    {"name": "협력 · 도움", "ko": ["협력", "도움", "돕", "지원", "협조", "연대", "보조", "함께", "동료"], "en": ["cooper", "help", "assist", "support", "collabor", "solidarity", "aid", "together", "ally", "mutual"]},
    {"name": "갈등 · 적대", "ko": ["갈등", "적대", "반목", "싸움", "투쟁", "분쟁", "원한", "증오", "적", "반항"], "en": ["conflict", "hostil", "fight", "struggle", "dispute", "enemy", "hatred", "antagon", "rebel", "feud", "warfare"]},
    {"name": "평화 · 화해", "ko": ["평화", "화해", "조화", "합의", "용서", "관용", "달래", "진정"], "en": ["peace", "reconcil", "harmony", "agreement", "forgive", "toleran", "calm", "soothe", "appease"]},
    {"name": "정직 · 도덕", "ko": ["정직", "도덕", "양심", "윤리", "선", "악", "죄책", "진실", "거짓", "허위"], "en": ["honest", "moral", "conscience", "ethic", "virtue", "guilt", "truth", "false", "integrity", "authentic"]},
    {"name": "속임 · 기만", "ko": ["속임", "기만", "거짓", "사기", "허세", "가식", "위장", "속이다", "허풍"], "en": ["deceit", "fraud", "fake", "phony", "pretend", "disguise", "trick", "bluff", "cheat", "hypocrisy", "flam"]},
    {"name": "위험 · 위협", "ko": ["위험", "위협", "해로", "손상", "파괴", "재난", "참사", "위기", "불길"], "en": ["danger", "threat", "harm", "damage", "destroy", "disaster", "catastrophe", "risk", "hazard", "ominous", "peril"]},
    {"name": "안전 · 보호", "ko": ["안전", "보호", "방어", "예방", "면역", "피난", "구조", "보호구역"], "en": ["safe", "protect", "defend", "prevent", "immune", "shelter", "rescue", "secure", "guard", "sanctuary"]},
    {"name": "권리 · 의무", "ko": ["권리", "의무", "책임", "규칙", "법규", "준수", "허가", "금지", "면제"], "en": ["right", "duty", "obligation", "responsib", "rule", "regulat", "permit", "forbid", "exempt", "comply", "conform"]},
    {"name": "사회 · 문화", "ko": ["사회", "문화", "관습", "전통", "풍습", "계층", "계급", "서민", "대중"], "en": ["social", "culture", "custom", "tradition", "society", "class", "public", "communal", "folk", "societal"]},
    {"name": "종교 · 신앙", "ko": ["종교", "신앙", "기독", "교회", "신", "성스", "기도", "축복", "죄", "속죄"], "en": ["religion", "faith", "christian", "church", "god", "sacred", "prayer", "bless", "sin", "protestant", "orthodox"]},
    {"name": "예술 · 창작", "ko": ["예술", "창작", "미술", "음악", "영화", "문학", "시", "연극", "공연", "유머", "풍자", "농담", "장난"], "en": ["art", "creative", "music", "film", "movie", "literature", "poem", "theater", "perform", "aesthetic", "scenic", "humor", "satire", "joke", "funny", "comic", "wit"]},
    {"name": "과학 · 기술", "ko": ["과학", "기술", "연구", "실험", "데이터", "디지털", "기계", "장치", "발명"], "en": ["science", "technology", "research", "experiment", "data", "digital", "machine", "device", "invent", "tech", "lab"]},
    {"name": "환경 · 에너지", "ko": ["환경", "오염", "에너지", "연료", "자원", "재활용", "기후"], "en": ["environment", "pollut", "energy", "fuel", "resource", "climate", "ecolog", "renew"]},
    {"name": "가정 · 생활", "ko": ["가정", "집", "생활", "가사", "청소", "세탁", "가구", "이불", "욕실"], "en": ["home", "house", "domestic", "household", "clean", "laundry", "furniture", "kitchen", "bathroom", "living"]},
    {"name": "쇼핑 · 물건", "ko": ["쇼핑", "구매", "판매", "상품", "물건", "가격", "할인", "교환", "환불"], "en": ["shop", "buy", "sell", "product", "goods", "purchase", "discount", "refund", "commodity", "retail"]},
    {"name": "컴퓨터 · 미디어", "ko": ["컴퓨터", "인터넷", "미디어", "방송", "뉴스", "광고", "소셜", "앱"], "en": ["computer", "internet", "media", "broadcast", "news", "advert", "social", "online", "digital", "software"]},
    {"name": "스포츠 · 경쟁", "ko": ["스포츠", "경기", "경쟁", "선수", "팀", "점수", "승부"], "en": ["sport", "game", "compet", "athlete", "team", "score", "match", "race", "champion"]},
    {"name": "색 · 외형", "ko": ["색", "외형", "모습", "옷", "복장", "아름", "못생", "화려", "단정"], "en": ["color", "colour", "appearance", "look", "clothes", "dress", "beauty", "ugly", "fancy", "attire", "garment"]},
    {"name": "소리 · 청각", "ko": ["소리", "소음", "울림", "속삭", "고함", "신음", "웃음", "울음", "듣", "청각", "엿듣", "귀", "경청"], "en": ["sound", "noise", "whisper", "shout", "moan", "laugh", "cry", "scream", "groan", "blare", "chirp", "hear", "listen", "overhear", "audible"]},
    {"name": "움직임 · 동작", "ko": ["움직", "걷", "달리", "뛰", "점프", "미끄", "넘", "기울", "흔들"], "en": ["move", "walk", "run", "jump", "slide", "climb", "fall", "swing", "stumble", "plunge", "stroll"]},
    {"name": "만짐 · 접촉", "ko": ["만지", "잡", "붙", "쥐", "밀", "당기", "던지", "치", "부딪"], "en": ["touch", "hold", "grab", "cling", "push", "pull", "throw", "hit", "collide", "nudge", "squeeze"]},
    {"name": "보기 · 시각", "ko": ["보", "시각", "눈", "응시", "바라", "엿보", "시야", "눈길"], "en": ["see", "look", "view", "watch", "gaze", "glance", "peek", "stare", "visual", "sight", "observe"]},
    {"name": "일상 · 습관", "ko": ["일상", "습관", "버릇", "루틴", "매일", "평소", "생활습관", "취미"], "en": ["daily", "routine", "habit", "usual", "customary", "regular", "hobby", "leisure"]},
    {"name": "형용 · 묘사", "ko": ["형용", "묘사", "특징", "성질", "상태", "모양", "느낌", "인상"], "en": ["adject", "descript", "trait", "quality", "state", "shape", "impress", "remarkable", "notable"]},
]

assert len(MEANING_CLUSTERS) == 50, f"Expected 50 clusters, got {len(MEANING_CLUSTERS)}"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    return text


def item_text(item: dict) -> str:
    return normalize(f"{item['word']} {item.get('meaning', '')}")


def score_item(item: dict, cluster: dict) -> int:
    blob = item_text(item)
    score = 0
    for kw in cluster.get("ko", []):
        if str(kw).lower() in blob:
            score += 3
    for kw in cluster.get("en", []):
        kw = str(kw).lower()
        if kw and kw in blob:
            score += 2
    return score


def assign_by_keywords(items: list[dict]) -> list[dict]:
    for item in items:
        best_name = FALLBACK
        best_score = 0
        for cluster in MEANING_CLUSTERS:
            score = score_item(item, cluster)
            if score > best_score:
                best_score = score
                best_name = str(cluster["name"])
        item["category"] = best_name
    return items


def propagate_by_neighbors(items: list[dict]) -> list[dict]:
    texts = [item_text(item) for item in items]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)

    assigned_idx = [i for i, item in enumerate(items) if item["category"] != FALLBACK]
    unassigned_idx = [i for i, item in enumerate(items) if item["category"] == FALLBACK]

    if not assigned_idx or not unassigned_idx:
        return items

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(matrix[assigned_idx])

    distances, indices = nn.kneighbors(matrix[unassigned_idx], return_distance=True)
    for local_i, vocab_i in enumerate(unassigned_idx):
        neighbor_local = int(indices[local_i][0])
        neighbor_vocab = assigned_idx[neighbor_local]
        items[vocab_i]["category"] = items[neighbor_vocab]["category"]

    return items


def ensure_all_categories(items: list[dict], min_size: int = 10) -> list[dict]:
    cluster_by_name = {str(c["name"]): c for c in MEANING_CLUSTERS}

    def counts() -> Counter[str]:
        return Counter(item["category"] for item in items)

    for cluster in MEANING_CLUSTERS:
        name = str(cluster["name"])
        missing = min_size - counts()[name]
        if missing <= 0:
            continue

        ranked: list[tuple[int, int, str]] = []
        for idx, item in enumerate(items):
            if item["category"] == name:
                continue
            ranked.append((score_item(item, cluster), idx, item["category"]))

        ranked.sort(key=lambda row: (-row[0], counts()[row[2]]))

        for score, idx, _ in ranked:
            if missing <= 0:
                break
            if score <= 0:
                break
            items[idx]["category"] = name
            missing -= 1

    still_missing = [str(c["name"]) for c in MEANING_CLUSTERS if counts()[str(c["name"])] == 0]
    if still_missing:
        texts = [item_text(item) for item in items]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(texts)

        for name in still_missing:
            cluster = cluster_by_name[name]
            seed_idx = max(
                range(len(items)),
                key=lambda i: score_item(items[i], cluster),
            )
            seed_vec = matrix[seed_idx]
            similarities = (matrix @ seed_vec.T).toarray().ravel()
            order = similarities.argsort()[::-1]
            for idx in order:
                if counts()[name] >= min_size:
                    break
                items[idx]["category"] = name

    return items


def rebalance_tiny_clusters(items: list[dict], min_size: int = 5) -> list[dict]:
    counts = Counter(item["category"] for item in items)
    tiny = {name for name, count in counts.items() if count < min_size and counts[name] > 0}

    if not tiny:
        return items

    cluster_keywords: dict[str, set[str]] = {}
    for cluster in MEANING_CLUSTERS:
        name = str(cluster["name"])
        words = {str(k).lower() for k in cluster.get("ko", [])} | {
            str(k).lower() for k in cluster.get("en", [])
        }
        cluster_keywords[name] = words

    large_names = [name for name in counts if name not in tiny]

    for item in items:
        if item["category"] not in tiny:
            continue
        blob = item_text(item)
        best_name = item["category"]
        best_score = 0
        for name in large_names:
            score = sum(1 for kw in cluster_keywords.get(name, set()) if kw and kw in blob)
            if score > best_score:
                best_score = score
                best_name = name
        if best_score > 0:
            item["category"] = best_name

    still_tiny = [
        item
        for item in items
        if item["category"] in tiny and Counter(i["category"] for i in items)[item["category"]] < min_size
    ]
    if still_tiny:
        texts = [item_text(item) for item in items]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        large_idx = [i for i, item in enumerate(items) if item["category"] not in tiny]
        tiny_idx = [i for i, item in enumerate(items) if item["category"] in tiny]
        if large_idx and tiny_idx:
            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(matrix[large_idx])
            _, indices = nn.kneighbors(matrix[tiny_idx], return_distance=True)
            for local_i, vocab_i in enumerate(tiny_idx):
                neighbor_vocab = large_idx[int(indices[local_i][0])]
                items[vocab_i]["category"] = items[neighbor_vocab]["category"]

    return items


def main() -> None:
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    items = assign_by_keywords(data["vocabulary"])
    items = propagate_by_neighbors(items)
    items = ensure_all_categories(items)
    items = rebalance_tiny_clusters(items)

    counts = Counter(item["category"] for item in items)
    if FALLBACK in counts:
        raise RuntimeError(f"{counts[FALLBACK]} words still unassigned")

    ordered = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    if len(ordered) != 50:
        missing = [str(c["name"]) for c in MEANING_CLUSTERS if str(c["name"]) not in ordered]
        raise RuntimeError(f"Expected 50 categories, got {len(ordered)}. Missing: {missing}")

    data["vocabulary"] = items
    data["meta"] = {
        "total_words": len(items),
        "total_categories": len(ordered),
        "categories": ordered,
        "clustering": "meaning_keywords_knn_v2",
    }

    VOCAB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Updated {len(items)} words into {len(ordered)} categories")
    print("Size range:", min(counts.values()), "-", max(counts.values()))
    for name, cnt in list(ordered.items())[:15]:
        print(f"  {cnt:4d}  {name}")


if __name__ == "__main__":
    main()
