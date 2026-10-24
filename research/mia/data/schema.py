"""Fixed three-fact support-ticket schema, rendered with NanoJev's own toy support generator.

F1 = main_topic, F2 = refund_status, F3 = impact_level. Their value sets, the state
body templates, the question instructions, the dynamic Choice candidates and the
Score levels all come from upstream ``scripts/build_toy_decisions.py``.

Upstream keys each phrasing template by its dataset split so templates never cross
splits. For membership inference the opposite is required: members and non-members
must come from the same distribution, so here the four upstream phrasings are just
template IDs drawn identically for every pool.

Upstream state IDs are the fact triple itself, which allows only 6*6*4 = 144 states.
To obtain thousands of distinct states with the same fixed fact roles, each state
gets a nuisance header (ticket number, customer, channel, product, order total).
Nuisance fields never change across counterfactual variants.
"""
from mia.paths import ensure_upstream_on_path

ensure_upstream_on_path()
import build_toy_decisions as upstream  # noqa: E402  (NanoJev's support/decision generator)

FAMILY_ID = "support_mia_v1"
FACT_NAMES = ("main_topic", "refund_status", "impact_level")
FACT_BY_INDEX = {1: "main_topic", 2: "refund_status", 3: "impact_level"}
FACT_VALUES = {
    "main_topic": tuple(upstream.ROUTES),
    "refund_status": tuple(upstream.REFUNDS),
    "impact_level": tuple(range(len(upstream.IMPACTS))),
}
TEMPLATE_KEYS = tuple(upstream.TEMPLATES)
LANGUAGES = ("zh", "en")

# Question IDs and primitives exactly as upstream support_record emits them.
QUESTION_IDS = ("route", "completed", "severity")
QID_PRIMITIVE = {"route": "choice", "completed": "boolean", "severity": "score"}
PRIMITIVE_QID = {v: k for k, v in QID_PRIMITIVE.items()}
PRIMITIVES = ("boolean", "choice", "score")
# The fact each question's rule reads; the other two facts are irrelevant to it.
QID_RULE_FACT = {"route": "main_topic", "completed": "refund_status", "severity": "impact_level"}

# Verbatim from upstream build_toy_decisions.support_record (commit recorded in paths.py);
# test_counterfactuals.py checks our builder reproduces upstream questions exactly.
INSTRUCTIONS = (
    ("只按工单主诉选择支持团队；退款记录和功能影响是另外两个字段。",
     "Choose a support team from the main topic only; refund and impact are separate fields."),
    ("退款是否已经实际到账完成？申请、处理中、批准或拒绝均不算到账。",
     "Has the refund actually been paid and completed? Requests, processing, approval, or denial do not count."),
    ("按明确记载的使用影响，选择最匹配的严重程度。",
     "Choose the severity level matching the explicitly recorded impact."),
)

FIRST_NAMES_EN = ("Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Liam", "Fatima", "Noah", "Elena",
                  "Kenji", "Sofia", "Omar", "Hannah", "Diego", "Yuki", "Grace", "Mateo", "Leila", "Ethan",
                  "Ana", "Ivan", "Chloe", "Ravi", "Zoe", "Samuel", "Mei", "Lucas", "Nadia", "Owen")
LAST_NAMES_EN = ("Chen", "Smith", "Patel", "Garcia", "Kim", "Nguyen", "Okafor", "Rossi", "Muller", "Silva",
                 "Haddad", "Tanaka", "Novak", "Brown", "Lopez", "Ivanova", "Khan", "Moreau", "Larsen", "Costa",
                 "Wright", "Sato", "Mensah", "Popescu", "Dubois", "Ali", "Fischer", "Reyes", "Walker", "Lin")
SURNAMES_ZH = ("王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴", "徐", "孙", "胡", "朱", "高",
               "林", "何", "郭", "马", "罗")
GIVEN_NAMES_ZH = ("伟", "芳", "娜", "敏", "静", "磊", "洋", "艳", "勇", "军", "杰", "娟", "涛", "明", "超",
                  "秀英", "霞", "平", "刚", "桂英")
PRODUCTS = ("CloudSync Pro", "PixelNote", "HomeHub Mini", "TrailRunner X", "AquaPure 3", "StudyPal",
            "FitBand Lite", "SoundWave Max", "PhotoVault", "ChefMate", "BrightDesk", "SafeLock Home",
            "MetroPass", "GreenLeaf Planner", "TaxEase", "KidsCode Box", "VoltCharge", "SnapPrint Duo",
            "ZenSleep", "PetTrack GPS", "RideShare Plus", "BookNest", "TravelLite Bag", "QuickInvoice")
CHANNELS = {"en": ("email", "live chat", "phone", "web form", "mobile app"),
            "zh": ("邮件", "在线客服", "电话", "网页表单", "手机应用")}
TICKET_RANGE = (100000, 999999)
AMOUNT_CENTS_RANGE = (500, 50000)


def language_index(language):
    if language not in LANGUAGES:
        raise ValueError(f"Unknown language {language!r}")
    return int(language == "en")


def draw_nuisance(rng, ticket):
    """Language-independent nuisance indices; rendering resolves them per language."""
    return {"ticket": int(ticket),
            "first_name": rng.randrange(len(FIRST_NAMES_EN)), "last_name": rng.randrange(len(LAST_NAMES_EN)),
            "channel": rng.randrange(len(CHANNELS["en"])), "product": rng.randrange(len(PRODUCTS)),
            "amount_cents": rng.randint(*AMOUNT_CENTS_RANGE)}


def render_header(nuisance, language):
    amount = nuisance["amount_cents"] / 100
    product = PRODUCTS[nuisance["product"]]
    if language == "en":
        name = f"{FIRST_NAMES_EN[nuisance['first_name']]} {LAST_NAMES_EN[nuisance['last_name']]}"
        channel = CHANNELS["en"][nuisance["channel"]]
        return f"Ticket #{nuisance['ticket']} from {name} via {channel}; product: {product}; order total ${amount:.2f}."
    name = SURNAMES_ZH[nuisance["last_name"] % len(SURNAMES_ZH)] + GIVEN_NAMES_ZH[nuisance["first_name"] % len(GIVEN_NAMES_ZH)]
    channel = CHANNELS["zh"][nuisance["channel"]]
    return f"工单号{nuisance['ticket']}；客户：{name}；渠道：{channel}；产品：{product}；订单金额{amount:.2f}元。"


def render_body(facts, template_key, language):
    """Upstream support state text, unchanged."""
    li = language_index(language)
    return upstream.TEMPLATES[template_key][li].format(
        issue=upstream.ROUTES[facts["main_topic"]][li],
        refund=upstream.REFUNDS[facts["refund_status"]][li],
        impact=upstream.IMPACTS[facts["impact_level"]][li])


def render_state(facts, nuisance, template_key, language):
    """The single renderer used for original states and every counterfactual variant."""
    validate_facts(facts)
    separator = " " if language == "en" else ""
    return render_header(nuisance, language) + separator + render_body(facts, template_key, language)


def build_questions(main_topic, template_key, language, choice_k, rng):
    """Same three questions as upstream support_record; Choice candidates depend only on F1 and rng."""
    li = language_index(language)
    prefix = upstream.QUESTION_PREFIX[template_key][li]
    options = {key: value[2 + li] for key, value in upstream.ROUTES.items()}
    criteria = upstream.dynamic_candidates(options, main_topic, upstream.NEAR_NEGATIVE[main_topic], choice_k, rng)
    lang = "en" if li else "zh"
    return {
        "route": {"type": "choice", "instructions": prefix + INSTRUCTIONS[0][li], "criteria": criteria},
        "completed": {"type": "boolean", "instructions": prefix + INSTRUCTIONS[1][li]},
        "severity": {"type": "score", "instructions": prefix + INSTRUCTIONS[2][li],
                     "criteria": list(upstream.SCORE_CRITERIA[lang])},
    }


def rule_gold(facts):
    """Deterministic upstream labels: each question reads exactly one fact."""
    validate_facts(facts)
    return {"route": facts["main_topic"], "completed": facts["refund_status"] == "issued",
            "severity": int(facts["impact_level"])}


def validate_facts(facts):
    if set(facts) != set(FACT_NAMES):
        raise ValueError(f"Facts must be exactly {FACT_NAMES}, got {sorted(facts)}")
    for name in FACT_NAMES:
        if facts[name] not in FACT_VALUES[name]:
            raise ValueError(f"Invalid {name} value {facts[name]!r}")


def gold_index(question, gold):
    """Candidate index of a gold answer, following upstream candidate ordering."""
    if question["type"] == "boolean":
        return int(bool(gold))
    if question["type"] == "choice":
        return list(question["criteria"]).index(gold)
    return int(gold)
