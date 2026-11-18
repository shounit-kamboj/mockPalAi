"""Tag, de-duplicate and write the MockPal question bank.

Usage: python3 -I build.py <extracted.jsonl> <out_dir> [--threshold 0.8] [--report]
"""
import csv
import json
import os
import re
import sys
import uuid
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SRC, OUT = sys.argv[1], sys.argv[2]
THRESH = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.8
REPORT = "--report" in sys.argv

PERMISSIVE = {"MIT", "BSD-3-Clause", "Apache-2.0", "CC-BY-4.0", "WTFPL"}
# Preference when two sources hold the same question: curated, human-written lists first.
SOURCE_RANK = [
    "yangshun/tech-interview-handbook", "h5bp/Front-end-Developer-Interview-Questions",
    "yangshun/front-end-interview-handbook", "Chalarangelo/30-seconds-of-interviews",
    "sudheerj/reactjs-interview-questions", "trimstray/test-your-sysadmin-skills",
    "alexeygrigorev/data-science-interviews", "alirezadir/Machine-Learning-Interviews",
    "iamtodor/data-science-interview-questions-and-answers", "donnemartin/system-design-primer",
    "donnemartin/interactive-coding-challenges", "xoraus/CrackingTheSQLInterview", "faqguru/faqguru",
    "ganqqwerty/123-Essential-JavaScript-Interview-Questions", "learning-zone/typescript-basics",
    "ElemeFE/node-interview", "gauthamjm007/backend-nodejs-golang-interview_qa",
    "ahmed-khateeb/back-end-interview-topics", "monashcoding/interview-questions",
    "arialdomartini/Back-End-Developer-Interview-Questions", "ashishps1/awesome-behavioral-interviews",
    "ashishps1/awesome-low-level-design", "aswanth6000/nodejs-interview-questions",
]

TECH = [
    (r"\breact(\.?js)?\b|\bjsx\b|\buse(state|effect|memo|callback|ref|context|reducer)\b", "react"),
    (r"\bredux\b", "redux"), (r"\bangular(js)?\b", "angular"), (r"\bvue(\.?js)?\b", "vue"),
    (r"\bnext\.?js\b", "nextjs"),
    (r"\bjavascript\b|\bjs\b|\becmascript\b|\bes6\b|\bclosures?\b|\bprototyp|\bhoisting\b|\bpromises?\b|"
     r"\basync/await\b|\bevent loop\b|\bthis keyword\b|\b=== ?\b", "javascript"),
    (r"\btypescript\b", "typescript"),
    (r"\bnode\.?js\b|\bnode js\b|\bnpm\b|\bexpress(\.?js)?\b|\blibuv\b", "nodejs"), (r"\bexpress(\.?js)?\b", "express"),
    (r"\bhtml5?\b|\bdoctype\b|\bsemantic (html|elements?|markup)\b|\bdom\b", "html"),
    (r"\bcss3?\b|\bflexbox\b|\bcss grid\b|\bselectors?\b|\bbox model\b|\bsass\b|\bscss\b|\bmedia quer", "css"),
    (r"\bpython\b|\bdjango\b|\bflask\b|\bfastapi\b|\bpep ?8\b", "python"),
    (r"\bjava\b|\bjvm\b|\bjdk\b|\bjre\b|\bspring boot\b", "java"),
    (r"\bgolang\b|\bgoroutines?\b|\bgo (compiler|programming|language|routines?)\b", "go"),
    (r"\bc#|\.net\b|\basp\.net\b", "csharp"), (r"\bruby\b|\brails\b", "ruby"),
    (r"\bsql\b|\bmysql\b|\bpostgres(ql)?\b|\bstored procedures?\b|\bnormali[sz]ation\b|\bprimary keys?\b|"
     r"\bforeign keys?\b|\binner join\b|\bouter join\b|\bleft join\b", "sql"),
    (r"\bdatabases?\b|\bdbms\b|\brdbms\b|\bacid\b|\bsharding\b|\breplication\b|\bindex(es|ing)?\b", "databases"),
    (r"\bnosql\b|\bmongo(db|ose)?\b|\bcassandra\b|\bdynamodb\b|\bredis\b|\bkey-value stores?\b", "nosql"),
    (r"\bgraphql\b", "graphql"),
    (r"\brest(ful)?\b(?=.*\b(api|service|endpoint)s?\b)|\brest api\b|\bapi design\b|\bendpoints?\b|\bapis?\b",
     "apis"),
    (r"\bhttps?\b|\bcookies?\b|\bcors\b|\bstatus codes?\b|\bhttp/2\b", "http"),
    (r"\btcp\b|\budp\b|\bdns\b|\bip address|\bsubnet|\brouting tables?\b|\bosi\b|\bnetwork(ing|s)?\b|\bsockets?\b|"
     r"\btls\b|\bssl\b|\bports?\b|\bfirewall", "networking"),
    (r"\bsecurity\b|\bxss\b|\bcross-site\b|\bcsrf\b|\bsql injection\b|\bencrypt|\bhash(ing)? passwords?\b|"
     r"\bauthenticat|\bauthori[sz]|\boauth\b|\bjwt\b|\bjson web token\b|\bvulnerab|\bcryptograph|\bsanitiz",
     "security"),
    (r"\btest(s|ing|able)?\b|\btdd\b|\bunit tests?\b|\bmock(s|ing)?\b|\bjest\b|\bmocha\b|\bjunit\b", "testing"),
    (r"\bgit\b|\bgithub\b|\bbranch(es|ing)?\b|\bmerge conflicts?\b|\brebase\b|\bcherry-pick\b|\bstash\b", "git"),
    (r"\bdocker\b|\bcontainers?\b|\bkubernetes\b|\bk8s\b", "containers"),
    (r"\bci/cd\b|\bcontinuous (integration|delivery|deployment)\b|\bjenkins\b|\bdeploy(ment|ments|ing)?\b",
     "ci-cd"),
    (r"\baws\b|\bazure\b|\bgcp\b|\bcloud\b|\bs3\b|\bec2\b|\bserverless\b", "cloud"),
    (r"\blinux\b|\bunix\b|\bbash\b|\bshell\b|\bkernel\b|\bsystemd\b|\bchmod\b|\binodes?\b|\bfile ?systems?\b|"
     r"\bcron\b|\bssh\b", "linux"),
    (r"\bthreads?\b|\bconcurren|\bmutex|\bdeadlocks?\b|\brace conditions?\b|\bsemaphores?\b|\bparallel|"
     r"\bworker threads?\b", "concurrency"),
    (r"\boop\b|\bobject[- ]oriented\b|\binheritance\b|\bpolymorphism\b|\bencapsulation\b|\babstract class"
     , "oop"),
    (r"\bdesign patterns?\b|\bsingleton\b|\bfactory\b|\bobserver\b|\bdecorator pattern\b|\bdependency "
     r"injection\b|\binversion of control\b|\bmvc\b|\bsolid\b|\bdry\b", "design-patterns"),
    (r"\bmicroservices?\b|\bservice[- ]oriented\b|\bsoa\b|\bmessage (queue|broker)s?\b|\bkafka\b|\brabbitmq\b|"
     r"\bevent[- ]driven\b", "microservices"),
    (r"\bdistributed\b|\bcap theorem\b|\beventual consistency\b|\bconsensus\b", "distributed-systems"),
    (r"\bscal(e|able|ability|ing)\b|\bload balanc|\bcach(e|es|ing)\b|\bcdn\b|\bhigh availability\b|"
     r"\bthroughput\b", "scalability"),
    (r"\bperformance\b|\boptimi[sz](e|ing|ation)\b|\bmemory leaks?\b|\bprofil(e|ing)\b|\blatency\b",
     "performance"),
    (r"\baccessib|\baria\b|\ba11y\b|\bscreen readers?\b", "accessibility"),
    (r"\bwebpack\b|\bbabel\b|\bbundl(e|er|ing)\b|\bvite\b|\bpolyfill", "build-tools"),
    (r"\balgorithms?\b|\bbig[- ]?o\b|\btime complexity\b|\bspace complexity\b|\bsort(ing|ed)?\b|"
     r"\bbinary search\b|\brecursi(on|ve)\b|\bdynamic programming\b|\bgreedy\b|\bbacktracking\b|\bbfs\b|"
     r"\bdfs\b|\bbreadth[- ]first\b|\bdepth[- ]first\b|\bpalindrome|\bfibonacci\b|\bpermutations?\b", "algorithms"),
    (r"\bdata structures?\b|\barrays?\b|\blinked lists?\b|\bstacks?\b|\bqueues?\b|\bheaps?\b|"
     r"\bhash ?(maps?|tables?|sets?)\b|\btrees?\b|\bgraphs?\b|\btries\b|\btrie\b|\bbst\b|\bmatrix\b",
     "data-structures"),
    (r"\bmachine learning\b|\bml\b|\boverfit|\bunderfit|\bregulari[sz]|\bcross[- ]validation\b|"
     r"\bfeatures? (selection|engineering|importance)\b|\bsupervised\b|\bunsupervised\b|"
     r"\bclassif(y|ier|iers|ication)\b|\bregression\b|\bclustering\b|\bk-?means\b|\bdecision trees?\b|"
     r"\brandom forests?\b|\bgradient boost|\bxgboost\b|\bsvm\b|\bsupport vector\b|\bknn\b|"
     r"\bnearest neighbou?rs?\b|\bnaive bayes\b|\bensemble\b|\bbias[- ]variance\b|\bhyperparameter|"
     r"\bimbalanced\b|\bpca\b|\bdimensionality\b|\btraining (data|set)\b|\bbagging\b|\bboosting\b",
     "machine-learning"),
    (r"\bdeep learning\b|\bneural (networks?|nets?)\b|\bcnns?\b|\brnns?\b|\blstm\b|\btransformers?\b|"
     r"\bbackprop|\bactivation functions?\b|\bdropout\b|\bbatch norm|\bconvolution|\bembeddings?\b|"
     r"\bgradient descent\b|\bsgd\b|\boptimizers?\b|\blearning rate\b|\bepochs?\b|\bvanishing gradient",
     "deep-learning"),
    (r"\bnlp\b|\bnatural language\b|\btokeni[sz]|\bword2vec\b|\btf-?idf\b|\bbert\b|\bllms?\b|"
     r"\blanguage models?\b|\brag\b|\bprompt(s|ing)?\b", "nlp"),
    (r"\bstatistic|\bp-?values?\b|\bhypothes[ie]s\b|\bconfidence intervals?\b|\bprobabilit|\bdistributions?\b|"
     r"\bvariance\b|\bstandard deviation\b|\bbayes|\bcorrelat|\ba/b test|\bsignifican|\bsampling\b|"
     r"\bt-test\b|\bcentral limit\b", "statistics"),
    (r"\bprecision\b|\brecall\b|\bf1\b|\broc\b|\bauc\b|\bconfusion matrix\b|\bmetrics?\b", "model-evaluation"),
    (r"\bpytorch\b|\btensors?\b|\bautograd\b|\bdataloader\b|\bnn\.module\b", "pytorch"),
    (r"\btensorflow\b|\bkeras\b", "tensorflow"),
    (r"\bpandas\b|\bdataframes?\b|\bnumpy\b|\bscikit|\bsklearn\b", "python-data"),
    (r"\brecommend(er|ation)s?\b|\branking\b", "recommender-systems"),
    (r"\bsystem design\b|\bdesign (a|an|the)\b.*\b(system|service|platform|app|application|website|engine|"
     r"cache|store|crawler|feed|chat|shortener|api)\b", "system-design"),
    (r"\bagile\b|\bscrum\b|\bkanban\b|\bsprints?\b|\bcode reviews?\b|\btechnical debt\b|\brefactor", "process"),
]
TECH = [(re.compile(p, re.I), s) for p, s in TECH]

BEHAVIOURAL = [
    (r"\bconflicts?\b|\bdisagree|\bdifficult (person|people|coworker|colleague|teammate|manager|customer|client)|"
     r"\bhard time working\b|\bargument", "conflict-resolution"),
    (r"\bteams?\b|\bteamwork\b|\bteammates?\b|\bcollaborat|\bcolleagues?\b|\bco-?workers?\b|\bpeers?\b",
     "teamwork"),
    (r"\blead(er|ers|ership|ing)?\b|\bmentor|\binfluenc|\bdelegat", "leadership"),
    (r"\bfail(ed|ure|ures)?\b|\bmistakes?\b|\bwent wrong\b|\bsetbacks?\b|\bregret", "failure-and-learning"),
    (r"\bfeedback\b|\bcriticism\b|\bcriticized\b|\bcritique\b", "feedback"),
    (r"\bdeadlines?\b|\bprioriti[sz]|\btime management\b|\bpressure\b|\bmultiple (tasks|projects|priorities)\b|"
     r"\bworkload\b|\bjuggl", "prioritization"),
    (r"\bcommunicat|\bnon-technical\b|\bstakeholders?\b|\bpresent(ation|ed)?s?\b|\bexplain(ed)? .* to\b",
     "communication"),
    (r"\bcustomers?\b|\bclients?\b|\busers?\b", "customer-focus"),
    (r"\bchang(e|es|ed|ing)\b|\badapt|\bambigu|\buncertain|\bunclear\b", "adaptability"),
    (r"\bdecisions?\b|\bdecide\b|\btrade-?offs?\b|\brisks?\b|\bjudg(e)?ment\b", "decision-making"),
    (r"\blearn(ed|ing|t)?\b|\bnew (skill|technology|technologies)\b|\bcurious|\bgrowth\b|\bimprove yourself\b|"
     r"\bup[- ]to[- ]date\b|\bkeep up\b",
     "learning"),
    (r"\bwhy (do you want|are you interested|did you apply)\b|\bwhy .*\b(company|role|position|us|here)\b|"
     r"\bmotivat|\bpassion|\bgoals?\b|\bwhere do you see yourself\b|\bcareer\b|\bstrengths?\b|\bweakness",
     "motivation-and-fit"),
    (r"\bproud|\baccomplish|\bachiev|\bsuccess", "achievements"),
    (r"\binitiative\b|\bownership\b|\babove and beyond\b|\bproactiv|\bwithout being asked\b", "ownership"),
    (r"\bproblems?\b|\bchalleng|\bdifficult|\bobstacles?\b", "problem-solving"),
    (r"\bethic|\bintegrity\b|\bhonest", "integrity"),
]
BEHAVIOURAL = [(re.compile(p, re.I), s) for p, s in BEHAVIOURAL]

ARIALDO_SECTIONS = {
    "open questions": ("behavioural", []), "general questions": ("behavioural", []),
    "bill gates": ("technical", ["puzzles"]), "languages": ("technical", ["programming-languages"]),
    "software architecture": ("technical", ["software-architecture"]),
    "code design": ("technical", ["code-design"]), "design patterns": ("technical", ["design-patterns"]),
    "logic and algorithms": ("technical", ["algorithms"]), "security": ("technical", ["security"]),
    "databases": ("technical", ["databases"]), "distributed systems": ("technical", ["distributed-systems"]),
    "web development": ("technical", ["web"]), "concurrency": ("technical", ["concurrency"]),
    "nosql": ("technical", ["nosql"]), "code versioning": ("technical", ["git"]),
    "service oriented": ("technical", ["microservices"]),
    "lifecycle and team": ("technical", ["software-process"]),
}


def norm_key(q):
    s = q.lower()
    s = re.sub(r"\bwhat'?s\b", "what is", s)
    s = re.sub(r"\b(can|could|would) you (please )?", "", s)
    s = re.sub(r"\bversus\b|\bvs\.?\b", "vs", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9+#]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tag(row):
    q, sec = row["question"], row.get("section", "")
    skills = list(row["skills"])
    if row["source"].startswith("arialdomartini"):
        for key, (cat, sk) in ARIALDO_SECTIONS.items():
            if key in sec.lower():
                row["category"] = cat
                skills += sk
                break
    if row["category"] == "behavioural":
        row["topic"] = "behavioural"
        comp = [s for rx, s in BEHAVIOURAL if rx.search(q)]
        skills = [s for s in skills if s not in ("javascript",)] + comp[:4]
    else:
        text = q if row["source"].startswith(("faqguru", "aswanth")) else f"{q} || {sec}"
        hits = [s for rx, s in TECH if rx.search(text)]
        skills += hits
    seen, out = set(), []
    for s in skills:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    row["skills"] = out[:8]
    return row


JUNK = re.compile(r"\b(foo|bar|baz|qux)\b|^(when|why|how) (not )?to use\b", re.I)
SHORT_PRONOUN = re.compile(r"\b(it|this|that|these|those|them)\b", re.I)


NEG = re.compile(r"\b(not|never|without|no)\b|n't\b", re.I)
OPS = re.compile(r"[=<>!+\-*/%&|^~#@$]{1,4}")


def mergeable(a, b):
    """Guard against merging opposites ("suitable" vs "not suitable") or operator questions
    whose words match but symbols differ ("== vs ===" in JS vs "#== vs #===" in Ruby)."""
    if bool(NEG.search(a)) != bool(NEG.search(b)):
        return False
    return sorted(OPS.findall(a)) == sorted(OPS.findall(b))


def greedy_keep(texts, vectorizer, thresh_fn, originals=None):
    X = vectorizer.fit_transform(texts)
    S = (X @ X.T).tocsr()
    originals = originals or texts
    lens = [len(t.split()) for t in texts]
    keep = np.ones(len(texts), dtype=bool)
    pairs = []
    for i in range(len(texts)):
        if not keep[i]:
            continue
        row = S.getrow(i)
        for j, v in zip(row.indices, row.data):
            if j > i and keep[j] and v >= thresh_fn(min(lens[i], lens[j])) and mergeable(originals[i],
                                                                                          originals[j]):
                keep[j] = False
                pairs.append((v, i, j))
    return keep, pairs


def main():
    rows = [json.loads(line) for line in open(SRC)]
    rows = [r for r in rows if "snippets of code" not in r.get("section", "").lower()]
    rows = [r for r in rows if not JUNK.search(r["question"])
            and not (len(r["question"].split()) <= 4 and SHORT_PRONOUN.search(r["question"]))]
    # The aswanth6000 list is mostly machine-written rewordings of the same few dozen topics:
    # collapse it on content words before comparing with everything else.
    asw = [r for r in rows if r["source"].startswith("aswanth6000")]
    rest = [r for r in rows if not r["source"].startswith("aswanth6000")]
    asw = list({norm_key(r["question"]): r for r in reversed(asw)}.values())[::-1]
    k_asw, _ = greedy_keep([norm_key(r["question"]) for r in asw],
                           TfidfVectorizer(stop_words="english", sublinear_tf=True), lambda n: 0.35)
    rows = rest + [r for r, k in zip(asw, k_asw) if k]
    rows = [tag(r) for r in rows]
    rank = {s: i for i, s in enumerate(SOURCE_RANK)}
    rows.sort(key=lambda r: (r["license"] not in PERMISSIVE, rank.get(r["source"], 99),
                             not (r.get("difficulty") or r.get("seniority"))))

    # exact duplicates
    seen, uniq = set(), []
    for r in rows:
        k = norm_key(r["question"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    # near duplicates across all sources (word 1-2 gram TF-IDF cosine), greedy in priority order;
    # short questions need a much closer match ("What is AWS?" vs "What is AWS EBS?")
    keys = [norm_key(r["question"]) for r in uniq]
    keep, idx_pairs = greedy_keep(keys, TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                  lambda n: THRESH if n >= 7 else 0.85, [r["question"] for r in uniq])
    pairs = [(v, uniq[i]["question"], uniq[j]["question"]) for v, i, j in idx_pairs]
    final = [r for r, k in zip(uniq, keep) if k]

    if REPORT:
        import random
        random.seed(1)
        bands = defaultdict(list)
        for v, a, b in pairs:
            bands[round(int(v * 20) / 20, 2)].append((v, a, b))
        for band in sorted(bands):
            print(f"--- similarity {band:.2f}-{band + 0.05:.2f}: {len(bands[band])} removed")
            for v, a, b in random.sample(bands[band], min(4, len(bands[band]))):
                print(f"   KEEP {a[:90]}\n   DROP {b[:90]}")

    os.makedirs(OUT, exist_ok=True)
    out_rows = []
    for r in final:
        out_rows.append(dict(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "mockpal-question:" + norm_key(r["question"]))),
            question=r["question"], category=r["category"], topic=r["topic"], skills=r["skills"],
            difficulty=r.get("difficulty"), seniority=r.get("seniority"), source=r["source"],
            source_url=r["source_url"], license=r["license"]))
    cat_order = {"behavioural": 0, "technical": 1, "coding": 2, "system_design": 3}
    out_rows.sort(key=lambda r: (cat_order.get(r["category"], 9), r["topic"], r["source"]))
    json.dump(out_rows, open(os.path.join(OUT, "questions.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["id", "question", "category", "topic", "skills", "difficulty", "seniority", "source",
                "source_url", "license"]
        w.writerow(cols)
        for r in out_rows:
            w.writerow([";".join(r[c]) if c == "skills" else (r[c] or "") for c in cols])
    print(f"input {len(rows)}  exact-unique {len(uniq)}  after near-dup({THRESH}) {len(out_rows)}")
    print("by category:", dict(Counter(r["category"] for r in out_rows)))
    print("by licence:", dict(Counter(r["license"] for r in out_rows)))
    print("by source:")
    for s, n in Counter(r["source"] for r in out_rows).most_common():
        print(f"  {n:5d}  {s}")


if __name__ == "__main__":
    main()
