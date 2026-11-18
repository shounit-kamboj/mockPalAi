"""Extract individual interview questions from cloned, openly licensed repos.

Usage: python3 -I extract.py <raw_dir> <shas.tsv> <out.jsonl>
Repos are treated as data only: files are read as text, nothing is executed.
"""
import fnmatch
import html
import json
import os
import re
import sys

RAW, SHAS, OUT = (sys.argv + [None] * 3)[1:4]

# ---------------------------------------------------------------- sources
# dir, repo, licence, include globs, defaults, per-file overrides
SOURCES = [
    dict(dir="h5bp_Front-end-Developer-Interview-Questions", repo="h5bp/Front-end-Developer-Interview-Questions",
         license="MIT", include=["src/questions/*.md"], category="technical", topic="frontend",
         files={"coding-questions.md": dict(category="coding", skills=["javascript"]),
                "css-questions.md": dict(skills=["css"]), "html-questions.md": dict(skills=["html"]),
                "javascript-questions.md": dict(skills=["javascript"]),
                "network-questions.md": dict(skills=["networking", "http"]),
                "performance-questions.md": dict(skills=["web-performance"]),
                "testing-questions.md": dict(skills=["testing"]),
                "general-questions.md": dict(category="behavioural", topic="general"),
                "fun-questions.md": dict(category="behavioural", topic="general")}),
    dict(dir="yangshun_front-end-interview-handbook", repo="yangshun/front-end-interview-handbook", license="MIT",
         include=["website/contents/css-questions.md", "website/contents/html-questions.md",
                  "website/contents/javascript-questions.md"],
         category="technical", topic="frontend", kinds={"heading"},
         files={"css-questions.md": dict(skills=["css"]), "html-questions.md": dict(skills=["html"]),
                "javascript-questions.md": dict(skills=["javascript"])}),
    dict(dir="yangshun_tech-interview-handbook", repo="yangshun/tech-interview-handbook", license="MIT",
         include=["apps/website/contents/behavioral-interview-questions.md"],
         category="behavioural", topic="general"),
    dict(dir="sudheerj_reactjs-interview-questions", repo="sudheerj/reactjs-interview-questions", license="MIT",
         include=["README.md"], category="technical", topic="frontend", skills=["react"]),
    dict(dir="Chalarangelo_30-seconds-of-interviews", repo="Chalarangelo/30-seconds-of-interviews", license="MIT",
         include=["questions/*.md"], category="technical", topic="frontend", first_heading_only=True,
         thirty_seconds=True),
    dict(dir="ganqqwerty_123-Essential-JavaScript-Interview-Questions",
         repo="ganqqwerty/123-Essential-JavaScript-Interview-Questions", license="BSD-3-Clause",
         include=["README.md"], category="technical", topic="frontend", skills=["javascript"]),
    dict(dir="learning-zone_typescript-basics", repo="learning-zone/typescript-basics", license="MIT",
         include=["README.md"], category="technical", topic="frontend", skills=["typescript"]),
    dict(dir="faqguru_faqguru", repo="faqguru/faqguru", license="MIT", include=["topics/en/*.md"],
         category="technical", topic="frontend", faqguru=True),
    dict(dir="aswanth6000_nodejs-interview-questions", repo="aswanth6000/nodejs-interview-questions",
         license="MIT", include=["README.md"], category="technical", topic="backend", skills=["nodejs"],
         allow_toc=True, kinds={"table"}),
    dict(dir="gauthamjm007_backend-nodejs-golang-interview_qa", repo="gauthamjm007/backend-nodejs-golang-interview_qa",
         license="MIT", include=["README.md"], category="technical", topic="backend"),
    dict(dir="ElemeFE_node-interview", repo="ElemeFE/node-interview", license="MIT",
         include=["sections/en-us/*.md"], category="technical", topic="backend", skills=["nodejs"]),
    dict(dir="ahmed-khateeb_back-end-interview-topics", repo="ahmed-khateeb/back-end-interview-topics",
         license="MIT", include=["README.md"], category="technical", topic="backend"),
    dict(dir="arialdomartini_Back-End-Developer-Interview-Questions",
         repo="arialdomartini/Back-End-Developer-Interview-Questions", license="GPL-2.0",
         include=["README.md"], category="technical", topic="backend", arialdo=True, para_after_heading=4,
         kinds={"para"}),
    dict(dir="xoraus_CrackingTheSQLInterview", repo="xoraus/CrackingTheSQLInterview", license="MIT",
         include=["README.md"], category="technical", topic="databases", skills=["sql"]),
    dict(dir="iamtodor_data-science-interview-questions-and-answers",
         repo="iamtodor/data-science-interview-questions-and-answers", license="MIT", include=["README.md"],
         category="technical", topic="data_science", kinds={"heading"}),
    dict(dir="alexeygrigorev_data-science-interviews", repo="alexeygrigorev/data-science-interviews",
         license="CC-BY-4.0", include=["theory.md"], category="technical", topic="machine_learning"),
    dict(dir="alirezadir_Machine-Learning-Interviews", repo="alirezadir/Machine-Learning-Interviews", license="MIT",
         include=["src/ml-fundamental.md", "src/ml-depth.md", "src/behavioral/behavior.md", "src/MLC/*.md"],
         category="technical", topic="machine_learning",
         files={"behavior.md": dict(category="behavioural", topic="general"),
                "ml-coding.md": dict(category="coding"), "pytorch-ml-coding.md": dict(skills=["pytorch"]),
                "ml-system-design.md": dict(category="system_design")}),
    dict(dir="trimstray_test-your-sysadmin-skills", repo="trimstray/test-your-sysadmin-skills", license="MIT",
         include=["README.md"], category="technical", topic="devops", trimstray=True),
    dict(dir="donnemartin_system-design-primer", repo="donnemartin/system-design-primer", license="CC-BY-4.0",
         include=["README.md"], category="system_design", topic="system_design", skills=["system-design"],
         kinds={"table"}, design_only=True),
    dict(dir="donnemartin_interactive-coding-challenges", repo="donnemartin/interactive-coding-challenges",
         license="Apache-2.0", include=["README.md"], category="coding", topic="cs_fundamentals",
         skills=["algorithms", "data-structures"], table_any=True),
    dict(dir="monashcoding_interview-questions", repo="monashcoding/interview-questions", license="MIT",
         include=["src/questions/behavioural.md", "src/questions/system_design.md"], category="behavioural",
         topic="general", files={"system_design.md": dict(category="system_design", topic="system_design",
                                                           skills=["system-design"])}),
    dict(dir="ashishps1_awesome-behavioral-interviews", repo="ashishps1/awesome-behavioral-interviews",
         license="GPL-3.0", include=["README.md"], category="behavioural", topic="general"),
    dict(dir="ashishps1_awesome-low-level-design", repo="ashishps1/awesome-low-level-design", license="GPL-3.0",
         include=["README.md"], category="system_design", topic="system_design",
         skills=["object-oriented-design"], design_only=True),
]

FAQGURU_TOPICS = {  # file stem -> (topic, skills, category)
    "agile": ("process", ["agile"], "technical"), "angular": ("frontend", ["angular"], None),
    "angularjs": ("frontend", ["angularjs"], None), "asp.net-mvc": ("backend", ["asp.net", "csharp"], None),
    "asp.net": ("backend", ["asp.net", "csharp"], None), "aws": ("devops", ["aws", "cloud"], None),
    "azure": ("devops", ["azure", "cloud"], None), "bootstrap": ("frontend", ["bootstrap", "css"], None),
    "c": ("backend", ["csharp"], None), "code-problems": ("cs_fundamentals", ["algorithms"], "coding"),
    "css": ("frontend", ["css"], None), "data-structures": ("cs_fundamentals", ["data-structures"], None),
    "design-patterns": ("cs_fundamentals", ["design-patterns", "oop"], None), "git": ("devops", ["git"], None),
    "golang": ("backend", ["go"], None), "graphql": ("backend", ["graphql"], None),
    "html5": ("frontend", ["html"], None), "java": ("backend", ["java"], None),
    "javascript": ("frontend", ["javascript"], None), "mongodb": ("databases", ["mongodb", "nosql"], None),
    "net-framework": ("backend", [".net", "csharp"], None), "nodejs": ("backend", ["nodejs"], None),
    "react": ("frontend", ["react"], None), "redux": ("frontend", ["redux", "react"], None),
    "ruby-on-rails": ("backend", ["ruby-on-rails", "ruby"], None), "ruby": ("backend", ["ruby"], None),
    "sass": ("frontend", ["sass", "css"], None), "sql": ("databases", ["sql"], None),
    "typeScript": ("frontend", ["typescript"], None), "vuejs": ("frontend", ["vue"], None),
    "webpack": ("frontend", ["webpack"], None),
}

# ---------------------------------------------------------------- rules
WH = {"what", "what's", "whats", "why", "how", "when", "where", "which", "who", "whom", "whose"}
AUX = {"is", "are", "was", "were", "can", "could", "do", "does", "did", "will", "would", "should", "shall",
       "have", "has", "had", "may", "might", "must", "isn't", "aren't", "don't", "doesn't", "can't"}
IMP = {"explain", "describe", "tell", "talk", "give", "walk", "design", "implement", "write", "compare",
       "contrast", "define", "discuss", "share", "name", "list", "differentiate", "distinguish", "demonstrate",
       "illustrate", "outline", "estimate", "create", "find", "determine", "calculate", "convert",
       "reverse", "check", "sort", "merge", "detect", "count", "identify", "elaborate", "remove", "rotate",
       "compress", "partition", "validate", "construct", "compute", "solve", "search", "flatten", "swap",
       "generate", "print", "insert", "delete"}
LEAD_OK_WITH_QMARK = WH | AUX | IMP | {"if", "in", "given", "suppose", "imagine", "consider", "assume", "say",
                                        "let's", "for", "as", "on", "with", "after", "without", "using", "between",
                                        "difference", "your", "you", "any", "from", "under", "at", "to", "beyond",
                                        "besides", "other", "apart", "of", "aside", "tell", "ever",
                                        "favorite", "favourite", "best", "biggest", "most", "next", "once", "think"}
BLACKLIST = ["contribut", "table of contents", "back to top", "pull request", "this repo", "this repository",
             "license", "click here", "sponsor", "subscribe", "buy me", "changelog", "how to use this",
             "how to read this", "who is this for", "about the author", "found a mistake", "know the answer",
             "do you know how to answer", "questions to ask", "ask the interviewer", "ask your interviewer",
             "reverse interview", "greatfrontend", "interview handbook", "devinterview", "leetcode.com",
             "star the repo", "this guide", "this book", "this list", "this page", "this section",
             "the answer", "answers here"]
SECTION_SKIP = re.compile(r"(questions? (you (can|could|should) |to )?ask|ask (the|your) interviewer|reverse "
                          r"interview|final questions|resources|references|further reading|contribut|licen[cs]e|"
                          r"credits|acknowledg|table of contents|useful links|books|courses|videos|tips|"
                          r"how to prepare|study plan|what to explain)", re.I)
CODE_DEPENDENT = re.compile(
    r"(\b(following|below|above|this|these|given|next)\s+(piece of\s+)?(code|snippet|program|statements?|"
    r"functions?|examples?|syntax|outputs?|quer(y|ies)|tables?|schema|class|expressions?|lines?|declarations?|"
    r"implementation|code snippets?)\b|what (will|would|does|is) (be )?(the )?(output|printed|logged|result|"
    r"returned|log|print)\b|\boutput of\b|console\.log|printed (on|to) the console|\bwhat gets (logged|printed)|"
    r"\bwhat would happen if we\b|\bfix (the|this) (bug|code)\b|\b(below|above)\b)", re.I)
PRONOUN_SHORT = re.compile(r"\b(these|this|that|them|those|it|they|above|below)\b", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FFFF☀-➿⬀-⯿️‍←-⇿⌀-⏿]")


HTML_TAGS = {"a", "b", "i", "u", "p", "br", "hr", "em", "strong", "code", "kbd", "sup", "sub", "pre", "div", "span",
             "img", "ul", "ol", "li", "table", "tr", "td", "th", "thead", "tbody", "details", "summary", "h1", "h2",
             "h3", "h4", "h5", "h6", "blockquote", "center", "small", "big", "font", "section", "article", "nav",
             "header", "footer", "picture", "source", "video", "iframe", "abbr", "mark", "del", "ins", "s", "dl",
             "dt", "dd", "input", "label", "button", "form", "select", "option", "textarea", "script", "style",
             "link", "meta", "html", "head", "body", "title", "svg", "path", "figure", "figcaption", "q", "cite"}


def clean(s):
    s = html.unescape(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    # keep placeholders such as <role> or <company name> (they are not HTML tags)
    s = re.sub(r"<([A-Za-z][A-Za-z ]{1,30})>",
               lambda m: m.group(0) if m.group(1).split()[0].lower() in HTML_TAGS
               else (f"&lt;{m.group(1)}&gt;" if re.fullmatch(r"[A-Z]\w*", m.group(1)) else f"[{m.group(1)}]"), s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&lt;", "<").replace("&gt;", ">")  # JSX component names such as <Router>
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[\[[^\]]*\]\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)
    s = s.replace("**", "")
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"\1", s)
    s = re.sub(r"(?<![\w])_(?!\s)([^_]+?)(?<!\s)_(?![\w])", r"\1", s)
    s = re.sub(r":[a-z_]+:", "", s)
    s = re.sub(r"\s*[—–-]+\s*Difficulty:\s*\w+\s*$", "", s, flags=re.I)
    s = EMOJI.sub("", s)
    s = re.sub(r"\s*\*{2,}\s*$", "", s)
    s = re.sub(r"\s*\(\s*(junior|mid|senior|easy|medium|hard)\s*\)\s*$", "", s, flags=re.I)
    s = re.sub(r"^\s*Q(?:uestion)?\s*\d*\s*[.:)\-]\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\d+\s*[.):\-]\s+", "", s)
    s = re.sub(r"^\s*\d+\)\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" -–—:|#>*")
    return s


def difficulty_of(raw):
    m = re.search(r"Difficulty:\**\s*(Easy|Medium|Hard)", raw, re.I)
    if m:
        return m.group(1).lower()
    if "👶" in raw:
        return "easy"
    if "🚀" in raw:
        return "hard"
    if "⭐" in raw:
        return "medium"
    return None


def english(s):
    letters = [c for c in s if c.isalpha()]
    return letters and sum(c.isascii() for c in letters) / len(letters) >= 0.97


def accept(q, kind, src):
    if not (12 <= len(q) <= (450 if kind == "para" else 320)) or not english(q):
        return False
    words = q.split()
    if len(words) < 3 or "](" in q or "http" in q.lower() or "|" in q:
        return False
    low = q.lower()
    if any(b in low for b in BLACKLIST) or CODE_DEPENDENT.search(q):
        return False
    if "{" in q or "}" in q or ";" in q:
        return False
    if q.endswith(":") or q.isupper():
        return False
    first = re.sub(r"[^a-z']", "", words[0].lower())
    if words[0].endswith(",") and not q.endswith("?"):
        return False  # "List, which is an ordered collection ..." is a statement, not an instruction
    if len(words) <= 7 and PRONOUN_SHORT.search(q) and q[0].islower():
        return False  # dangling sub-bullet like "how do you choose among these?"
    if kind == "para":
        return "?" in q or first in IMP
    if src.get("design_only"):
        return first == "design"
    if src.get("table_any") and kind == "table":
        return first in IMP or q.endswith("?")
    if "\\" in q:
        return False
    if q.endswith("?"):
        return first in LEAD_OK_WITH_QMARK or kind in ("heading", "summary", "q")
    if first in WH:
        return kind in ("heading", "summary", "bold", "q", "table")
    if first in AUX:
        return kind in ("heading", "summary", "bold", "q")
    if first in IMP:
        if kind == "list":
            return len(words) >= 5
        return kind in ("heading", "summary", "bold", "q", "table")
    return False


def finalize(q):
    q = q[0].upper() + q[1:]
    first = re.sub(r"[^a-z']", "", q.split()[0].lower())
    if not q.endswith(("?", ".", "!")) and (first in WH or first in AUX):
        q += "?"
    return q


def parse_line(line):
    s = line.strip()
    m = re.search(r"<summary>(.*?)</summary>", s, re.I)
    if m:
        return "summary", m.group(1), None
    kind = None
    m = re.match(r"^((?:[-*+]|\d+[.)])\s+)+", s)
    if m:
        kind, s = "list", s[m.end():]
    m = re.match(r"^(#{1,6})\s+", s)
    if m:
        return "heading", s[m.end():], len(m.group(1))
    if s.startswith("|"):
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        for c in cells:
            if c and not re.fullmatch(r"[-:\s]+|\d+\.?|#", c):
                return "table", c, None
        return None, None, None
    if re.match(r"^(\*\*|__).+(\*\*|__)\W*\S*\W*$", s) and kind is None:
        return "bold", s, None
    if re.match(r"^(Q\d*\s*[.:)]|Question\s*\d*\s*[.:)])", s, re.I):
        return "q", s, None
    if kind == "list":
        return "list", s, None
    return None, None, None


def strip_fences(text):
    """Blank out fenced code. CommonMark-style: a closing fence is a bare run of the same
    fence characters, so an info string like ```jsx can only open a block."""
    out, fence = [], None
    for line in text.split("\n"):
        st = line.strip()
        if fence:
            if re.fullmatch(re.escape(fence[0]) + "{%d,}" % len(fence), st):
                fence = None
            out.append("")
            continue
        m = re.match(r"^(`{3,}|~{3,})", st)
        if m and "`" not in st[len(m.group(1)):]:
            fence = m.group(1)
            out.append("")
            continue
        out.append(line)
    return out


def trimstray_meta(stack):
    joined = " / ".join(h for h in stack if h)
    sen = None
    for key, val in (("junior", "junior"), ("regular", "mid"), ("senior", "senior"), ("guru", "senior")):
        if key in joined.lower():
            sen = val
    skills = []
    for key, val in (("system", ["linux"]), ("network", ["networking"]), ("devops", ["devops"]),
                     ("cyber", ["security"]), ("simple", ["linux"])):
        if key in joined.lower():
            skills = val
    return sen, skills


def main():
    shas = dict(line.rstrip("\n").split("\t") for line in open(SHAS))
    n = 0
    with open(OUT, "w") as out:
        for src in SOURCES:
            root = os.path.join(RAW, src["dir"])
            files = []
            for dirpath, _, names in os.walk(root):
                if "/.git" in dirpath:
                    continue
                for name in names:
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    if any(fnmatch.fnmatch(rel, g) for g in src["include"]):
                        files.append(rel)
            for rel in sorted(files):
                fname = os.path.basename(rel)
                stem = fname[:-3]
                meta = dict(category=src["category"], topic=src["topic"], skills=list(src.get("skills", [])))
                meta.update(src.get("files", {}).get(fname, {}))
                if src.get("faqguru"):
                    t, sk, cat = FAQGURU_TOPICS.get(stem, ("general", [], None))
                    meta.update(topic=t, skills=sk)
                    if cat:
                        meta["category"] = cat
                text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
                seniority_file = None
                if src.get("thirty_seconds"):
                    tags = re.search(r"<!--\s*tags:\s*\(([^)]*)\)", text)
                    exp = re.search(r"<!--\s*expertise:\s*\((\d)\)", text)
                    if tags:
                        meta["skills"] = [t.strip().lower() for t in tags.group(1).split(",") if t.strip()]
                        t0 = meta["skills"][0] if meta["skills"] else ""
                        if t0 == "node":
                            meta.update(topic="backend")
                    if exp:
                        seniority_file = {"0": "junior", "1": "mid", "2": "senior"}.get(exp.group(1))
                stack = [None] * 7
                got_first = False
                pending_para = False
                for line in strip_fences(text):
                    kind, body, level = parse_line(line)
                    if src.get("para_after_heading"):
                        if kind == "heading":
                            pending_para = level >= src["para_after_heading"]
                        elif pending_para and line.strip():
                            pending_para = False
                            if kind is None:
                                kind, body = "para", line.strip()
                    if kind is None:
                        continue
                    if kind == "heading":
                        stack[level] = clean(body)
                        for i in range(level + 1, 7):
                            stack[i] = None
                        for i in range(1, 7):  # a table of contents ends at the next heading of any level
                            if i != level and stack[i] and re.search(r"table of contents|^contents$", stack[i], re.I):
                                stack[i] = None
                    parents = [h for i, h in enumerate(stack) if h and (kind != "heading" or i < level)]
                    if any(SECTION_SKIP.search(h) for h in parents) and not (
                            src.get("allow_toc") and all(not SECTION_SKIP.search(h) or "contents" in h.lower()
                                                         for h in parents)):
                        continue
                    if src.get("kinds") and kind not in src["kinds"]:
                        continue
                    if src.get("first_heading_only"):
                        if got_first or kind != "heading":
                            continue
                    diff = difficulty_of(body)
                    q = clean(body)
                    if not accept(q, kind, src):
                        continue
                    got_first = True
                    rec = dict(question=finalize(q), category=meta["category"], topic=meta["topic"],
                               skills=list(meta["skills"]), difficulty=diff, seniority=seniority_file,
                               section=" / ".join(parents)[:200], source=src["repo"], source_file=rel,
                               source_url=f"https://github.com/{src['repo']}/blob/{shas[src['dir']]}/{rel}",
                               license=src["license"], kind=kind)
                    if src.get("trimstray"):
                        sen, sk = trimstray_meta(parents)
                        rec["seniority"] = sen
                        rec["skills"] = sk or rec["skills"]
                    if src.get("arialdo") and re.search(r"open questions|bill gates|team|soft", rec["section"],
                                                         re.I):
                        rec["category"] = "behavioural" if "open" in rec["section"].lower() else rec["category"]
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    print("extracted", n)


if __name__ == "__main__":
    main()
