"""
System-prompt heuristic detector.

We do not have ground truth for "this string is a system prompt", so
we use a multi-feature heuristic:
  1. Length above a threshold (system prompts are rarely under 80 chars).
  2. Instruction-following language ("You are", "Your task", "Always",
     "Never", "Do not").
  3. Multi-sentence structure (at least two terminating punctuation marks).
  4. Absence of URL-like or log-like signatures.
  5. Presence of role-shaping cues ("assistant", "the user", "respond",
     "answer in", "step by step").

Each feature contributes a weight; total weight gives a score in [0, 1].
A configurable threshold (default 0.55) decides whether a string is
flagged as a candidate system prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_apk_scanner.models import (
    Finding,
    FindingKind,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.scanners.base import BaseScanner

INSTRUCTION_OPENERS = re.compile(
    r"^\s*(you are|you will|your task|your role|as an?|"
    r"act as|you must|you should|you may|i want you to|"
    r"please act|please respond|the assistant)\b",
    re.IGNORECASE,
)

#: Instruction openers in the other languages the frame actually contains.
#:
#: AndroZoo spans AppChina, Anzhi and other regional stores
#: (METHODOLOGY.md §2.2), so an English-only detector does not merely miss
#: some prompts at random — it misses them in proportion to how non-English
#: an app is. Validation found a pilot application shipping thirteen Chinese
#: prompt templates (e.g. 你是 [主題] 的教授，請總結以下文本) that scored zero on
#: every English feature and were reported as no findings at all.
#:
#: CJK has no word boundaries, so these are matched without ``\b``.
INSTRUCTION_OPENERS_INTL = re.compile(
    # Chinese: 你是/您是 (you are), 作為/作为 (as a), 請/请 (please), 扮演 (act as),
    # 我要你 (I want you to), 你的任務/任务 (your task)
    r"(你是|您是|作為|作为|請你|请你|請扮演|请扮演|扮演一(个|個)|我要你|"
    r"你的任務|你的任务|你將|你将|你必須|你必须|你應該|你应该|以下是你的"
    # Japanese: あなたは (you are), してください (please do), 役割 (role)
    r"|あなたは|あなたの役割|してください|振る舞って"
    # Korean: 당신은 (you are), 역할 (role), 해주세요 (please do)
    r"|당신은|당신의 역할|역할을 맡|해주세요"
    # Spanish / Portuguese: eres un, actúa como, tu tarea, você é, aja como
    r"|eres un|eres una|actúa como|actua como|tu tarea|tu función"
    r"|você é|voce e|aja como|sua tarefa|comporte-se como"
    # French: tu es un, vous êtes, agis comme, ton rôle
    r"|tu es un|tu es une|vous êtes un|vous etes un|agis comme|ton rôle|votre rôle"
    # German: du bist ein, verhalte dich, deine aufgabe
    r"|du bist ein|du bist eine|sie sind ein|verhalte dich|deine aufgabe"
    # Russian: ты — , вы — , веди себя, твоя задача
    r"|ты\s*[—-]\s*|вы\s*[—-]\s*|веди себя|твоя задача|ваша задача"
    # Italian / Turkish / Indonesian / Vietnamese / Arabic / Hindi
    r"|sei un|sei una|agisci come|il tuo compito"
    r"|sen bir|görevin|davran"
    r"|kamu adalah|anda adalah|bertindak sebagai|tugas anda"
    r"|bạn là|nhiệm vụ của bạn|hãy đóng vai"
    r"|أنت مساعد|أنت خبير|مهمتك|تصرف ك"
    r"|आप एक|तुम एक|आपका कार्य)",
    re.IGNORECASE,
)

ROLE_LANGUAGE = re.compile(
    r"\b(assistant|the user|respond|answer in|reply with|step.by.step|"
    r"chain.of.thought|reasoning|do not|never|always|must not|"
    r"you cannot|do not reveal|system prompt|tools? available|"
    r"available tools)\b",
    re.IGNORECASE,
)

#: Role-shaping cues in other languages, matched without word boundaries.
ROLE_LANGUAGE_INTL = re.compile(
    # Chinese: 助手/助理 (assistant), 用戶/用户 (user), 回答 (answer), 不要 (do not),
    # 必須 (must), 總是 (always), 從不 (never), 一步一步 (step by step), 系統提示
    r"(助手|助理|用戶|用户|回答|回覆|回复|不要|不得|必須|必须|總是|总是|"
    r"從不|从不|一步一步|逐步|系統提示|系统提示|提示詞|提示词|扮演|語氣|语气"
    # Task verbs and role nouns — the imperative core of a Chinese prompt.
    r"|總結|总结|翻譯|翻译|生成|撰寫|撰写|列出|解釋|解释|分析|改寫|改写"
    r"|請問|请问|專家|专家|教授|以下文本|根據以下|根据以下"
    # Japanese / Korean
    r"|アシスタント|ユーザー|回答して|してはいけません|必ず"
    r"|어시스턴트|사용자|답변|하지 마|반드시"
    # Spanish / Portuguese / French / German / Russian
    r"|asistente|usuario|responde|no debes|siempre|nunca"
    r"|assistente|usuário|responda|não deve"
    r"|assistant|utilisateur|réponds|ne doit pas|toujours|jamais"
    r"|assistent|benutzer|antworte|niemals|immer"
    r"|ассистент|помощник|пользовател|отвечай|никогда|всегда"
    # Turkish / Indonesian / Vietnamese / Arabic / Hindi
    r"|asistan|kullanıcı|cevap ver|asla"
    r"|asisten|pengguna|jawab|jangan"
    r"|trợ lý|người dùng|trả lời|không được"
    r"|مساعد|المستخدم|أجب|لا تفعل"
    r"|सहायक|उपयोगकर्ता|उत्तर)",
    re.IGNORECASE,
)

#: Sentence terminators beyond ASCII: CJK full-width stops, Arabic full stop
#: and question mark, Devanagari danda.
SENTENCE_TERMINATORS = ".!?。！？．…۔؟।"

#: Additional clause separators counted only for CJK-dense text.
#:
#: The underlying feature is "structured prose rather than an identifier or
#: token list". In Latin scripts sentence terminators mark that; CJK
#: instructions routinely run several clauses joined by the full-width comma
#: and end without any terminator at all, so counting only 。！？ scores a
#: perfectly ordinary Chinese prompt as unstructured.
CJK_CLAUSE_SEPARATORS = "，、；："

#: Scripts whose characters carry far more meaning each than Latin letters,
#: so a prompt of equivalent content is much shorter in character count.
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

#: Standalone bracketed slots, the shape of a fill-in prompt template
#: ("As the [role] at [company]…", "你是 [主題] 的教授").
#:
#: The opening bracket must not follow a word character, which is what
#: separates a template slot from an analytics parameter: `[公司]` and
#: `[topic]` match, `app[name]` and `billing_details[email]` do not. That
#: distinction matters — those parameter strings are everywhere in Firebase
#: and Stripe code and would otherwise swamp the signal.
#:
#: Added after manual review found a pilot application shipping thirteen
#: Chinese templates of exactly this shape.
#: Also matches ``{{char}}`` / ``{{user}}`` mustache slots, which are the
#: character-card convention rather than the bracketed one. Validation on
#: another pilot application found genuine role-play prompts using this form and
#: no bracketed slots at all, so a bracket-only pattern misses them entirely.
#: A slot names something ("[topic]", "[公司]", "{{char}}"). A bare index
#: ("[0]", "[i]", "[t]") names nothing and is ordinary array syntax.
#:
#: The distinction is load-bearing, not cosmetic. Minified JavaScript is dense
#: with bare indices, and because template_slots also sets the strong-signal
#: flag, matching them let minified bundles bypass ``WEAK_CEILING`` entirely:
#: five webpack chunks in one corpus application scored
#: 0.57-0.63 as "system prompts". Requiring a slot to start with a letter and
#: run to at least two characters keeps every real template form and drops the
#: syntax.
TEMPLATE_SLOT = re.compile(
    r"(?<![\w\]])\[[A-Za-zÀ-ɏЀ-ӿ؀-ۿ"
    r"ऀ-ॿ一-鿿぀-ヿ가-힯]"
    r"[^\[\]\n]{1,23}\]"
    r"|\{\{[^{}\n]{1,24}\}\}"
)

#: Source or minified program text.
#:
#: Reading web bundles (Defect 14's fix) brought Cordova/Capacitor and Next.js
#: apps into scope, and with them their vendored JavaScript. Minified code is
#: long, punctuation-dense and full of tokens the role-language feature counts,
#: so it scores like structured prose on every feature the detector measures.
#: It is never a prompt.
CODE_LIKE = re.compile(
    r'"use strict"|__WEBPACK|module\.exports|hasOwnProperty|prototype\.'
    r"|typeof [a-z]{1,3}[.)=]|function\s*\([a-z],[a-z]\)|\)\{return |=>\{"
    r"|throw new (TypeError|Error)\(|Object\.defineProperty"
    r"|\bvar [a-z]=|\breturn [a-z]\[|null!==|void 0",
    re.IGNORECASE,
)

URL_LIKE = re.compile(r"https?://|www\.|\.(com|org|io|ai|net|gov)/")

#: Signals that the text addresses a *model* rather than merely resembling
#: instructional prose. See ``STRONG_SIGNAL_REQUIRED`` for why this matters.
PROMPT_MARKERS = re.compile(
    r"system prompt|do not reveal|don't reveal|you are an? (ai|assistant|helpful)"
    r"|ignore (previous|prior|all) instructions|your instructions"
    r"|系統提示|系统提示|提示詞|提示词|你的任務|你的任务"
    r"|システムプロンプト|시스템 프롬프트",
    re.IGNORECASE,
)

#: Embedded markup.
#:
#: Both large false-positive classes found in the first corpus run carry HTML:
#: MIT App Inventor ships component documentation as ``<p>A box for the user
#: to enter…``, and one application ships its privacy policy and terms as
#: ``assets/html/*.html`` split into ``<p>`` paragraphs. A prompt written for a
#: model rarely carries tag markup; policy and documentation prose routinely
#: does. This is a penalty rather than a veto because an app may legitimately
#: instruct a model to emit HTML.
MARKUP_LIKE = re.compile(r"</?(p|b|i|br|div|span|ul|li|ol|h[1-6]|strong|em)\s*/?>", re.IGNORECASE)

#: Container paths whose contents are policy, legal or help documents.
#:
#: ``POLICY_LIKE`` matches the *vocabulary* of a policy, but a policy document
#: is extracted one paragraph at a time and most individual paragraphs never
#: mention "privacy policy" or 隐私政策 — the header does, the body does not.
#: That is precisely how one application produced 46 findings from
#: `policy_cn.html` and `protocol_cn.html` while scoring policy_penalty = 0.0
#: on every one. The file path carries the evidence the paragraph lacks.
#:
#: Deliberately restricted to **legal** documents. An earlier revision also
#: matched `help`, `faq`, `guide`, `tutorial`, `about` and `readme`, which
#: cost a true positive: a genuine prompt in `assets/help/assistant.json`
#: scored 0.54 against a 0.55 threshold and was suppressed. Documentation
#: paths are a plausible home for a real prompt; a terms-of-service file is
#: not. Documentation false positives are handled by the strong-signal gate
#: instead, which is the mechanism that actually distinguishes them.
#: ``NOTICES`` is Flutter's bundled third-party licence file and carries no
#: extension, so an earlier revision missed it: MPL 2.0 text ("You may create
#: and distribute a Larger Work under terms of Your choice…") scored 0.65 in
#: one Flutter application. Licence prose is imperative, multi-clause and
#: addressed to a reader, which is why it scores like an instruction.
POLICY_PATH = re.compile(
    r"(polic(y|ies)|privacy|protocol|terms|agreement|eula|licen[cs]e"
    r"|disclaimer|tos[._-]|legal|notices?$|copying$|third[_-]?party)",
    re.IGNORECASE,
)

#: Software licence prose, wherever it is stored.
#:
#: The path check above catches the conventional locations; this catches the
#: text itself when a licence is inlined into a resource or a bundle.
LICENCE_LIKE = re.compile(
    r"larger work|covered software|secondary licenses?|this licen[cs]e"
    r"|redistributions? (of|in) (source|binary)|warranties or conditions"
    r"|GNU (General|Lesser)|Apache Licen[cs]e|MIT Licen[cs]e|Mozilla Public",
    re.IGNORECASE,
)

#: A JSON array of short quoted strings, i.e. a localised UI label list.
#:
#: One application scored 1.0 on an Arabic array of gallery
#: category labels ("create animation sheets", "social media cover images").
#: Concatenated labels look like multi-clause prose to every feature the
#: detector measures, and in a non-Latin script the length gate does not
#: separate them either. The distinguishing shape is the separator density:
#: a prompt is continuous text, a label array is punctuated by `","` every
#: few words.
LABEL_ARRAY = re.compile(r'","')

#: Containers holding machine-generated identifier pools rather than prose.
#:
#: Introduced after Defect 14. Reading assets properly recovered real prompts,
#: and also exposed three new false-positive classes that share one cause:
#: string-extracting a binary yields concatenated identifiers whose adjacency
#: is an artefact of the file format, not of anyone writing a sentence.
#:
#: * Unity IL2CPP `global-metadata.dat` — type and method names run together
#:   ("<Container TypeAsTypeAs TypeCo…"), which scores on length, role
#:   vocabulary and apparent multi-clause structure.
#: * Lottie animation JSON (`res/raw/*.json` with `"nm"` layer names).
#:
#: These are compiler and tool output, so no genuine prompt can live in them.
#:
#: **`resources.arsc` is deliberately NOT in this list.** An earlier revision
#: included it, on the reasoning that it is a compiled binary. That was wrong
#: and cost eight true positives in one app. That app ships a
#: prompt library in its string resources ("I want you to act as a javascript
#: console…", "I want you to act as my legal advisor…"), which is exactly
#: where an Android developer *should* put user-facing prompt text. The
#: resource table is compiled, but its contents are authored. Container format
#: is not evidence about authorship, and treating it as such suppresses the
#: richest source of prompts in the corpus.
#:
#: The penalty is applied rather than a veto because `res/raw/` also
#: legitimately holds prompt files.
MACHINE_POOL_PATH = re.compile(
    r"(global-metadata\.dat|\.pak$|\.dat$"
    r"|/lottie/|animation.*\.json$|\.aab$|R\.txt$)",
    re.IGNORECASE,
)

#: Framework diagnostics that open like an instruction.
#:
#: "You are running with invalid…", "See https://react.dev/link/…" match the
#: instruction-opener feature and clear the strong-signal gate. They address a
#: *developer*, not a model. Distinct from POLICY_LIKE, which addresses a user.
FRAMEWORK_NOISE = re.compile(
    r"you are running|you are using|you are importing|you are calling"
    r"|react\.dev|reactjs\.org|angular\.io|vuejs\.org|flutter\.dev"
    r"|stackoverflow\.com|github\.com/[\w.-]+/issues"
    r"|file an issue|report a bug|upgrade to version|or higher"
    # Narrowed 2026-08-13: "you should use" and "use X instead" are ordinary
    # phrases in an advice-giving prompt and suppressed a genuine one
    # ("I want you to act as a mental health adviser...", 0.247). Framework
    # advisories are identified by their subject, not by generic modals.
    r"|you should migrate to|you must upgrade to|no longer supported"
    r"|has been renamed to|is deprecated|deprecated in favou?r of"
    r"|deprecated|will be removed in|unsupported operand",
    re.IGNORECASE,
)

#: Legal, policy and marketing vocabulary.
#:
#: Terms-of-service, FAQ, prize-draw rules and refund policies are the
#: hardest negatives for this detector: like a system prompt they are
#: multi-sentence, imperative and full of role language. The difference is
#: *who is addressed* — a system prompt instructs the model, a policy
#: instructs the user — which no surface feature captures directly.
#:
#: Validation on one pilot application (which ships genuine LLM prompts
#: localised into ~20 languages under `gpt_desc*` keys, alongside prize-draw
#: rules under `app_Activity Rules`) showed policy text accounting for the
#: majority of that app's false positives.
POLICY_LIKE = re.compile(
    r"terms of (use|service)|privacy policy|refund|subscription|auto-renew"
    r"|billing|liability|warrant(y|ies)|prize|sweepstake|giveaway|winner"
    r"|voucher|coupon|eligibilit|governing law|intellectual property"
    r"|all rights reserved"
    r"|datenschutz|nutzungsbedingungen|gewinnspiel|erstattung|preiseinlösung"
    r"|política de privacidad|términos|reembolso|sorteo|duración de la actividad"
    r"|conditions d'utilisation|remboursement|confidentialité"
    r"|الخصوصية|الشروط"
    r"|隐私政策|隱私政策|服务条款|服務條款|退款|抽獎|抽奖|活動規則|活动规则|訂閱|订阅|條款|条款"
    r"|購読|利用規約|プライバシー|返金|知的財産権|キャンペーン|規約"
    r"|구독|이용약관|개인정보|환불",
    re.IGNORECASE,
)

LOG_LIKE = re.compile(
    r"\b(stacktrace|exception|null pointer|caused by:|at \w+\.\w+:\d+)\b",
    re.IGNORECASE,
)


@dataclass
class PromptScore:
    score: float
    components: dict[str, float]


class SystemPromptScanner(BaseScanner):
    """Heuristic detector for embedded LLM system prompts."""

    name = "system_prompts"

    MIN_LENGTH = 80
    #: Minimum length for runs carrying an independent strong signal (CJK
    #: density or an instruction opener). A Chinese prompt carrying the same
    #: instruction as an 80-character English one is roughly a third the
    #: length, so applying MIN_LENGTH uniformly discards them by construction.
    MIN_LENGTH_SHORT = 28
    #: Fraction of CJK characters above which the CJK minimum applies.
    CJK_DENSITY = 0.25
    DEFAULT_THRESHOLD = 0.55

    #: Ceiling applied to candidates carrying no strong signal.
    #:
    #: A system prompt is text *addressed to a model*. The features that show
    #: this are an instruction opener, fill-in template slots, or explicit
    #: prompt vocabulary. Length, generic role words and multi-sentence
    #: structure show only that the text is instructional prose — which
    #: describes help text, component documentation, terms of service and
    #: recipe instructions equally well.
    #:
    #: Measured on the first corpus run: 1,023 of 1,155 system_prompt findings
    #: scored exactly 0.60 through the bare combination
    #: ``length 0.20 + role_language 0.30 + multi_sentence 0.10``, with no
    #: opener and no template slot. Every one was MIT App Inventor runtime
    #: documentation, in apps with no LLM integration whatsoever. Weighted
    #: scoring alone cannot separate those cases, because on every feature it
    #: measures they genuinely resemble a prompt.
    #:
    #: Set below DEFAULT_THRESHOLD so the bare combination cannot clear the
    #: bar however long or role-dense the text is.
    WEAK_CEILING = 0.50

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def scan(self, sources: list[tuple[str, str]]) -> list[Finding]:
        findings: list[Finding] = []
        for source_file, content in sources:
            for candidate, offset in self._extract_candidates(content):
                score = self._score(candidate, source_file=source_file)
                if score.score >= self.threshold:
                    findings.append(self._make_finding(
                        source_file, candidate, offset, score
                    ))
        return findings

    def _extract_candidates(
        self, content: str
    ) -> list[tuple[str, int]]:
        """
        Pull candidate strings from a content blob.

        We look for quoted long strings (Java/Kotlin string literals
        compile down to UTF-16 entries in classes.dex; Androguard
        surfaces them as Python str). We split on common delimiters
        that are unlikely to appear inside a system prompt.
        """
        # The classes.dex string table is one big concatenation; extract
        # contiguous runs of >= MIN_LENGTH printable characters.
        candidates: list[tuple[str, int]] = []
        run_start = 0
        run_chars: list[str] = []

        for i, ch in enumerate(content):
            if ch.isprintable() and ch not in {"\x00", "\x01"}:
                if not run_chars:
                    run_start = i
                run_chars.append(ch)
            else:
                if self._long_enough(run_chars):
                    candidates.append(("".join(run_chars), run_start))
                run_chars = []

        if self._long_enough(run_chars):
            candidates.append(("".join(run_chars), run_start))

        return candidates

    def _is_cjk_dense(self, text: str) -> bool:
        """Whether enough of the text is CJK to treat it as a dense script."""
        if not text:
            return False
        return len(CJK_RE.findall(text)) / len(text) >= self.CJK_DENSITY

    def _long_enough(self, run_chars: list[str]) -> bool:
        """Whether a printable run is long enough to be a candidate prompt.

        A single character minimum is script-blind, and being script-blind
        here means being biased: the same instruction is ~3x shorter in
        Chinese than English, and an Arabic prompt of 74 characters is a
        perfectly ordinary one. Both were discarded before scoring.

        So the short minimum applies when the run carries an independent
        strong signal — CJK density, or an instruction opener in any
        supported language. Scoring still decides; this gate only controls
        what reaches it.
        """
        if not run_chars:
            return False
        if len(run_chars) >= self.MIN_LENGTH:
            return True
        if len(run_chars) < self.MIN_LENGTH_SHORT:
            return False
        text = "".join(run_chars)
        if self._is_cjk_dense(text):
            return True
        return bool(
            INSTRUCTION_OPENERS.search(text) or INSTRUCTION_OPENERS_INTL.search(text)
        )

    def _score(self, text: str, *, source_file: str = "") -> PromptScore:
        """Score a candidate string in [0, 1].

        ``source_file`` is optional so that scoring stays unit-testable on a
        bare string, but it carries real evidence: a paragraph extracted from
        ``assets/html/policy_cn.html`` is policy text whether or not that
        paragraph happens to contain the word "policy".
        """
        components: dict[str, float] = {}

        # Length contribution, clamped to [0, 1] and scaled to the script.
        #
        # The original expression, min(1.0, (len - 80) / 520), goes *negative*
        # for any candidate shorter than the English minimum — so a short
        # string was penalised rather than merely unrewarded. That is a bug in
        # any language; it bites hardest in CJK, where a complete instruction
        # is routinely 30-40 characters and scored -0.018 for its trouble.
        if self._is_cjk_dense(text):
            base, span = self.MIN_LENGTH_SHORT, 180
        else:
            base, span = self.MIN_LENGTH, 520
        length_norm = max(0.0, min(1.0, (len(text) - base) / span))
        components["length"] = round(0.20 * length_norm, 3)

        # Instruction openers (strong signal).
        # The English openers are anchored to the start of the string; the
        # international set is not, because a DEX string run often begins
        # mid-sentence and CJK has no leading-token boundary to anchor on.
        if INSTRUCTION_OPENERS.search(text) or INSTRUCTION_OPENERS_INTL.search(text):
            components["opener"] = 0.35
        else:
            components["opener"] = 0.0

        # Role-shaping language frequency.
        role_hits = len(ROLE_LANGUAGE.findall(text)) + len(
            ROLE_LANGUAGE_INTL.findall(text)
        )
        components["role_language"] = round(min(0.30, 0.06 * role_hits), 3)

        # Fill-in template slots. Two or more is a strong template signal;
        # one alone is common in ordinary UI copy.
        slots = len(TEMPLATE_SLOT.findall(text))
        components["template_slots"] = 0.15 if slots >= 2 else 0.0

        # Multi-sentence structure.
        # Structure: prose runs across clauses, identifier lists do not.
        # CJK sentences are longer and use fewer separators per unit of
        # content, so one full-width separator there carries about as much
        # evidence as two terminators in a Latin script.
        markers = SENTENCE_TERMINATORS
        required = 2
        if self._is_cjk_dense(text):
            markers += CJK_CLAUSE_SEPARATORS
            required = 1
        sentence_terminators = sum(text.count(c) for c in markers)
        components["multi_sentence"] = (
            0.10 if sentence_terminators >= required else 0.0
        )

        # Policy/marketing text penalty — see POLICY_LIKE.
        components["policy_penalty"] = -0.25 if POLICY_LIKE.search(text) else 0.0

        # Container-path penalty — see POLICY_PATH. Applied independently of
        # the vocabulary penalty because the two carry different evidence:
        # the path says what the document *is*, the vocabulary says what this
        # paragraph *mentions*.
        components["path_penalty"] = (
            -0.25 if source_file and POLICY_PATH.search(source_file) else 0.0
        )

        # Embedded markup penalty — see MARKUP_LIKE.
        components["markup_penalty"] = -0.20 if MARKUP_LIKE.search(text) else 0.0

        # Machine-generated string pools — see MACHINE_POOL_PATH.
        components["machine_pool_penalty"] = (
            -0.30 if source_file and MACHINE_POOL_PATH.search(source_file) else 0.0
        )

        # Framework diagnostics addressed to a developer — see FRAMEWORK_NOISE.
        components["framework_penalty"] = (
            -0.35 if FRAMEWORK_NOISE.search(text) else 0.0
        )

        # Program text — see CODE_LIKE.
        is_code = bool(CODE_LIKE.search(text))
        components["code_penalty"] = -0.40 if is_code else 0.0

        # Software licence prose — see LICENCE_LIKE.
        components["licence_penalty"] = -0.35 if LICENCE_LIKE.search(text) else 0.0

        # Localised label arrays — see LABEL_ARRAY. Scaled by separator
        # density so a prompt that merely quotes a phrase is unaffected.
        seps = len(LABEL_ARRAY.findall(text))
        components["label_array_penalty"] = -0.35 if seps >= 3 else 0.0

        # Penalties.
        if URL_LIKE.search(text):
            components["url_penalty"] = -0.20
        else:
            components["url_penalty"] = 0.0

        if LOG_LIKE.search(text):
            components["log_penalty"] = -0.30
        else:
            components["log_penalty"] = 0.0

        score = max(0.0, min(1.0, sum(components.values())))

        # Strong-signal gate — see WEAK_CEILING. Applied after summing so the
        # recorded components still show what the text actually scored; only
        # the verdict is capped, which keeps the finding auditable.
        # Program text cannot supply a strong signal. Without this, a bundle
        # containing `[i]` clears the gate on syntax alone.
        has_strong = not is_code and (
            components["opener"] > 0.0
            or components["template_slots"] > 0.0
            or bool(PROMPT_MARKERS.search(text))
        )
        components["strong_signal"] = 1.0 if has_strong else 0.0
        if not has_strong:
            score = min(score, self.WEAK_CEILING)

        return PromptScore(score=round(score, 3), components=components)

    def _make_finding(
        self,
        source_file: str,
        candidate: str,
        offset: int,
        score: PromptScore,
    ) -> Finding:
        # Truncate evidence aggressively — a leaked prompt is itself
        # potentially sensitive content for the developer.
        return Finding(
            kind=FindingKind.SYSTEM_PROMPT,
            severity=Severity.MEDIUM,
            title="Embedded system prompt (heuristic match)",
            description=(
                "A high-scoring candidate for an embedded LLM system "
                "prompt was found. Embedded prompts disclose intended "
                "agent behaviour and are a primary reconnaissance "
                "target for prompt-injection adversaries. Consider "
                "moving to server-side composition."
            ),
            location=SourceLocation(
                file_path=source_file, line_or_offset=offset
            ),
            evidence=candidate[:200],
            confidence=score.score,
            provider=None,
            metadata={"score_components": score.components},
        )
