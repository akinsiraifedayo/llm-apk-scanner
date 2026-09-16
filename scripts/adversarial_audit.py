"""Adversarial audit of scanned APKs, especially the ones we called EMPTY.

Independent of the scanner in both extraction and detection:
  * extraction  - raw zipfile, every entry, plus UTF-16 and gzip/nested-zip
                  unwrapping the scanner does not do
  * detection   - a deliberately wider vocabulary than the detectors act on,
                  per METHODOLOGY.md 3.3, so a false negative is visible

Reports DISAGREEMENTS both ways. Silence here is the only evidence we have
that the empty apps are genuinely empty.
"""
import gzip as gz, io, json, os, re, sqlite3, sys, zipfile, collections

WIDE = {
 # --- inference endpoints, broader than the scanner's list ------------------
 'endpoint.openai':    rb'api\.openai\.com|openai\.azure\.com',
 'endpoint.anthropic': rb'api\.anthropic\.com',
 'endpoint.google':    rb'generativelanguage\.googleapis|aiplatform\.googleapis',
 'endpoint.other':     rb'api\.(mistral\.ai|cohere\.(com|ai)|groq\.com|together\.(xyz|ai)'
                       rb'|perplexity\.ai|deepseek\.com|fireworks\.ai|x\.ai|minimax\.chat'
                       rb'|moonshot\.cn|lingyiwanwu\.com)|openrouter\.ai|dashscope\.aliyuncs'
                       rb'|open\.bigmodel\.cn|aip\.baidubce|spark-api\.xf-yun|ark\.cn-beijing\.volces',
 'endpoint.selfhost':  rb'localhost:(11434|8080|5000)|ollama|/v1/chat/completions|/api/generate',
 # --- credentials -----------------------------------------------------------
 'key.openai':    rb'sk-[A-Za-z0-9_\-]{10,}T3BlbkFJ[A-Za-z0-9_\-]{10,}|sk-proj-[A-Za-z0-9_\-]{30,}',
 'key.anthropic': rb'sk-ant-[A-Za-z0-9\-_]{20,}',
 'key.google':    rb'AIza[0-9A-Za-z_\-]{35}',
 'key.other':     rb'gsk_[A-Za-z0-9]{30,}|xai-[A-Za-z0-9]{30,}|sk-or-v1-[A-Za-z0-9]{30,}'
                  rb'|hf_[A-Za-z0-9]{30,}|r8_[A-Za-z0-9]{30,}|pplx-[A-Za-z0-9]{30,}|fw_[A-Za-z0-9]{30,}',
 # --- prompts / agentic -----------------------------------------------------
 'prompt.system':  rb'[Yy]ou are (a|an|the|OI|ChatGPT|Gemini)[ ,][a-zA-Z]',
 'prompt.preset':  rb'I want you to act as',
 'prompt.cjk':     rb'\xe4\xbd\xa0\xe6\x98\xaf|\xe6\x82\xa8\xe6\x98\xaf|\xe4\xbd\xa0\xe7\x9a\x84\xe4\xbb\xbb\xe5\x8a\xa1',
 'prompt.confid':  rb'never reveal|do not reveal|don.t reveal|do not disclose|keep.{0,12}confidential',
 'schema.tool':    rb'"type"\s*:\s*"function"|"input_schema"|"tool_choice"|"function_call"|"tools"\s*:\s*\[',
 'rag.embed':      rb'text-embedding-[\w\-]+|/v1/embeddings|sentence-transformers',
 'rag.store':      rb'chroma\.sqlite3|faiss|\.lance|pinecone|weaviate|qdrant|milvus|pgvector',
 'model.id':       rb'gpt-[345][\w.\-]*|claude-[\w.\-]+|gemini-[\w.\-]+|llama-?[23][\w.\-]*'
                   rb'|qwen[\w.\-]*|mistral-[\w.\-]+|deepseek-[\w.\-]+',
 'sdk.ns':         rb'com/(openai|anthropic|cohereai)|langchain|LlamaIndex',
}
INTERESTING = ('endpoint.', 'key.openai', 'key.anthropic', 'key.other',
               'prompt.system', 'prompt.preset', 'prompt.confid',
               'schema.tool', 'rag.', 'sdk.ns')

def unwrap(name, data, depth=0):
    """Yield (label, bytes) for the entry and anything nested inside it."""
    yield name, data
    if depth >= 2 or len(data) > 40_000_000:
        return
    if data[:2] == b'\x1f\x8b':
        try: yield from unwrap(name+'!gz', gz.decompress(data), depth+1)
        except Exception: pass
    elif data[:2] == b'PK' and name != '<root>':
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z2:
                for i2 in z2.infolist()[:400]:
                    try: yield from unwrap(f'{name}!{i2.filename}', z2.read(i2), depth+1)
                    except Exception: pass
        except Exception: pass

def audit(path):
    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.file_size > 80_000_000: continue
            try: raw = z.read(info.filename)
            except Exception: continue
            for label, data in unwrap(info.filename, raw):
                # search raw bytes AND a utf-16 -> ascii fold
                probes = [data]
                if b'\x00' in data[:2000]:
                    probes.append(data.replace(b'\x00', b''))
                for probe in probes:
                    for name, pat in WIDE.items():
                        for m in re.findall(pat, probe):
                            v = m if isinstance(m, bytes) else m[0]
                            hits[name][label].add(v[:70].decode('utf8','replace'))
    return hits

def main(paths):
    con = sqlite3.connect('corpus/ledger_v2.db')
    for p in paths:
        sha = os.path.splitext(os.path.basename(p))[0].upper()
        row = con.execute("select pkg_name,n_findings,llm_integrated from apk where sha256=?", (sha,)).fetchone()
        pkg, n, llm = row if row else ('?', None, None)
        dp = f'results/detail_v2/{sha}.json.gz'
        rep = []
        if os.path.exists(dp):
            d = json.load(gz.open(dp,'rt'))
            rep = [(f['kind'], f['evidence'][:44]) for f in d['findings']]
        hits = audit(p)
        interesting = {k:v for k,v in hits.items() if any(k.startswith(i) for i in INTERESTING)}
        verdict = 'AGREE'
        if not rep and interesting: verdict = '*** SCANNER SAID EMPTY, AUDIT FOUND SIGNAL ***'
        elif rep and not interesting: verdict = '(scanner found; audit saw only weak signal)'
        print(f'\n{"="*80}\n{pkg}  [scanner: {n} findings, llm={llm}]  {verdict}')
        print(f'  scanner reported: {rep if rep else "nothing"}')
        if not interesting:
            print('  audit: no strong signal')
        for k in sorted(interesting):
            for loc, vals in sorted(interesting[k].items()):
                print(f'    {k:16} {loc[:44]:44} {sorted(vals)[:2]}')

if __name__ == '__main__':
    main(sys.argv[1:])
