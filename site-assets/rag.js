/* Browser port of thesis-rag (github.com/poojaprasadrn/thesis-rag): the same LangGraph flow
   rewrite -> retrieve -> grade -> (generate | rewrite once more | clarify), with the app's
   no-API-key extractive generator. Index built by ingest.py (120-word chunks, 30 overlap,
   TF-IDF 1-2-grams, sublinear tf). Generation with Claude runs only in the Python app. */
(function (root) {
  "use strict";
  var TOP_K = 6, MIN_SCORE = 0.08;
  var CLARIFY = "I couldn't find that in the thesis. Try asking about the annotation study, the " +
    "classifiers, the SLURM batch inference or the trend results.";
  var WORD = /[\p{L}\p{N}_]+/gu, TOKEN = /[\p{L}\p{N}_]{2,}/gu;
  var idx = null, termId = null, stop = null;

  function load() {
    if (idx) return idx;
    idx = root.RAG_INDEX;
    termId = new Map(idx.terms.map(function (t, i) { return [t, i]; }));
    stop = new Set(idx.stop);
    return idx;
  }

  function rewriteQuery(question) {
    var q = question.replace(/\b(please|can you|could you|tell me|explain|what is|what are|what's)\b/gi, " ");
    q = q.replace(/\s+/g, " ").replace(/^[ ?.]+|[ ?.]+$/g, "");
    q = q || question;
    var expansions = [["agreement", "Krippendorff alpha average"], ["annotator", "annotators ratings"],
      ["best", "performed best accuracy"], ["accuracy", "accuracy F1"], ["cluster", "SLURM Docker nodes"]];
    var low = q.toLowerCase();
    var extra = expansions.filter(function (e) { return low.indexOf(e[0]) !== -1; }).map(function (e) { return e[1]; });
    return (q + " " + extra.join(" ")).trim();
  }

  // Same analyzer as scikit-learn's TfidfVectorizer(lowercase, stop_words="english", ngram_range=(1, 2))
  function analyze(text) {
    var toks = (text.toLowerCase().match(TOKEN) || []).filter(function (t) { return !stop.has(t); });
    var out = toks.slice();
    for (var i = 0; i + 1 < toks.length; i++) out.push(toks[i] + " " + toks[i + 1]);
    return out;
  }

  function retrieve(query, k) {
    load();
    k = k || TOP_K;
    var tf = new Map();
    analyze(query).forEach(function (t) { if (termId.has(t)) tf.set(t, (tf.get(t) || 0) + 1); });
    var q = [], norm = 0;
    tf.forEach(function (c, t) { var id = termId.get(t), w = (1 + Math.log(c)) * idx.idf[id]; q.push([id, w, t]); norm += w * w; });
    norm = Math.sqrt(norm) || 1;
    var scores = new Float64Array(idx.chunks.length);
    var hits = new Map();
    q.forEach(function (e) {
      var post = idx.postings[e[0]] || [], w = e[1] / norm;
      post.forEach(function (p) {
        scores[p[0]] += w * p[1];
        if (!hits.has(p[0])) hits.set(p[0], []);
        hits.get(p[0]).push(e[2]);
      });
    });
    var order = Array.from(scores.keys()).sort(function (a, b) { return scores[b] - scores[a]; }).slice(0, k);
    return order.map(function (i) {
      return { id: i, page: idx.chunks[i].page, text: idx.chunks[i].text, score: scores[i], terms: hits.get(i) || [] };
    });
  }

  function grade(docs) { return docs.length > 0 && docs[0].score >= MIN_SCORE; }

  function extractiveAnswer(question, docs) {
    var stopQ = new Set(["the", "a", "an", "of", "in", "was", "is", "what", "how", "which", "did", "do", "and", "to", "for", "on"]);
    var qwords = new Set((question.toLowerCase().match(WORD) || []).filter(function (w) { return !stopQ.has(w); }));
    var wantsNumber = /\b(how many|how much|what was|what were|accuracy|score|f1|agreement|alpha)\b/i.test(question);
    var scored = [];
    docs.forEach(function (d) {
      d.text.split(/(?<=[.!?])\s+/).forEach(function (sent) {
        var words = new Set(sent.toLowerCase().match(WORD) || []);
        var overlap = 0;
        qwords.forEach(function (w) { if (words.has(w)) overlap++; });
        if (!overlap || sent.length < 25) return;
        var bonus = wantsNumber && /\d/.test(sent) ? (/\d\.\d/.test(sent) ? 3 : 1) : 0;
        scored.push({ s: overlap + bonus + d.score, sent: sent.trim(), page: d.page });
      });
    });
    if (!scored.length) return { text: docs[0].text.slice(0, 300), pages: [docs[0].page] };
    scored.sort(function (a, b) { return b.s - a.s; });
    var out = [], seen = new Set();
    for (var i = 0; i < scored.length && out.length < 2; i++) {
      if (!seen.has(scored[i].sent)) { out.push(scored[i]); seen.add(scored[i].sent); }
    }
    return { text: out.map(function (o) { return o.sent; }).join(" "), pages: out.map(function (o) { return o.page; }) };
  }

  // Runs the graph and returns a step-by-step trace for the UI
  function run(question) {
    var trace = [], state = { question: question, attempts: 0 };
    for (;;) {
      state.query = state.attempts === 0 ? rewriteQuery(question) : question;
      state.attempts += 1;
      trace.push({ node: "rewrite", query: state.query, attempt: state.attempts });
      state.docs = retrieve(state.query);
      trace.push({ node: "retrieve", docs: state.docs });
      state.relevant = grade(state.docs);
      trace.push({ node: "grade", relevant: state.relevant, best: state.docs.length ? state.docs[0].score : 0 });
      if (state.relevant) {
        var ans = extractiveAnswer(question, state.docs);
        var pages = Array.from(new Set(state.docs.map(function (d) { return d.page; }))).sort(function (a, b) { return a - b; });
        trace.push({ node: "generate", answer: ans.text, answerPages: ans.pages, pages: pages });
        break;
      }
      if (state.attempts >= 2) { trace.push({ node: "clarify", answer: CLARIFY }); break; }
    }
    return trace;
  }

  root.ThesisRAG = { run: run, rewriteQuery: rewriteQuery, retrieve: retrieve, grade: grade,
    extractiveAnswer: extractiveAnswer, MIN_SCORE: MIN_SCORE, TOP_K: TOP_K, size: function () { return load().chunks.length; } };
})(typeof window !== "undefined" ? window : globalThis);
