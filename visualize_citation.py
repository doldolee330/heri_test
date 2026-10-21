# -*- coding: utf-8 -*-
"""
T-003 해설-근거 매칭 시각화 뷰어 생성기.

평가셋 JSONL({질문, 근거문서, 답변, ...})에서 유물을 표본 추출해,
생성 해설의 각 어절이 근거문서에 존재하는지(측정 지표와 동일한
어간 매칭 규칙)를 양방향으로 하이라이트한 자체완결 HTML을 만든다.

- 해설 쪽: 문장별 판정(인용가능/미달/판정제외) + 근거에 존재하는 어절 하이라이트
- 근거 쪽: 해설에서 사용된 어절 하이라이트, 해설 문장 클릭 시 해당 문장이
  활성화한 근거 어절만 강조
- 오프라인 시연 가능(외부 리소스는 Google Fonts뿐, 없어도 시스템 폰트로 동작)

사용 예:
  python visualize_citation.py \
      --input serving_local/eval_set_full_explanation.jsonl \
      --evidence serving_local/data/aimuse_items_enriched_final_v6.json \
      --num 15 --output output/citation_viewer.html
"""
import argparse
import html
import json
import random

from citation_metric import (CITATION_THRESHOLD, MIN_SENT_CHARS, citation_score,
                             is_abstention, is_interactive, judge_answer,
                             make_char_ngrams, normalize, normalize_joined,
                             split_sentences, stem)


def build_summary(rows, threshold, n_bins=20):
    """입력 파일 전체에 대한 집계(지표 표·히스토그램용). 측정 스크립트와 동일 판정."""
    s = {"nItems": 0, "nJudged": 0, "nCitable": 0,
         "exInteractive": 0, "exAbstention": 0, "exShort": 0}
    hist = [0] * n_bins
    for r in rows:
        ans = r.get("답변") or r.get("해설") or ""
        if not ans.strip():
            continue
        j = judge_answer(ans, r["근거문서"], threshold=threshold)
        s["nItems"] += 1
        s["nJudged"] += j["n_total"]
        s["nCitable"] += j["n_citable"]
        s["exInteractive"] += j["n_interactive"]
        s["exAbstention"] += j["n_abstention"]
        s["exShort"] += j["n_short"]
        for _, score, _ok in j["sentences"]:
            hist[min(n_bins - 1, int(score * n_bins))] += 1
    s["acc"] = round(s["nCitable"] / s["nJudged"] * 100, 2) if s["nJudged"] else 0.0
    s["threshold"] = threshold
    s["target"] = 70.0
    return s, hist


def chunk_tokens(text):
    """원문을 공백 단위 어절로 분리하되 원문 표기를 보존한다."""
    out, cur = [], ""
    for ch in text:
        if ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def stem_of(token):
    """측정 지표와 동일한 어간 (citation_metric.stem 사용 — 정의 단일화)."""
    w = normalize(token).replace(" ", "")
    if len(w) < 2:
        return None
    return stem(w)


NLI_LABELS = ["완전함의", "부분함의", "중립", "모순"]


def build_item(row, relic_names, threshold, nli_map=None):
    answer, evidence = row["답변"], row["근거문서"]
    ev_joined = normalize_joined(evidence)
    ev_grams = make_char_ngrams(evidence)
    ans_joined = normalize_joined(answer)

    # ---- 해설: 문장별 판정 + 어절 매칭 ----
    sentences = []
    n_citable = n_judged = 0
    for sent in split_sentences(answer):
        if len(normalize_joined(sent)) < MIN_SENT_CHARS:
            status = "short"
        elif is_interactive(sent):
            status = "interactive"
        elif is_abstention(sent):
            status = "abstention"
        else:
            status = None
        score = citation_score(sent, ev_joined, ev_grams)
        nli = None
        if status is None:
            if nli_map is not None:
                nli = nli_map.get(n_judged)  # 판정 대상 문장 순번 = measure_nli_judge의 si
            n_judged += 1
            status = "citable" if score >= threshold else "low"
            if status == "citable":
                n_citable += 1
        toks = []
        for t in chunk_tokens(sent):
            if t.isspace():
                toks.append([t, 0])
                continue
            st = stem_of(t)
            toks.append([t, 1 if (st and st in ev_joined) else 0])
        s = {"toks": toks, "score": round(score, 3), "status": status}
        if nli:
            s["nli"] = nli
        sentences.append(s)

    # ---- 근거: 어절별 '해설에서 사용됨' + 문장별 활성 목록 ----
    ev_tokens = []           # [원문어절, 전체매칭여부]
    word_indices = []        # 공백 아닌 어절의 인덱스 목록
    for t in chunk_tokens(evidence):
        if t.isspace():
            ev_tokens.append([t, 0])
            continue
        st = stem_of(t)
        ev_tokens.append([t, 1 if (st and st in ans_joined) else 0])
        word_indices.append(len(ev_tokens) - 1)

    # 문장 클릭 시 강조: 해당 문장 어절 어간이 포함하는 근거 어절
    sent_to_ev = []
    for s in sentences:
        sj = normalize_joined("".join(t for t, _ in s["toks"]))
        act = [i for i in word_indices
               if ev_tokens[i][1] and (stem_of(ev_tokens[i][0]) or "") in sj]
        sent_to_ev.append(act)

    rid = str(row.get("유물ID", ""))
    return {
        "id": rid,
        "name": relic_names.get(rid, ""),
        "visitor": row.get("관람객유형", ""),
        "question": row.get("질문", ""),
        "acc": round(n_citable / n_judged * 100, 1) if n_judged else 0.0,
        "nJudged": n_judged,
        "nCitable": n_citable,
        "sentences": sentences,
        "evTokens": ev_tokens,
        "sentToEv": sent_to_ev,
    }


def stratified_sample(idx_rows, n, threshold, seed=42):
    """항목 정확도 기준 상/중/하 고르게 표본 추출. 입력/반환은 (행번호, 행) 쌍."""
    random.seed(seed)
    pool = random.sample(idx_rows, min(len(idx_rows), max(n * 20, 200)))
    scored = []
    for ir in pool:
        j = judge_answer(ir[1]["답변"], ir[1]["근거문서"], threshold=threshold)
        if j["n_total"] == 0:
            continue
        scored.append((j["n_citable"] / j["n_total"], ir))
    scored.sort(key=lambda x: x[0])
    k = n // 3
    lo = scored[:len(scored) // 4] or scored
    mid = scored[len(scored) // 3:2 * len(scored) // 3] or scored
    hi = scored[-len(scored) // 4:] or scored
    picked = (random.sample(lo, min(k, len(lo)))
              + random.sample(mid, min(k, len(mid)))
              + random.sample(hi, min(n - 2 * k, len(hi))))
    return [r for _, r in picked]


STANDALONE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n')

TEMPLATE = """<title>해설 근거 하이라이트</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Noto+Serif+KR:wght@600&display=swap">
<style>
/* 레이아웃: 상단 요약 바 + 좌(해설)/우(근거) 2패널, 폰 폭에서 세로 스택 */
:root{
  --bg:#f7f6f2; --panel:#ffffff; --ink:#26312e; --muted:#6b7672;
  --line:#dcdfd8; --accent:#2e7d6b;            /* 청자 녹청 */
  --hl:#cdeadf; --hl-strong:#8fd3bb; --hl-ink:#174f41; --hover:#eaf2ee;
  --warn:#b97324; --warn-bg:#f7ead7; --dim:#9aa39f;
  --chart-ok:#1e8a68; --chart-low:#b97324;     /* 차트 상태색 (CVD 검증) */
  --nli-full:#1e8a68; --nli-part:#3d74c0; --nli-contra:#c2452d; --nli-contra-bg:#f7ddd6;
  --sans:'Noto Sans KR',-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
  --serif:'Noto Serif KR','Apple SD Gothic Neo',serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#171c1a; --panel:#1f2522; --ink:#e3e8e4; --muted:#93a09a;
  --line:#343c38; --accent:#5cbfa4;
  --hl:#24473c; --hl-strong:#2f6d58; --hl-ink:#bfe8d8; --hover:#26312c;
  --warn:#d79a4e; --warn-bg:#3c2f1c; --dim:#5e6964;
  --chart-ok:#2f9d7e; --chart-low:#c0792b;
  --nli-full:#2f9d7e; --nli-part:#5f8ae0; --nli-contra:#cc5a40; --nli-contra-bg:#46241c; color-scheme:dark; } }
:root[data-theme="dark"]{
  --bg:#171c1a; --panel:#1f2522; --ink:#e3e8e4; --muted:#93a09a;
  --line:#343c38; --accent:#5cbfa4;
  --hl:#24473c; --hl-strong:#2f6d58; --hl-ink:#bfe8d8; --hover:#26312c;
  --warn:#d79a4e; --warn-bg:#3c2f1c; --dim:#5e6964;
  --chart-ok:#2f9d7e; --chart-low:#c0792b;
  --nli-full:#2f9d7e; --nli-part:#5f8ae0; --nli-contra:#cc5a40; --nli-contra-bg:#46241c; color-scheme:dark; }
body{background:var(--bg);color:var(--ink);font-family:var(--sans);
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:20px;}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:14px;}
header h1{font-family:var(--serif);font-weight:600;font-size:1.45rem;margin:0;text-wrap:balance;}
header .sub{color:var(--muted);font-size:.86rem;margin-top:4px;}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;}
.bar select{flex:1 1 260px;min-width:0;padding:8px 10px;border:1px solid var(--line);
  border-radius:8px;background:var(--panel);color:var(--ink);font:inherit;font-size:.9rem;}
.bar button{padding:8px 14px;border:1px solid var(--line);border-radius:8px;
  background:var(--panel);color:var(--ink);font:inherit;cursor:pointer;}
.bar button:hover{border-color:var(--accent);}
.bar button:focus-visible,.bar select:focus-visible{outline:2px solid var(--accent);outline-offset:1px;}
.stats{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline;
  border:1px solid var(--line);border-radius:10px;background:var(--panel);
  padding:10px 14px;font-size:.88rem;}
.stats b{font-size:1.05rem;font-variant-numeric:tabular-nums;}
.stats .acc{color:var(--accent);}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:.8rem;color:var(--muted);align-items:center;}
.legend .sw{display:inline-block;width:.95em;height:.95em;border-radius:3px;vertical-align:-2px;margin-right:5px;}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;}
@media (max-width:860px){.panes{grid-template-columns:1fr;}}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:12px;min-width:0;}
.pane h2{font-size:.82rem;letter-spacing:.08em;color:var(--muted);font-weight:700;
  margin:0;padding:10px 16px;border-bottom:1px solid var(--line);}
.pane .body{padding:14px 16px 18px;line-height:1.85;font-size:.95rem;
  max-height:72vh;overflow-y:auto;overscroll-behavior:contain;}
mark{background:var(--hl);color:var(--hl-ink);border-radius:3px;padding:0 1px;}
mark.on{background:var(--hl-strong);}
.sent{display:block;border-left:3px solid transparent;padding:2px 8px;margin:2px 0;
  border-radius:4px;cursor:pointer;}
.sent:hover{background:var(--hover);background:color-mix(in srgb,var(--accent) 7%,transparent);}
.sent.sel,.sent:focus-visible{outline:2px solid var(--accent);outline-offset:-1px;}
.sent.citable{border-left-color:var(--accent);}
.sent.low{border-left-color:var(--warn);background:var(--warn-bg);}
.sent.excl{border-left-color:var(--dim);color:var(--muted);}
/* NLI 모드: 라벨이 판정의 1차 기준 */
.nli .sent.n-full{border-left-color:var(--nli-full);background:transparent;}
.nli .sent.n-part{border-left-color:var(--nli-part);background:transparent;}
.nli .sent.n-neutral{border-left-color:var(--dim);background:transparent;color:inherit;}
.nli .sent.n-contra{border-left-color:var(--nli-contra);background:var(--nli-contra-bg);}
.nli .sent .tag .nb{font-weight:700;}
.nli .sent.n-full .nb{color:var(--nli-full);} .nli .sent.n-part .nb{color:var(--nli-part);}
.nli .sent.n-neutral .nb{color:var(--muted);} .nli .sent.n-contra .nb{color:var(--nli-contra);}
#nlibar svg{width:100%;height:auto;display:block;}
.sent .tag{font-size:.7rem;color:var(--muted);margin-left:6px;white-space:nowrap;
  font-variant-numeric:tabular-nums;}
.sent.low .tag{color:var(--warn);}
.q{font-size:.85rem;color:var(--muted);border:1px dashed var(--line);border-radius:8px;
  padding:8px 12px;}
footer{color:var(--muted);font-size:.78rem;line-height:1.6;}
/* 전체 평가 결과 패널 */
.overall{display:grid;grid-template-columns:minmax(290px,2fr) 3fr;gap:14px;align-items:stretch;}
@media (max-width:860px){.overall{grid-template-columns:1fr;}}
.overall .pane .body{max-height:none;overflow:visible;}
.mtable{width:100%;border-collapse:collapse;font-size:.88rem;}
.mtable td{padding:6px 4px;border-bottom:1px solid var(--line);}
.mtable td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:500;}
.mtable tr:last-child td{border-bottom:none;}
.hero{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-top:12px;}
.hero .num{font-size:2.1rem;font-weight:700;color:var(--chart-ok);font-variant-numeric:tabular-nums;}
.hero .pill{font-size:.78rem;font-weight:700;letter-spacing:.06em;padding:3px 10px;border-radius:999px;
  background:var(--hl);color:var(--hl-ink);}
.hero .pill.fail{background:var(--warn-bg);color:var(--warn);}
.hero .goal{color:var(--muted);font-size:.84rem;}
.chartbox{min-width:0;}
.chartbox svg{width:100%;height:auto;display:block;}
.chartlegend{display:flex;gap:16px;font-size:.78rem;color:var(--muted);margin-top:6px;flex-wrap:wrap;}
.chartlegend .sw{display:inline-block;width:.9em;height:.9em;border-radius:3px;vertical-align:-1.5px;margin-right:5px;}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:4px 8px;font-size:.78rem;box-shadow:0 2px 8px rgba(0,0,0,.12);z-index:9;}
</style>
<div class="wrap">
<header>
  <h1>해설 근거 하이라이트</h1>
  <div class="sub">T-003 Citation Accuracy 측정과 동일한 어간 매칭 기준 — 생성 해설의 어절이 근거문서에 존재하면 같은 색으로 표시. 해설 문장을 클릭하면 그 문장이 딛고 선 근거 위치가 진하게 강조됩니다.</div>
</header>
<div class="overall">
  <section class="pane">
    <h2 id="ovtitle">전체 평가 결과</h2>
    <div class="body">
      <table class="mtable" id="mtable"></table>
      <div class="hero" id="hero"></div>
    </div>
  </section>
  <section class="pane chartbox">
    <h2 id="charttitle">판정 문장 일치도 분포</h2>
    <div class="body">
      <div id="nlibar" hidden></div>
      <div id="hist"></div>
      <div class="chartlegend" id="chartlegend">
        <span><span class="sw" style="background:var(--chart-ok)"></span>인용 가능 (일치도 ≥ __THRESHOLD__)</span>
        <span><span class="sw" style="background:var(--chart-low)"></span>일치도 미달</span>
      </div>
    </div>
  </section>
</div>
<div class="bar">
  <button id="prev" aria-label="이전 유물">◀ 이전</button>
  <select id="sel" aria-label="유물 선택"></select>
  <button id="next" aria-label="다음 유물">다음 ▶</button>
</div>
<div class="stats" id="stats"></div>
<div class="legend">
  <span><span class="sw" style="background:var(--hl)"></span>근거문서와 겹치는 어절</span>
  <span><span class="sw" style="background:var(--hl-strong)"></span>선택 문장의 근거 위치</span>
  <span><span class="sw" style="border-left:3px solid var(--accent);background:transparent"></span>인용 가능 문장 (일치도 ≥ __THRESHOLD__)</span>
  <span><span class="sw" style="background:var(--warn-bg)"></span>일치도 미달 문장</span>
  <span><span class="sw" style="background:transparent;border:1px solid var(--dim)"></span>판정 제외(권유·유보·단문)</span>
</div>
<div class="q" id="qbox"></div>
<div class="panes">
  <section class="pane"><h2>생성 해설 (CESS_NatLang)</h2><div class="body" id="ans"></div></section>
  <section class="pane"><h2>근거문서</h2><div class="body" id="ev"></div></section>
</div>
<footer>문장 일치도 = max(어절 어간 커버리지, 문자 2~3gram 포함률), 판정 임계치 __THRESHOLD__ — 측정 스크립트(measure_citation_accuracy.py)와 동일. 하이라이트는 어절 커버리지 기준 표시(2자 미만 어절은 분모 제외)라, 표시 일치도가 칠해진 비율보다 높을 수 있습니다. 표본은 항목 정확도 상·중·하에서 고르게 추출한 참고용이며, 항목 수치·평균은 공식 지표가 아닙니다.__OFFICIAL__</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const CFG=__CFG__;
const DATA=JSON.parse(document.getElementById('data').textContent);

// ---- 전체 평가 결과 패널 ----
(function renderOverall(){
  const S=CFG.summary; if(!S) return;
  const N=CFG.nli;
  if(N) document.querySelector('.wrap').classList.add('nli');
  if(CFG.label) document.getElementById('ovtitle').textContent='전체 평가 결과 — '+CFG.label;
  const fmt=n=>n.toLocaleString('ko-KR');
  const rows=[
    ['평가 유물 수', fmt(S.nItems)+' 종'],
    ['판정 대상 문장 수', fmt(S.nJudged)+' 개'],
  ];
  if(N){
    rows.push(['완전함의 / 부분함의', fmt(N.counts['완전함의'])+' / '+fmt(N.counts['부분함의'])+' 개']);
    rows.push(['중립 / 모순', fmt(N.counts['중립'])+' / '+fmt(N.counts['모순'])+' 개']);
    rows.push(['판정자 (NLI)', N.judge+' · temperature 0']);
    rows.push(['참고: 어휘 일치도 기준', S.acc.toFixed(2)+'%']);
  }else{
    rows.push(['인용 가능 문장 수', fmt(S.nCitable)+' 개']);
    rows.push(['일치도 임계치', S.threshold.toFixed(2)]);
  }
  rows.push(['판정 제외 (권유·상호작용)', fmt(S.exInteractive)+' 개']);
  rows.push(['판정 제외 (판단 유보 / 단문)', fmt(S.exAbstention)+' / '+fmt(S.exShort)+' 개']);
  document.getElementById('mtable').innerHTML=
    rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
  const accMain=N?N.acc:S.acc;
  const pass=accMain>=S.target;
  document.getElementById('hero').innerHTML=
    `<span class="num">${accMain.toFixed(2)}%</span>`+
    `<span class="pill${pass?'':' fail'}">${pass?'통과':'미달'}</span>`+
    `<span class="goal">Citation Accuracy${N?' (완전함의+부분함의)':''} · 성능목표 ${S.target.toFixed(0)}% 이상`+
    (N?`<br>완전함의 단독 ${N.accFull.toFixed(2)}%`:'')+`</span>`;
  if(N){
    document.getElementById('charttitle').textContent='NLI 판정 라벨 구성';
    const order=[['완전함의','var(--nli-full)'],['부분함의','var(--nli-part)'],['중립','var(--dim)'],['모순','var(--nli-contra)']];
    const total=N.total, W=560, BH=64, T=8, B=26, bh=BH-T-B;
    let x=0, segs='', labs='';
    order.forEach(([k,c])=>{
      const v=N.counts[k]||0; if(!v) return;
      const w=(W*v/total);
      segs+=`<rect x="${x.toFixed(1)}" y="${T}" width="${Math.max(0,w-2).toFixed(1)}" height="${bh}" rx="3" fill="${c}" data-tip="${k} ${v.toLocaleString('ko-KR')}문장 (${(v/total*100).toFixed(1)}%)"/>`;
      if(w>56) labs+=`<text x="${(x+w/2).toFixed(1)}" y="${T+bh/2+4}" text-anchor="middle" font-size="11" font-weight="600" fill="#fff">${(v/total*100).toFixed(1)}%</text>`;
      x+=w;
    });
    document.getElementById('nlibar').hidden=false;
    document.getElementById('nlibar').innerHTML=
      `<svg viewBox="0 0 ${W} ${BH}" role="img" aria-label="NLI 판정 라벨 구성 누적 막대">${segs}${labs}`+
      `<text x="0" y="${BH-8}" font-size="10" fill="var(--muted)">판정 대상 ${total.toLocaleString('ko-KR')}문장 중 라벨 비율</text></svg>`;
    document.getElementById('chartlegend').innerHTML=
      order.map(([k,c])=>`<span><span class="sw" style="background:${c}"></span>${k} ${fmt(N.counts[k]||0)}</span>`).join('')+
      `<span style="flex-basis:100%">아래: 어휘 일치도 분포 (참고 지표)</span>`;
  }
  // 히스토그램 (SVG) — CFG.hist가 없으면 카드 숨김
  if(!CFG.hist){document.querySelector('.chartbox').hidden=true;}
  const H=CFG.hist||[], n=H.length, th=S.threshold;
  const W=560, HT=230, L=44, R=10, T=14, B=30, pw=W-L-R, ph=HT-T-B;
  const ymax=Math.max(1,...H);
  const okC='var(--chart-ok)', lowC='var(--chart-low)';
  let bars='';
  for(let i=0;i<n;i++){
    const x=L+pw*i/n, bw=pw/n-2, h=ph*H[i]/ymax, y=T+ph-h;
    const lo=i/n, hi=(i+1)/n;
    const c=(lo+1e-9)>=th?okC:lowC;
    bars+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1,bw).toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${c}" data-tip="일치도 ${lo.toFixed(2)}–${hi.toFixed(2)} · ${H[i].toLocaleString('ko-KR')}문장"/>`;
  }
  let grid='',xlab='';
  for(let g=0;g<=4;g++){
    const v=Math.round(ymax*g/4), y=T+ph-ph*g/4;
    grid+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`+
          `<text x="${L-6}" y="${y+4}" text-anchor="end" font-size="10" fill="var(--muted)">${v.toLocaleString('ko-KR')}</text>`;
  }
  for(let t=0;t<=5;t++){
    const v=t/5, x=L+pw*v;
    xlab+=`<text x="${x}" y="${HT-10}" text-anchor="middle" font-size="10" fill="var(--muted)">${v.toFixed(1)}</text>`;
  }
  const tx=L+pw*th;
  const thline=`<line x1="${tx}" y1="${T-4}" x2="${tx}" y2="${T+ph}" stroke="var(--ink)" stroke-dasharray="4 3" stroke-width="1.4"/>`+
    `<text x="${tx+5}" y="${T+8}" font-size="10.5" fill="var(--ink)">임계치 ${th.toFixed(2)}</text>`;
  document.getElementById('hist').innerHTML=
    `<svg viewBox="0 0 ${W} ${HT}" role="img" aria-label="판정 문장 일치도 분포 히스토그램">${grid}${bars}${thline}${xlab}`+
    `<text x="${L+pw/2}" y="${HT-0.5}" text-anchor="middle" font-size="10" fill="var(--muted)">문장-근거문서 일치도</text></svg>`;
  // 호버 툴팁
  const tip=document.createElement('div');tip.id='tip';tip.hidden=true;document.body.appendChild(tip);
  document.querySelectorAll('#hist svg, #nlibar svg').forEach(svg=>{
    svg.addEventListener('mousemove',e=>{
      const t=e.target.closest('rect[data-tip]');
      if(!t){tip.hidden=true;return;}
      tip.textContent=t.dataset.tip;tip.hidden=false;
      tip.style.left=Math.min(window.innerWidth-170,e.clientX+12)+'px';
      tip.style.top=(e.clientY-30)+'px';
    });
    svg.addEventListener('mouseleave',()=>{tip.hidden=true;});
  });
})();
const sel=document.getElementById('sel'),ans=document.getElementById('ans'),
      ev=document.getElementById('ev'),stats=document.getElementById('stats'),
      qbox=document.getElementById('qbox');
let cur=0,selSent=-1;
DATA.forEach((it,i)=>{
  const o=document.createElement('option');o.value=i;
  o.textContent=`${it.id} ${it.name||''} — 인용가능 ${it.acc}%`;sel.appendChild(o);
});
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function render(){
  const it=DATA[cur];sel.value=cur;selSent=-1;
  stats.innerHTML=`<span>유물 <b>${esc(it.id)}</b> ${esc(it.name||'')}</span>`+
    `<span>판정 문장 <b>${it.nJudged}</b></span>`+
    `<span>인용 가능 <b>${it.nCitable}</b></span>`+
    `<span>항목 Citation Accuracy <b class="acc">${it.acc}%</b></span>`;
  qbox.textContent = CFG.task==='qa'
    ? '요청(질의응답): "'+it.question+'" · 관람객 유형 '+(it.visitor||'')
    : '요청(해설 생성): "\\''+(it.name||it.id)+'\\'에 대해 설명해 주세요." · 관람객 유형 '+(it.visitor||'')+' (페르소나로 반영, 별도 질문은 입력되지 않음)';
  ans.innerHTML=it.sentences.map((s,si)=>{
    let cls=s.status==='citable'?'citable':s.status==='low'?'low':'excl';
    let label=s.status==='citable'||s.status==='low'
      ?`일치도 ${s.score.toFixed(2)}`
      :({interactive:'권유·상호작용',abstention:'판단 유보',short:'단문'}[s.status]||'판정 제외');
    if(CFG.nli&&s.nli){
      cls+=' '+({'완전함의':'n-full','부분함의':'n-part','중립':'n-neutral','모순':'n-contra'}[s.nli]||'');
      label=`<span class="nb">${s.nli}</span> · 일치도 ${s.score.toFixed(2)}`;
    }
    const body=s.toks.map(([t,m])=>/^\\s+$/.test(t)?esc(t):(m?'<mark>'+esc(t)+'</mark>':esc(t))).join('');
    return `<span class="sent ${cls}" data-si="${si}" role="button" tabindex="0" aria-pressed="false">${body}<span class="tag">${label}</span></span>`;
  }).join('');
  ev.innerHTML=it.evTokens.map(([t,m],ti)=>{
    if(/^\\s+$/.test(t))return esc(t).replace(/\\n/g,'<br>');
    return m?`<mark data-ti="${ti}">${esc(t)}</mark>`:esc(t);
  }).join('');
}
function selectSent(si){
  selSent=(selSent===si)?-1:si;
  const it=DATA[cur];
  ans.querySelectorAll('.sent').forEach(el=>{
    const on=+el.dataset.si===selSent;
    el.classList.toggle('sel',on);el.setAttribute('aria-pressed',on?'true':'false');
  });
  const act=new Set(selSent>=0?it.sentToEv[selSent]:[]);
  let first=null;
  ev.querySelectorAll('mark').forEach(el=>{
    const on=act.has(+el.dataset.ti);
    el.classList.toggle('on',on);
    if(on&&!first)first=el;
  });
  if(first)first.scrollIntoView({block:'center',behavior:'smooth'});
}
ans.addEventListener('click',e=>{const s=e.target.closest('.sent');if(s)selectSent(+s.dataset.si);});
ans.addEventListener('keydown',e=>{
  if(e.key==='Enter'||e.key===' '){const s=e.target.closest('.sent');if(s){e.preventDefault();selectSent(+s.dataset.si);}}
});
sel.addEventListener('change',()=>{cur=+sel.value;render();});
document.getElementById('prev').addEventListener('click',()=>{cur=(cur-1+DATA.length)%DATA.length;render();});
document.getElementById('next').addEventListener('click',()=>{cur=(cur+1)%DATA.length;render();});
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="평가셋 JSONL")
    ap.add_argument("--evidence", default="", help="유물명 보강용 evidence JSON (선택)")
    ap.add_argument("--num", type=int, default=15, help="표본 유물 수")
    ap.add_argument("--ids", default="", help="쉼표구분 유물ID 지정 (지정 시 --num 무시)")
    ap.add_argument("--threshold", type=float, default=CITATION_THRESHOLD)
    ap.add_argument("--output", required=True, help="출력 HTML 경로")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", choices=["explanation", "qa"], default="explanation",
                    help="수집 과업 (요청 표기 방식 결정)")
    ap.add_argument("--dataset-label", default="",
                    help="전체 평가 결과 패널 제목에 붙일 데이터셋 설명 (예: '테스트셋 20% (2,594종, 시드 42)')")
    ap.add_argument("--nli-judgments", default="",
                    help="measure_nli_judge.py의 judgments_pos.jsonl — 지정 시 NLI 판정 모드")
    ap.add_argument("--nli-judge-name", default="gpt-4.1", help="판정자 표기명")
    ap.add_argument("--limit", type=int, default=0,
                    help="입력 앞 N행만 사용 (measure_nli_judge --limit과 일치시킬 것)")
    ap.add_argument("--page-title", default="해설 근거 하이라이트", help="페이지 제목")
    ap.add_argument("--no-hist", action="store_true", help="전체 패널의 일치도 분포 차트 숨김")
    ap.add_argument("--official-note", default="",
                    help="footer에 붙일 공식 지표 설명 한 문장 (예: '공식 지표는 전수 12,968종 … 74.49%%.')")
    ap.add_argument("--no-standalone", action="store_true",
                    help="doctype/charset/viewport 머리글 생략 (아티팩트 발행용)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    nli_by_row, nli_summary = {}, None
    if args.nli_judgments:
        counts = {l: 0 for l in NLI_LABELS}
        for l in open(args.nli_judgments, encoding="utf-8"):
            d = json.loads(l)
            nli_by_row.setdefault(d["ri"], {})[d["si"]] = d["label"]
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        total = sum(counts.values())
        supp = counts["완전함의"] + counts["부분함의"]
        nli_summary = {
            "counts": counts, "total": total,
            "acc": round(supp / total * 100, 2) if total else 0.0,
            "accFull": round(counts["완전함의"] / total * 100, 2) if total else 0.0,
            "judge": args.nli_judge_name,
        }

    relic_names = {}
    if args.evidence:
        ev = json.load(open(args.evidence, encoding="utf-8"))
        for e in (ev if isinstance(ev, list) else ev.values()):
            if isinstance(e, dict) and e.get("id"):
                relic_names[str(e["id"])] = str(e.get("유물", "") or "")

    idx_rows = list(enumerate(rows))
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        picked = [ir for ir in idx_rows if str(ir[1].get("유물ID")) in want]
    elif args.num >= len(rows):
        picked = idx_rows
    else:
        picked = stratified_sample(idx_rows, args.num, args.threshold, args.seed)

    items = [build_item(r, relic_names, args.threshold,
                        nli_map=nli_by_row.get(ri) if nli_by_row else None)
             for ri, r in picked]
    if not items:
        raise SystemExit("표본이 비어 있습니다 — --ids/--num/입력 파일을 확인하세요.")
    items.sort(key=lambda x: x["id"])  # 중립 정렬 (성능순 정렬은 선별 의심을 부름)
    summary, hist = build_summary(rows, args.threshold)
    data = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    cfg = json.dumps({"task": args.task, "label": args.dataset_label,
                      "summary": summary, "hist": (None if args.no_hist else hist), "nli": nli_summary},
                     ensure_ascii=False).replace("<", "\\u003c")
    official = (" " + html.escape(args.official_note)) if args.official_note else ""
    out = (TEMPLATE.replace("__DATA__", data)
           .replace("__CFG__", cfg)
           .replace("__THRESHOLD__", f"{args.threshold:.2f}")
           .replace("__OFFICIAL__", official)
           .replace("<title>해설 근거 하이라이트</title>",
                    "<title>" + html.escape(args.page_title) + "</title>"))
    if not args.no_standalone:
        out = STANDALONE_HEAD + out
    with open(args.output, "w", encoding="utf-8") as wf:
        wf.write(out)
    print(f"{args.output}: 유물 {len(items)}종, "
          f"{sum(i['nJudged'] for i in items)}개 판정 문장, {len(out)/1024:.0f}KB")


if __name__ == "__main__":
    main()
