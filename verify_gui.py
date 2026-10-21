# -*- coding: utf-8 -*-
"""
T-003 문화유산 해설 Citation Accuracy 검증 GUI (T-001 검증 GUI와 동일한 사용 흐름).

- TL-1(서버)에서 실행, TL-2 브라우저로 접속 (기본 포트 7860)
- [전체 평가 시작] 버튼 → ① 해설 생성(공식 수집 스크립트, --resume)
  ② 공식 측정(measure_citation_accuracy.py) ③ 결과 표·차트·하이라이트 뷰어 표시
- 측정은 공식 스크립트를 subprocess로 그대로 호출 → GUI 수치 = 공식 수치

사용:
  python3 verify_gui.py \
      --questions testset05/questions_05.jsonl \
      --evidence /home/heri/test_prepare/serving_local/data/aimuse_items_enriched_final_v6.json \
      --serve-url http://127.0.0.1:5000 \
      --eval-out output/gui_run/eval_set.jsonl \
      --port 7860
  # 사전 수집본으로 즉시 측정만 시연하려면: --eval-out <기존 eval_set.jsonl>
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from citation_metric import judge_answer  # 유형별 집계·히스토그램용 (판정 로직 동일)

app = Flask(__name__)
ARGS = None
STATE = {"phase": "idle", "msg": "대기 중", "n_done": 0, "n_total": 0,
         "rate": 0.0, "metrics": None, "per_visitor": None,
         "started": 0.0, "error": ""}
LOCK = threading.Lock()


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run_pipeline(regenerate):
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "output", "gui_run")
    os.makedirs(out_dir, exist_ok=True)
    eval_path = ARGS.eval_out
    n_total = count_lines(ARGS.questions)
    with LOCK:
        STATE.update(phase="collect", msg="해설 생성 중", n_total=n_total,
                     started=time.time(), error="", metrics=None)
    try:
        if regenerate or count_lines(eval_path) < n_total:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(base, "generate_responses_api.py"),
                 "--server", ARGS.serve_url, "--parallel", str(ARGS.parallel),
                 "--questions", ARGS.questions, "--evidence", ARGS.evidence,
                 "--output", eval_path, "--task", "explanation", "--resume"],
                cwd=base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                done = count_lines(eval_path)
                el = max(1e-9, time.time() - STATE["started"])
                with LOCK:
                    STATE.update(n_done=done, rate=done / el)
                time.sleep(2)
            if proc.returncode != 0:
                raise RuntimeError(f"수집 스크립트 종료 코드 {proc.returncode}")
        with LOCK:
            STATE.update(phase="measure", msg="Citation Accuracy 측정 중",
                         n_done=count_lines(eval_path))

        r = subprocess.run(
            [sys.executable, os.path.join(base, "measure_citation_accuracy.py"),
             "--input", eval_path, "--out-dir", out_dir],
            cwd=base, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("측정 실패: " + r.stderr[-300:])
        metrics = {}
        for line in open(os.path.join(out_dir, "metrics.txt"), encoding="utf-8"):
            if ":" in line:
                k, v = line.split(":", 1)
                metrics[k.strip()] = v.strip()

        # 유형별 정확도 + 일치도 히스토그램 (판정 로직은 공식과 동일 함수)
        with LOCK:
            STATE.update(phase="analyze", msg="유형별 집계 중")
        per = {}
        for l in open(eval_path, encoding="utf-8"):
            row = json.loads(l)
            j = judge_answer(row.get("답변") or "", row["근거문서"])
            vt = row.get("관람객유형", "기타")
            a = per.setdefault(vt, [0, 0])
            a[0] += j["n_citable"]; a[1] += j["n_total"]
        per_visitor = {k: round(c / t * 100, 2) for k, (c, t) in sorted(per.items()) if t}

        # 하이라이트 뷰어 생성
        subprocess.run(
            [sys.executable, os.path.join(base, "visualize_citation.py"),
             "--input", eval_path, "--evidence", ARGS.evidence,
             "--num", "100000", "--output", os.path.join(out_dir, "viewer.html"),
             "--dataset-label", ARGS.label, "--no-hist",
             "--official-note", "본 화면은 검증 GUI가 공식 측정 스크립트를 호출해 생성한 결과입니다."],
            cwd=base, capture_output=True, text=True)

        with LOCK:
            STATE.update(phase="done", msg="평가 완료", metrics=metrics,
                         per_visitor=per_visitor)
    except Exception as e:
        with LOCK:
            STATE.update(phase="error", msg="오류", error=str(e))


@app.route("/start", methods=["POST"])
def start():
    if STATE["phase"] in ("collect", "measure", "analyze"):
        return jsonify(ok=False, msg="이미 실행 중")
    regenerate = request.json.get("regenerate", False) if request.is_json else False
    threading.Thread(target=run_pipeline, args=(regenerate,), daemon=True).start()
    return jsonify(ok=True)


@app.route("/status")
def status():
    with LOCK:
        return jsonify(dict(STATE))


@app.route("/viewer")
def viewer():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "output", "gui_run", "viewer.html")
    if not os.path.exists(p):
        return "뷰어가 아직 생성되지 않았습니다. [전체 평가 시작] 후 이용하세요.", 404
    head = ('<!doctype html><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">')
    return head + open(p, encoding="utf-8").read()


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>문화유산 해설 Citation Accuracy 검증 GUI</title>
<style>
:root{--bg:#f7f6f3;--panel:#fff;--ink:#272e2c;--muted:#68726e;--line:#dcded9;
--accent:#2e7d6b;--ok:#1e8a68;--low:#b97324;--okbg:#e7f2ee;
--sans:'Noto Sans KR',-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;
padding:18px max(16px,2vw);}
.wrap{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:14px;}
h1{font-size:1.3rem;margin:0;}
.sub{color:var(--muted);font-size:.84rem;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;}
button{background:#e8731a;color:#fff;border:0;border-radius:8px;font:inherit;font-weight:700;
padding:12px 28px;cursor:pointer;font-size:1rem;flex:1 1 260px;}
button:disabled{opacity:.55;cursor:default;}
label.chk{font-size:.85rem;color:var(--muted);}
.prog{height:14px;background:var(--line);border-radius:999px;overflow:hidden;display:none;}
.prog b{display:block;height:100%;background:var(--accent);width:0%;transition:width .5s;}
.pmsg{font-size:.85rem;color:var(--muted);}
table{border-collapse:collapse;width:100%;font-size:.88rem;}
td,th{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left;}
td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:500;}
tr.hl td{color:var(--accent);font-weight:700;border-top:2px dashed var(--low);border-bottom:2px dashed var(--low);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media(max-width:820px){.grid{grid-template-columns:1fr;}}
h2{font-size:.86rem;letter-spacing:.06em;color:var(--muted);margin:0 0 8px;}
svg{width:100%;height:auto;display:block;}
a.viewbtn{display:inline-block;background:var(--okbg);color:var(--accent);font-weight:700;
padding:9px 18px;border-radius:8px;text-decoration:none;}
#results{display:none;}
.err{color:#b4322a;font-weight:600;}
</style>
<div class="wrap">
<div>
  <h1>문화유산 해설 Citation Accuracy 검증 GUI</h1>
  <div class="sub">현재 구성: __LABEL__ · 임계값 0.40 (사전 설정) · 측정은 공식 스크립트(measure_citation_accuracy.py) 호출<br>
  같은 네트워크의 다른 PC 접속 주소: <b>http://__HOST__:__PORT__</b></div>
</div>
<div class="card">
  <div class="row">
    <button id="startbtn" onclick="start()">전체 평가 시작</button>
    <label class="chk"><input type="checkbox" id="regen"> 해설 재생성 포함(미체크 시 기수집분 이어서/즉시 측정)</label>
  </div>
  <div style="margin-top:10px" class="prog" id="prog"><b id="progbar"></b></div>
  <div class="pmsg" id="pmsg"></div>
</div>
<div id="results">
<div class="card">
  <h2>평가 지표</h2>
  <table id="mtable"></table>
</div>
<div class="card"><h2>관람객 유형별 Citation Accuracy</h2><div id="barchart" style="max-width:640px"></div></div>
<div class="card">
  <h2>문장별 근거 하이라이트 — 유물별 확인 (648종 전체)</h2>
  <iframe id="hlframe" title="해설-근거 하이라이트" style="width:100%;height:800px;border:1px solid var(--line);border-radius:8px;background:#fff"></iframe>
  <div class="sub" style="margin-top:6px">상단 드롭다운·◀▶로 유물 선택 · 해설 문장 클릭 시 근거 위치 강조 · <a href="/viewer" target="_blank">새 창으로 열기</a></div>
</div>
</div>
</div>
<script>
let timer=null;
function start(){
  fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({regenerate:document.getElementById('regen').checked})});
  document.getElementById('startbtn').disabled=true;
  document.getElementById('prog').style.display='block';
  timer=setInterval(poll,1500); poll();
}
function poll(){
  fetch('/status').then(r=>r.json()).then(s=>{
    const pm=document.getElementById('pmsg');
    if(s.phase==='collect'){
      const pct=s.n_total? (100*s.n_done/s.n_total):0;
      document.getElementById('progbar').style.width=pct.toFixed(1)+'%';
      const eta=s.rate>0? Math.round((s.n_total-s.n_done)/s.rate/60):'-';
      pm.textContent=`해설 생성 중 ${s.n_done}/${s.n_total}건 (${s.rate.toFixed(2)}건/s, 남은 예상 ${eta}분)`;
    }else if(s.phase==='measure'||s.phase==='analyze'){
      document.getElementById('progbar').style.width='100%';
      pm.textContent=s.msg+'…';
    }else if(s.phase==='done'){ clearInterval(timer); render(s); }
    else if(s.phase==='error'){ clearInterval(timer);
      pm.innerHTML='<span class="err">오류: '+s.error+'</span>';
      document.getElementById('startbtn').disabled=false; }
  });
}
function render(s){
  document.getElementById('pmsg').textContent='평가 완료';
  document.getElementById('startbtn').disabled=false;
  document.getElementById('results').style.display='block';
  const fr=document.getElementById('hlframe'); if(fr&&!fr.src) fr.src='/viewer';
  const m=s.metrics;
  const rows=[
    ['평가한 유물 수', m['instances_evaluated']+' 종'],
    ['판정 대상 문장 수', (+m['sentences_total']).toLocaleString()+' 개'],
    ['인용 가능 문장 수', (+m['sentences_citable']).toLocaleString()+' 개'],
    ['판정 제외 (권유·상호작용 / 유보 / 단문)', m['sentences_excluded_interactive']+' / '+m['sentences_excluded_abstention']+' / '+m['sentences_excluded_short']],
    ['__HL__Citation Accuracy', m['citation_accuracy']],
    ['성능목표 대비', m['result']==='PASS' ? '70% 이상 — 통과' : '미달'],
    ['일치도 임계치', m['threshold']],
  ];
  document.getElementById('mtable').innerHTML=rows.map(r=>{
    const hl=r[0].startsWith('__HL__');
    return `<tr${hl?' class="hl"':''}><td>${r[0].replace('__HL__','')}</td><td>${r[1]}</td></tr>`;
  }).join('');
  // 유형별 막대
  const pv=s.per_visitor||{}; const keys=Object.keys(pv);
  const W=480,H=250,L=36,B=40,T=16,pw=W-L-10,ph=H-T-B;
  let bars='',gl='';
  for(let g=0;g<=4;g++){const y=T+ph-ph*g/4;
    gl+=`<line x1="${L}" y1="${y}" x2="${W-10}" y2="${y}" stroke="var(--line)"/>`+
        `<text x="${L-5}" y="${y+4}" text-anchor="end" font-size="10" fill="var(--muted)">${25*g}</text>`;}
  keys.forEach((k,i)=>{
    const bw=pw/keys.length, x=L+i*bw+bw*0.15, w=bw*0.7, h=ph*pv[k]/100, y=T+ph-h;
    bars+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="var(--ok)"/>`+
      `<text x="${x+w/2}" y="${y-5}" text-anchor="middle" font-size="11" font-weight="700" fill="var(--ink)">${pv[k].toFixed(1)}%</text>`+
      `<text x="${x+w/2}" y="${H-18}" text-anchor="middle" font-size="11" fill="var(--muted)">${k}</text>`;});
  const th=T+ph-ph*0.70;
  document.getElementById('barchart').innerHTML=
    `<svg viewBox="0 0 ${W} ${H}">${gl}${bars}<line x1="${L}" y1="${th}" x2="${W-10}" y2="${th}" stroke="var(--low)" stroke-dasharray="5 3" stroke-width="1.5"/><text x="${W-12}" y="${th-4}" text-anchor="end" font-size="10" fill="var(--low)">목표 70%</text></svg>`;
}
</script>"""


@app.route("/")
def index():
    host = request.host.split(":")[0]
    return (PAGE.replace("__LABEL__", ARGS.label)
                .replace("__HOST__", host)
                .replace("__PORT__", str(ARGS.port)))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--serve-url", default="http://127.0.0.1:5000")
    ap.add_argument("--eval-out", default="output/gui_run/eval_set.jsonl")
    ap.add_argument("--label", default="테스트셋 5% (648종, 시드 42)")
    ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ARGS = ap.parse_args()
    print(f"* 검증 GUI: http://127.0.0.1:{ARGS.port}  (같은 네트워크: http://<TL-1 IP>:{ARGS.port})")
    app.run(host=ARGS.host, port=ARGS.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
