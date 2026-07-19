import os
import sys
import json
import time
import tempfile
import subprocess
import threading
import webbrowser
import uuid

from pathlib import Path

import pandas as pd
from flask import (
    Flask, request, jsonify, send_file,
    render_template_string, session
)

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder="assets", static_url_path="/utils/assets")
app.secret_key = os.environ.get("SECRET_KEY")
print(app.secret_key)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

# ─── App state ─────────────────────────────────────────
# ─── App state (per-session) ──────────────────────────
tasks = {}   # task_id -> {"work_dir":..., "pdf_path":..., "csv_path":..., "running":...}

STEPS = {
    "tr": [
        ("Demo veri oluşturuluyor.",                           "\U0001F4C2"),
        ("Veriler toplulaştırılıyor.",                        "\U0001F4CA"),
        ("Uygun bağışçıların terk riski tahmin ediliyor.",    "\U0001F52E"),
        ("RFM analizi yapılıyor.",                             "\U0001F4C8"),
        ("Rapor hazırlanıyor.",                                "\U0001F4C4"),
        ("Final veri seti oluşturuluyor.",                    "\U0001F4BE"),
    ],
    "en": [
        ("Generating demo data.",                              "\U0001F4C2"),
        ("Consolidating data.",                                "\U0001F4CA"),
        ("Predicting churn risk for eligible donors.",         "\U0001F52E"),
        ("Running RFM analysis.",                              "\U0001F4C8"),
        ("Preparing the report.",                               "\U0001F4C4"),
        ("Building the final dataset.",                        "\U0001F4BE"),
    ],
}

def _steps(lang):
    return STEPS.get(lang, STEPS["tr"])

# ─── HTML templates ───────────────────────────────────
def _t(key, lang):
    """Simple inline translation helper"""
    tr = {
        "title": "Bağışçı Segmentasyon Analizi",
        "subtitle": "RFM analizi \u00B7 Terk riski tahmini \u00B7 Detaylı raporlama",
        "nav_home": "\U0001F3E0 Ana Sayfa",
        "nav_demo": "\U0001F916 Demo Veri",
        "section_data": "Bu Demo Hakkında",
        "top_intro": "Aşağıdaki <b>Analizi Başlat</b> butonuna basıldığında; belirlenen ayarlarla sentetik bir bağışçı veri seti üretilir, bu veri üzerinde RFM segmentasyonu ve kayıp (churn) riski tahmini yapılır ve indirilebilir bir PDF rapor ile CSV veri seti sunulur. Herhangi bir dosya yüklenmesine gerek yoktur — tüm süreç otomatik işlemektedir.",
        "about_title": "Bu Proje Hakkında",
        "about_text": "Bu uygulama; bir kurum için geliştirilen bağışçı segmentasyonu ve kayıp riski analizi projesinin sadeleştirilmiş ve herkese açık hale getirilmiş bir versiyonudur. Orijinal projede bu analize ek olarak, kuruma özel ve aksiyona dönük bir strateji dokümanı da hazırlanmıştır; bu genel erişime açık sürümde yalnızca analiz ve raporlama akışı yer almaktadır.",
        "churn_title": "Kayıp (Churn) Riski Nasıl Hesaplanıyor?",
        "churn_html": "<ul class=\"info-list\"><li><b>Zamanı Geri Sarma (Cutoff Mantığı):</b> Model, geçmişte belirli bir tarihi (örneğin ayın son günü) bir \u201ckarar anı\u201d olarak seçer; kendini o tarihteymiş gibi konumlandırıp bağışçının o ana kadarki geçmişine bakar.</li><li><b>Uygun Bağışçıyı Belirleme:</b> Herkes değil, yalnızca belirli kriterleri karşılayan (örn. son 180\u2013365 gün içinde en az iki kez bağış yapmış) aktif bağışçılar analize dahil edilir. Amaç, elimizdeki aktif kitleyi korumaya odaklanmaktır.</li><li><b>Geleceği Gözlemleyerek Etiketleme:</b> Karar anından itibaren ileriye dönük bir pencere açılır (örn. sonraki 180\u2013365 gün). Bağışçı bu süre içinde tekrar bağış yaparsa \u201cbağlı\u201d, yapmazsa \u201ckayıp\u201d olarak etiketlenir.</li><li><b>Aylık Kontrol Noktaları:</b> Her ay sonu ayrı bir kontrol tarihi olarak ele alınır; aynı bağışçı farklı aylar için veri setinde birden çok kez yer alabilir. Böylece modelin öğrenebileceği örnek sayısı artar.</li></ul>",
        "rfm_title": "RFM Analizi Nedir?",
        "rfm_text": "RFM, bağışçıları üç boyutta değerlendiren klasik bir segmentasyon yaklaşımıdır: <b>Recency (Yenilik)</b> \u2014 en son ne zaman bağış yaptı, <b>Frequency (Sıklık)</b> \u2014 ne sıklıkla bağış yapıyor, <b>Monetary (Parasal Değer)</b> \u2014 toplam bağış tutarı ne kadar. Önemli bir detay: RFM analizi seçilen zaman penceresine duyarlıdır; aynı bağışçı, farklı bir tarih aralığı seçildiğinde farklı bir segmentte yer alabilir. Orijinal projede bağışçıların segment geçişlerine yönelik olarak düşüş, yükseliş vb. durumlar için ayrı bir strateji hazırlanmıştır.",
        "parameters": "Analiz Parametreleri",
        "cutoff": "Cutoff Tarihi",
        "cutoff_desc": "Terk riski için referans bitiş tarihi.",
        "low_risk": "Düşük Risk Eşiği (%)",
        "low_risk_desc": "Bu değerin altı düşük risk.",
        "high_risk": "Yüksek Risk Eşiği (%)",
        "high_risk_desc": "Bu değerin üstü yüksek risk.",
        "pdf_name": "PDF Çıktı Dosya Adı",
        "kurum_adi": "Kurum Adı",
        "kurum_adi_desc": "Raporda görüntülenecek kurum/kuruluş adı.",
        "info_ready": "Analiz, yukarıda belirtilen parametreler ile üretilen veriler kullanılarak çalışacaktır.",
        "start_btn": "Analizi Başlat",
        "status_title": "Analiz Durumu",
        "outputs": "Çıktılar",
        "dl_pdf": "PDF Raporu İndir",
        "dl_csv": "CSV Verisini İndir",
        "new_analysis": "Yeni Analiz",
        "demo_title": "Demo Veri Parametreleri",
        "demo_desc": "Belirlediğiniz parametrelerle sentetik bağış ve bağışçı verisi oluşturun.",
        "demo_seed": "Random Seed",
        "demo_seed_desc": "Tekrarlanabilir veri için sabit bir sayı girin.",
        "demo_seed_enable": "Aynı Veriyi Tekrar Kullan",
        "demo_seed_enable_desc": "İşaretlemezsen her seferinde tamamen farklı, rastgele bir veri seti oluşturulur. İşaretlersen, aşağıya istediğin bir numarayı yazarak aynı veriyi istediğin zaman tekrar oluşturabilirsin.",
        "demo_seed_placeholder": "Bir numara yaz, örn: 42",
        "demo_donors": "Bağışçı Sayısı",
        "demo_donations": "Bağış Sayısı",
        "loading": "Başlatılıyor\u2026",
        "running": "Çalışıyor\u2026",
        "completed": "Tamamlandı",
        "error": "Hata",
    }
    en = {
        "title": "Donor Segmentation Analysis",
        "subtitle": "RFM analysis \u00B7 Churn prediction \u00B7 Detailed reporting",
        "nav_home": "\U0001F3E0 Home",
        "nav_demo": "\U0001F916 Demo Data",
        "section_data": "About This Demo",
        "top_intro": "When the <b>Start Analysis</b> button below is clicked, a synthetic donor dataset is generated using the configured settings, RFM segmentation and churn-risk prediction are run on it, and a downloadable PDF report along with a CSV dataset are provided. No file upload is required \u2014 the entire process runs automatically.",
        "about_title": "About This Project",
        "about_text": "This app is a simplified, publicly shareable version of a donor segmentation and churn-risk analysis project originally developed for an organization. In the original project, an institution-specific, action-oriented strategy document was also produced alongside this analysis; that strategic component is not included in this public version, which presents only the analysis and reporting pipeline.",
        "churn_title": "How Is Churn Risk Calculated?",
        "churn_html": "<ul class=\"info-list\"><li><b>Rewinding Time (Cutoff Logic):</b> The model picks a specific point in the past (e.g., the last day of a month) as a \u201cdecision moment,\u201d positions itself as if it were that date, and looks back at the donor's history up to that point.</li><li><b>Defining Eligible Donors:</b> Not everyone is tracked \u2014 only donors meeting certain criteria (e.g., at least two donations in the last 180\u2013365 days) are included. The goal is to focus on protecting the active donor base we already have.</li><li><b>Labeling by Looking Forward:</b> A forward-looking window opens from the decision moment (e.g., the next 180\u2013365 days). If the donor gives again within that window, they're labeled \u201cretained\u201d; if not, they're labeled \u201cchurned.\u201d</li><li><b>Monthly Checkpoints:</b> Each month-end is treated as its own checkpoint, so the same donor can appear multiple times in the training data across different months \u2014 giving the model more examples to learn from.</li></ul>",
        "rfm_title": "What Is RFM Analysis?",
        "rfm_text": "RFM is a classic segmentation approach that scores donors on three dimensions: <b>Recency</b> \u2014 how recently did they give, <b>Frequency</b> \u2014 how often do they give, <b>Monetary</b> \u2014 how much have they given in total. One important detail: RFM is sensitive to the selected time window; the same donor can land in a different segment depending on the date range chosen. In the original project, a separate strategy was developed to address donors' segment transitions \u2014 such as declines or upgrades \u2014 over time.",
        "parameters": "Analysis Parameters",
        "cutoff": "Cutoff Date",
        "cutoff_desc": "Reference end date for churn risk analysis.",
        "low_risk": "Low Risk Threshold (%)",
        "low_risk_desc": "Values below this are low risk.",
        "high_risk": "High Risk Threshold (%)",
        "high_risk_desc": "Values above this are high risk.",
        "pdf_name": "PDF Output Filename",
        "kurum_adi": "Institution Name",
        "kurum_adi_desc": "The institution/organization name shown in the report.",
        "info_ready": "The analysis will run on data generated from the parameters specified above.",
        "start_btn": "Start Analysis",
        "status_title": "Analysis Status",
        "outputs": "Outputs",
        "dl_pdf": "Download PDF Report",
        "dl_csv": "Download CSV Data",
        "new_analysis": "New Analysis",
        "demo_title": "Demo Data Parameters",
        "demo_desc": "Create synthetic donation and donor data with your chosen parameters.",
        "demo_seed": "Random Seed",
        "demo_seed_desc": "Enter a fixed number for reproducible data.",
        "demo_seed_enable": "Reuse the Same Data",
        "demo_seed_enable_desc": "Leave this unchecked to get a brand new, random dataset every time. Check it and type any number below to recreate that exact same dataset whenever you want.",
        "demo_seed_placeholder": "Type a number, e.g. 42",
        "demo_donors": "Donor Count",
        "demo_donations": "Donation Count",
        "loading": "Starting\u2026",
        "running": "Running\u2026",
        "completed": "Completed",
        "error": "Error",
    }
    lang_map = {"en": en, "tr": tr}
    return lang_map.get(lang, tr).get(key, key)


def build_home_html(lang="tr"):
    t = lambda k: _t(k, lang)
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t("title")}</title>
<link rel="icon" type="image/png" href="utils/assets/logo.png">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {{ --navy:#1a2e5a; --navy-dark:#0f1e3d; --accent2:#2d4fa0;
  --bg:#fff; --surface:#f7f8fa; --border:#e0e4ea;
  --text:#111827; --muted:#6b7280; --label:#374151;
  --success:#16a34a; --danger:#dc2626; }}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text)}}
.hero{{background:linear-gradient(135deg,var(--navy-dark),var(--navy) 60%,#2d4fa0);padding:2.5rem 3rem;text-align:center}}
.hero h1{{font-family:'Source Serif 4',serif;font-size:2.1rem;color:#fff;line-height:1.2;margin-bottom:.4rem}}
.hero p{{color:rgba(255,255,255,.7);font-size:.95rem;font-weight:300}}
.topnav{{display:flex;gap:.25rem;justify-content:center;margin-top:1.4rem;flex-wrap:wrap}}
.topnav a{{padding:.4rem 1.1rem;border-radius:6px;font-size:.88rem;font-weight:500;
  color:rgba(255,255,255,.7);text-decoration:none;border:1px solid rgba(255,255,255,.2);transition:all .15s}}
.topnav a.active,.topnav a:hover{{background:rgba(255,255,255,.2);color:#fff;border-color:rgba(255,255,255,.5)}}
.lang-switch{{margin-left:auto;display:flex;gap:.25rem}}
.lang-switch a{{font-size:.8rem;padding:.3rem .7rem}}
.container{{max-width:820px;margin:0 auto;padding:0 2.5rem 4rem}}
.section-title{{font-size:.75rem;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--navy);margin:2rem 0 .8rem;
  padding-bottom:.35rem;border-bottom:2px solid var(--navy);display:block}}
.caption{{font-size:.85rem;color:var(--muted);margin-bottom:1rem;line-height:1.5}}
.info-heading{{font-size:1rem;font-weight:600;color:var(--navy);margin:1.6rem 0 .4rem}}
.info-list{{margin:.4rem 0 1rem 1.3rem;padding:0}}
.info-list li{{font-size:.85rem;color:var(--muted);margin-bottom:.6rem;line-height:1.55}}
.info-list li b{{color:var(--label)}}
.upload-row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem}}
.upload-box{{background:var(--surface);border:1.5px dashed var(--border);
  border-radius:8px;padding:1rem;transition:border-color .15s;cursor:pointer;
  text-align:center;position:relative}}
.upload-box:hover{{border-color:var(--accent2)}}
.upload-box input[type=file]{{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}}
.upload-box .label{{font-size:.9rem;font-weight:500;color:var(--label);margin-bottom:.25rem}}
.upload-box .hint{{font-size:.8rem;color:var(--muted)}}
.upload-box .filename{{font-size:.82rem;color:var(--success);margin-top:.4rem;font-weight:500;word-break:break-all}}
details{{margin:1rem 0}}
summary{{cursor:pointer;font-size:.93rem;font-weight:500;color:var(--navy);padding:.5rem 0;user-select:none}}
.param-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.8rem 1.5rem;padding:.8rem 0 .4rem}}
.param-item label{{display:block;font-size:.86rem;font-weight:500;color:var(--label);margin-bottom:.3rem}}
.param-item .desc{{font-size:.8rem;color:var(--muted);margin-top:.25rem;line-height:1.4}}
input[type=text],input[type=number]{{width:100%;padding:.4rem .65rem;
  border:1px solid var(--border);border-radius:6px;font-family:inherit;
  font-size:.95rem;color:var(--text);background:#fff;outline:none;transition:border-color .15s}}
input:focus{{border-color:var(--accent2);box-shadow:0 0 0 3px rgba(45,79,160,.12)}}
input:disabled{{background:var(--surface);color:var(--muted)}}
.checkbox-row{{display:flex;align-items:center;gap:.5rem;margin:.4rem 0}}
.checkbox-row input{{width:auto}}
.btn{{display:block;width:100%;padding:.7rem 2rem;background:var(--navy);color:#fff;border:none;
  border-radius:8px;font-family:inherit;font-size:1rem;font-weight:600;cursor:pointer;
  transition:background .15s,transform .15s,box-shadow .15s;margin-top:1rem}}
.btn:hover:not(:disabled){{background:var(--navy-dark);transform:translateY(-1px);box-shadow:0 4px 16px rgba(26,46,90,.25)}}
.btn:disabled{{background:#d1d5db;color:#9ca3af;cursor:not-allowed;transform:none}}
.btn-outline{{background:#fff;color:var(--navy);border:2px solid var(--navy)}}
.btn-outline:hover:not(:disabled){{background:var(--surface)}}
.btn-success{{background:var(--success)}}
.btn-success:hover:not(:disabled){{background:#15803d}}
hr{{border:none;border-top:1px solid var(--border);margin:1.5rem 0}}
.progress-bar-wrap{{background:#e5e7eb;border-radius:99px;height:8px;overflow:hidden;margin:.5rem 0 1rem}}
.progress-bar{{height:100%;border-radius:99px;background:var(--navy);transition:width .4s ease;width:0%}}
.log-container{{background:#f9fafb;border:1px solid var(--border);border-radius:8px;
  padding:1rem 1.25rem;font-family:'Fira Code','Courier New',monospace;font-size:.9rem;
  max-height:300px;overflow-y:auto;line-height:2}}
.log-line{{color:#374151}}
.log-line.done{{color:var(--success);font-weight:500}}
.log-line.active{{color:var(--navy);font-weight:600;animation:pulse 1.2s infinite}}
.log-line.error{{color:var(--danger)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.badge{{display:inline-block;padding:.2rem .7rem;border-radius:4px;font-size:.75rem;
  font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin:.75rem 0}}
.badge-success{{background:#dcfce7;color:#15803d}}
.badge-running{{background:#dbeafe;color:#1e40af}}
.badge-error{{background:#fee2e2;color:#b91c1c}}
.dl-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.5rem;margin-top:1rem}}
.dl-title{{font-size:.75rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--navy);margin-bottom:1rem;padding-bottom:.4rem;border-bottom:2px solid var(--navy)}}
.dl-row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.btn-dl{{background:var(--navy);color:#fff;border:none;border-radius:8px;padding:.65rem 1.5rem;
  font-family:inherit;font-size:.95rem;font-weight:600;cursor:pointer;width:100%;
  text-decoration:none;display:block;text-align:center;transition:background .15s}}
.btn-dl:hover{{background:var(--navy-dark)}}
.info-box{{background:#eff6ff;color:#1e40af;border-radius:6px;padding:.75rem 1rem;font-size:.9rem;margin:.5rem 0 1rem}}
.error-box{{background:#fee2e2;color:#b91c1c;border-radius:6px;padding:.75rem 1rem;
  font-size:.88rem;margin:.5rem 0;white-space:pre-wrap;font-family:monospace;max-height:200px;overflow-y:auto}}
#status-section{{display:none}}
#download-section{{display:none}}
</style>
</head>
<body>

<div class="hero">
  <h1>{t("title")}</h1>
  <p>{t("subtitle")}</p>
  <div class="topnav">
    <div class="lang-switch">
      <a href="/lang/tr" style="font-weight:bold{'!important' if lang=='tr' else ''}">TR</a>
      <a href="/lang/en" style="font-weight:bold{'!important' if lang=='en' else ''}">EN</a>
    </div>
  </div>
</div>

<div class="container">

  <!-- BU DEMO HAKKINDA -->
  <div class="info-box">{t("top_intro")}</div>

  <span class="section-title">{t("section_data")}</span>
  <p class="caption" style="line-height:1.7">{t("about_text")}</p>

  <h3 class="info-heading">{t("churn_title")}</h3>
  <div class="caption" style="line-height:1.7">{t("churn_html")}</div>

  <h3 class="info-heading">{t("rfm_title")}</h3>
  <p class="caption" style="line-height:1.7">{t("rfm_text")}</p>

  <!-- DEMO VERİ -->
  <details>
    <summary>\U0001F916 {t("demo_title")}</summary>
    <p class="caption">{t("demo_desc")}</p>
    <div class="param-grid">
      <div class="param-item">
        <label for="demo-donors">{t("demo_donors")}</label>
        <input type="number" id="demo-donors" value="60" min="10" max="10000">
      </div>
      <div class="param-item" style="grid-column:1/-1">
        <label for="demo-donations">{t("demo_donations")}</label>
        <input type="number" id="demo-donations" value="1000" min="1000" max="100000">
      </div>
      <div class="param-item" style="grid-column:1/-1">
        <div class="checkbox-row">
          <input type="checkbox" id="demo-seed-enable" onchange="toggleSeed()">
          <label for="demo-seed-enable" style="margin:0">{t("demo_seed_enable")}</label>
        </div>
        <div class="desc">{t("demo_seed_enable_desc")}</div>
        <input type="number" id="demo-seed" placeholder="{t("demo_seed_placeholder")}" disabled style="margin-top:.5rem">
      </div>
    </div>
  </details>

  <!-- PARAMETRELER -->
  <details>
    <summary>\u2699\uFE0F {t("parameters")}</summary>
    <div class="param-grid">
      <div class="param-item">
        <label for="p-cutoff">{t("cutoff")}</label>
        <input type="text" id="p-cutoff" placeholder="Boş = None  |  örn: 2024-01-01">
        <div class="desc">{t("cutoff_desc")}</div>
      </div>
      <div class="param-item" style="grid-column:1/-1">
        <label for="p-output">{t("pdf_name")}</label>
        <input type="text" id="p-output" value="bagisci_segmentasyonu_raporu.pdf">
      </div>
    </div>
  </details>

  <div id="info-box" class="info-box">
    \u2139\uFE0F {t("info_ready")}
  </div>

  <button class="btn" id="start-btn" onclick="startAnalysis()">
    {t("start_btn")}
  </button>

  <!-- DURUM -->
  <div id="status-section">
    <hr>
    <span class="section-title">{t("status_title")}</span>
    <div class="progress-bar-wrap"><div class="progress-bar" id="prog-bar"></div></div>
    <div class="log-container" id="log-container"></div>
    <div id="badge-area"></div>
    <div id="error-area"></div>
  </div>

  <!-- İNDİRME -->
  <div id="download-section">
    <hr>
    <div class="dl-card">
      <div class="dl-title">{t("outputs")}</div>
      <div class="dl-row">
        <a class="btn-dl" id="dl-pdf" href="/download/pdf">\U0001F4C4 {t("dl_pdf")}</a>
        <a class="btn-dl" id="dl-csv" href="/download/csv">\U0001F4CA {t("dl_csv")}</a>
      </div>
    </div>
    <button class="btn btn-outline" onclick="resetApp()" style="margin-top:1rem">
      \U0001F504 {t("new_analysis")}
    </button>
  </div>

</div>

<script>
var STEPS = {json.dumps(_steps(lang))};
var pollTimer = null;

function toggleSeed() {{
  document.getElementById('demo-seed').disabled = !document.getElementById('demo-seed-enable').checked;
}}

async function startAnalysis() {{
  var btn = document.getElementById('start-btn');
  btn.disabled = true;
  btn.textContent = '{t("loading")}';

  var fd = new FormData();
  var seedEnabled = document.getElementById('demo-seed-enable').checked;
  fd.append('demo_seed_enabled', seedEnabled);
  if (seedEnabled) {{
    fd.append('demo_seed', document.getElementById('demo-seed').value);
  }}
  fd.append('demo_donors',    document.getElementById('demo-donors').value || 60);
  fd.append('demo_donations', document.getElementById('demo-donations').value || 1000);
  fd.append('cutoff',     document.getElementById('p-cutoff').value.trim());
  fd.append('output',     document.getElementById('p-output').value);

  var res = await fetch('/start', {{method:'POST', body:fd}});
  var data = await res.json();
  if (data.ok) {{
    document.getElementById('status-section').style.display = 'block';
    document.getElementById('download-section').style.display = 'none';
    startPolling();
  }} else {{
    alert('Error: ' + data.error);
    btn.disabled = false;
    btn.textContent = '{t("start_btn")}';
  }}
}}

function startPolling() {{
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(poll, 1000);
  poll();
}}

async function poll() {{
  try {{
    var res = await fetch('/status');
    var s = await res.json();
    renderStatus(s);
    if (s.finished) {{
      clearInterval(pollTimer);
      pollTimer = null;
    }}
  }} catch(e) {{}}
}}

function renderStatus(s) {{
  var n = STEPS.length;
  var cur = s.step || 0;
  var pct = s.finished && !s.error ? 100 : Math.min((cur / n) * 100, 100);
  document.getElementById('prog-bar').style.width = pct + '%';
  var html = '';
  for (var i = 0; i < STEPS.length; i++) {{
    var text = STEPS[i][0], icon = STEPS[i][1];
    var num = i + 1, cls = '', ind = '\u2B1C';
    if (s.finished && !s.error || num < cur) {{ cls = 'done'; ind = '\u2705'; }}
    else if (num === cur && !s.finished) {{ cls = 'active'; ind = '\u23F3'; }}
    html += '<div class="log-line ' + cls + '">' + ind + ' ' + icon + ' ' + text + '</div>';
  }}
  document.getElementById('log-container').innerHTML = html;
  var badge = document.getElementById('badge-area');
  var errArea = document.getElementById('error-area');
  if (s.error) {{
    badge.innerHTML = '<div class="badge badge-error">\u2717 {t("error")}</div>';
    errArea.innerHTML = '<div class="error-box">' + escHtml(s.error) + '</div>';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('start-btn').textContent = '{t("start_btn")}';
  }} else if (s.finished) {{
    badge.innerHTML = '<div class="badge badge-success">\u2713 {t("completed")}</div>';
    errArea.innerHTML = '';
    document.getElementById('download-section').style.display = 'block';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('start-btn').textContent = '{t("start_btn")}';
  }} else {{
    badge.innerHTML = '<div class="badge badge-running">\u23F3 {t("running")}</div>';
    errArea.innerHTML = '';
  }}
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function resetApp() {{
  fetch('/reset', {{method:'POST'}}).then(function() {{ location.reload(); }});
}}

window.addEventListener('DOMContentLoaded', async function() {{
  var res = await fetch('/status');
  var s = await res.json();
  if (s.running || s.finished) {{
    document.getElementById('status-section').style.display = 'block';
    renderStatus(s);
    if (s.running) startPolling();
    if (s.finished && !s.error)
      document.getElementById('download-section').style.display = 'block';
  }}
}});
</script>
</body>
</html>"""

# ─── Flask routes ─────────────────────────────────────

@app.route("/")
def index():
    lang = session.get("lang", "tr")
    return build_home_html(lang)


@app.route("/lang/<code>")
def set_lang(code):
    if code in ("tr", "en"):
        session["lang"] = code
    return index()


def _read_status(task):
    empty = {"step": 0, "total": len(STEPS["tr"]), "message": "", "finished": False,
              "error": None, "running": False}
    if not task:
        return empty
    wd = task.get("work_dir")
    if not wd:
        empty["running"] = task.get("running", False)
        return empty
    p = os.path.join(wd, "status.json")
    for _ in range(5):
        if os.path.exists(p):
            try:
                txt = open(p, "r", encoding="utf-8").read().strip()
                if txt:
                    d = json.loads(txt)
                    d["running"] = task["running"]
                    return d
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    empty["running"] = task.get("running", False)
    return empty


@app.route("/status")
def status():
    task_id = session.get("task_id")
    task = tasks.get(task_id) if task_id else None
    return jsonify(_read_status(task))


@app.route("/start", methods=["POST"])
def start():
    task_id = session.get("task_id")
    existing = tasks.get(task_id) if task_id else None
    if existing and existing.get("running"):
        return jsonify({"ok": False, "error": "Zaten çalışıyor."})

    seed_enabled = request.form.get("demo_seed_enabled", "false").lower() == "true"
    seed_raw = request.form.get("demo_seed", "").strip()
    if seed_enabled and seed_raw != "":
        demo_seed = int(seed_raw)
    else:
        # Sabit seed istenmediyse gerçekten rastgele bir sayı üret
        # (işletim sisteminin rastgele sayı kaynağından)
        demo_seed = int.from_bytes(os.urandom(4), "big")

    demo_donors    = request.form.get("demo_donors", 60, type=int)
    demo_donations = request.form.get("demo_donations", 1000, type=int)

    DEMO_DONORS_MIN, DEMO_DONORS_MAX = 10, 10000
    DEMO_DONATIONS_MIN, DEMO_DONATIONS_MAX = 1000, 100000

    if demo_donors is None or not (DEMO_DONORS_MIN <= demo_donors <= DEMO_DONORS_MAX):
        return jsonify({"ok": False,
                        "error": f"Bağışçı sayısı {DEMO_DONORS_MIN} ile {DEMO_DONORS_MAX} arasında olmalıdır."})
    if demo_donations is None or not (DEMO_DONATIONS_MIN <= demo_donations <= DEMO_DONATIONS_MAX):
        return jsonify({"ok": False,
                        "error": f"Bağış sayısı {DEMO_DONATIONS_MIN} ile {DEMO_DONATIONS_MAX} arasında olmalıdır."})

    cutoff      = request.form.get("cutoff", "").strip() or None
    output_name = request.form.get("output", "bagisci_segmentasyonu_raporu.pdf")

    work_dir = tempfile.mkdtemp(prefix="demo_analiz_")

    pdf_out    = os.path.join(work_dir, output_name)
    csv_out    = os.path.join(work_dir, "bagisci_segmentasyonu_verisi.csv")
    status_out = os.path.join(work_dir, "status.json")

    cwd = os.getcwd()

    lang = session.get("lang", "tr")

    script = f"""
import sys, os, json, traceback
sys.path.insert(0, r'{cwd}')
os.chdir(r'{cwd}')

STATUS_FILE = r'{status_out}'

def update_status(step, total, message, finished=False, error=None):
    payload = json.dumps({{"step":step,"total":total,"message":message,
                           "finished":finished,"error":error}}, ensure_ascii=False)
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, STATUS_FILE)

TOTAL = {len(STEPS["tr"])}

try:
    from utils.demo_data_generator import generate_demo_data
    from utils.consolidator import consolidate
    from utils.churn_risk_predictor import predict_churn_risk
    from utils.rfm_analyzer import analyze_rfm
    from utils.report_generator import generate_report

    N_BAGISCI   = {demo_donors}
    N_BAGIS     = {demo_donations}
    RANDOM_SEED = {demo_seed}
    CUTOFF_DATE = {repr(cutoff)}
    OUTPUT_PATH = r'{pdf_out}'
    LANGUAGE    = '{lang}'

    update_status(1, TOTAL, "Demo veri oluşturuluyor.")
    donation_data, donor_data = generate_demo_data(
        n_bagisci=N_BAGISCI,
        n_bagis=N_BAGIS,
        random_seed=RANDOM_SEED
    )

    update_status(2, TOTAL, "Veriler toplulaştırılıyor.")
    consolidated_data = consolidate(df=donation_data, df_bagiscilar=donor_data)

    update_status(3, TOTAL, "Uygun bağışçıların terk riski tahmin ediliyor.")
    churns = predict_churn_risk(
        duzenlenmis_data=donation_data,
        toplulasmis_data=consolidated_data,
        cutoff_date=CUTOFF_DATE
    )

    update_status(4, TOTAL, "RFM analizi yapılıyor.")
    rfms = analyze_rfm(donation_data)

    name_surname = (
        donation_data[["Bagisci No", "Ad Soyad"]]
        .drop_duplicates(subset="Bagisci No")
    )
    final_data = (
        consolidated_data
        .merge(name_surname, on="Bagisci No", how="left")
        .merge(churns, on="Bagisci No", how="left")
        .merge(rfms, on="Bagisci No", how="left")
    )
    if "Terk Riski" in final_data.columns:
        final_data["Terk Riski"] = final_data["Terk Riski"].fillna("Terk Riski Yok")

    update_status(5, TOTAL, "Rapor hazırlanıyor.")
    generate_report(
        final_data=final_data,
        duzenlenmis_data=donation_data,
        output_path=OUTPUT_PATH,
        language=LANGUAGE
    )

    update_status(6, TOTAL, "Final veri seti oluşturuluyor.")
    final_data.to_csv(
        r'{csv_out}', index=False, sep=";",
        encoding="utf-8-sig", decimal=",", lineterminator="\\n")

    update_status(6, TOTAL, "Tamamlandı.", finished=True)

except Exception as e:
    update_status(0, TOTAL, str(e), finished=True, error=traceback.format_exc())
"""

    script_path = os.path.join(work_dir, "run_analysis.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    with open(status_out, "w", encoding="utf-8") as f:
        json.dump({"step": 0, "total": len(STEPS["tr"]), "message": _t("loading", lang),
                   "finished": False, "error": None}, f)

    task_id = uuid.uuid4().hex
    session["task_id"] = task_id
    tasks[task_id] = {
        "work_dir": work_dir,
        "pdf_path": pdf_out,
        "csv_path": csv_out,
        "running": True,
    }

    def run():
        subprocess.run([sys.executable, script_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tasks[task_id]["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/download/pdf")
def download_pdf():
    task = tasks.get(session.get("task_id"))
    p = task.get("pdf_path") if task else None
    if p and os.path.exists(p):
        return send_file(p, as_attachment=True, download_name=os.path.basename(p))
    return "PDF bulunamadı.", 404


@app.route("/download/csv")
def download_csv():
    task = tasks.get(session.get("task_id"))
    p = task.get("csv_path") if task else None
    if p and os.path.exists(p):
        return send_file(p, as_attachment=True, download_name=os.path.basename(p))
    return "CSV bulunamadı.", 404


@app.route("/reset", methods=["POST"])
def reset():
    task_id = session.pop("task_id", None)
    if task_id:
        tasks.pop(task_id, None)
    return jsonify({"ok": True})


# ─── Start ────────────────────────────────────────────
if __name__ == "__main__":
    port = 8080
    threading.Timer(1.5, lambda: webbrowser.open(f"http://0.0.0.0:{port}")).start()
    print(f"\nUygulama başlatıldı -> http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
