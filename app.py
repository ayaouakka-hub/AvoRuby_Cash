"""
app.py — AvoRuby Cash · Streamlit Application
===============================================
Run : streamlit run app.py
"""

import sys
import os
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import logging

from AvoRuby_Back.src.models import traiter_dossier_agriculteur
from AvoRuby_Back.src.assistat.rag_engine import init_rag, get_assistant_advice

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

st.set_page_config(
    page_title="AvoRuby Cash — Scoring Credit Agricole",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# CSS — injecté une seule fois, pas d'indentation problématique
# ══════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--cforest:#0B1F10;--cgdk:#145A2F;--cgmd:#1E7A44;--cglt:#2A9D5C;--cred:#B91C1C;--cream:#F7F2E9;--csand:#EDE6D6;--cchar:#1A1A1A;--cslate:#374151;--cmuted:#6B7280;--cborder:#D4CAB8;--r:10px;}
.stApp{background-color:var(--cream) !important;font-family:'DM Sans',sans-serif;color:var(--cchar);}
.block-container{padding-top:1.5rem !important;padding-bottom:3rem !important;max-width:1280px !important;}
.avr-hdr{background:var(--cforest);border-bottom:3px solid var(--cgmd);padding:2rem 2.5rem;border-radius:16px;margin-bottom:2rem;}
.avr-eye{font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--cglt);margin:0 0 6px;}
.avr-hdr h1{font-family:'Playfair Display',serif;font-size:2.5rem;font-weight:700;color:#fff;margin:0 0 4px;line-height:1.15;}
.avr-hdr h1 span{color:var(--cglt);}
.avr-sub{font-size:.92rem;color:rgba(247,242,233,.6);font-weight:300;margin:0;}
.avr-badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
.avr-bdg{font-size:9px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--cglt);border:1px solid rgba(42,157,92,.35);border-radius:30px;padding:2px 9px;}
.sec{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:600;color:var(--cforest);border-left:4px solid var(--cgmd);padding-left:10px;margin:1.6rem 0 1rem;}
.sec-red{border-left-color:var(--cred);}
.fcard{background:#fff;border:1px solid var(--cborder);border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1rem;}
.fcard-t{font-size:9px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--cmuted);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--csand);}
.cbox{background:#FFF5F5;border:1.5px solid rgba(185,28,28,.2);border-radius:var(--r);padding:1.2rem 1.4rem;margin-bottom:1rem;}
.cbox-t{font-size:9px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--cred);margin-bottom:4px;}
.cbox-h{font-size:.82rem;color:var(--cmuted);line-height:1.5;margin:4px 0 0;}
.avr-div{border:none;border-top:1px solid var(--cborder);margin:1.6rem 0;}
.dec{border-radius:16px;padding:1.6rem 2rem;margin-bottom:1rem;}
.dec-app{background:#F0FDF4;border:2px solid #16A34A;}
.dec-con{background:#FFFBEB;border:2px solid #D97706;}
.dec-ref{background:#FEF2F2;border:2px solid var(--cred);}
.dec-lbl{font-size:9px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;margin:0 0 4px;}
.dec-app .dec-lbl{color:#16A34A;} .dec-con .dec-lbl{color:#D97706;} .dec-ref .dec-lbl{color:var(--cred);}
.dec h2{font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:700;margin:0 0 4px;line-height:1.1;}
.dec-app h2{color:#15803D;} .dec-con h2{color:#92400E;} .dec-ref h2{color:#991B1B;}
.dec-note{font-size:.85rem;color:var(--cslate);margin:0;}
.alert-ok{background:#F0FDF4;border:1px solid rgba(22,163,74,.25);border-left:4px solid #16A34A;border-radius:var(--r);padding:.8rem 1rem;font-size:.83rem;color:#14532D;margin-top:8px;line-height:1.5;}
.alert-ko{background:#FEF2F2;border:1px solid rgba(185,28,28,.25);border-left:4px solid var(--cred);border-radius:var(--r);padding:.8rem 1rem;font-size:.83rem;color:#7F1D1D;margin-top:8px;line-height:1.5;}
.scard{background:var(--cforest);border-radius:16px;padding:1.6rem;text-align:center;}
.scard-l{font-size:9px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--cglt);margin:0 0 8px;}
.scard-v{font-family:'Playfair Display',serif;font-size:3.5rem;font-weight:700;color:#fff;line-height:1;margin:0;}
.scard-mx{font-size:1.1rem;color:rgba(247,242,233,.35);font-weight:300;}
.sband{display:inline-block;margin-top:8px;padding:3px 12px;border-radius:30px;font-size:10px;font-weight:600;}
.sb-g{background:rgba(22,163,74,.2);color:#4ADE80;border:1px solid rgba(22,163,74,.3);}
.sb-a{background:rgba(217,119,6,.2);color:#FCD34D;border:1px solid rgba(217,119,6,.3);}
.sb-r{background:rgba(185,28,28,.2);color:#FCA5A5;border:1px solid rgba(185,28,28,.3);}
.kpi{background:#fff;border:1px solid var(--cborder);border-radius:var(--r);padding:1.1rem 1rem;text-align:center;margin-bottom:.6rem;}
.kpi-l{font-size:9px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--cmuted);margin:0 0 4px;}
.kpi-v{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:600;color:var(--cgdk);margin:0;line-height:1.2;}
.kpi-u{font-size:.75rem;color:var(--cmuted);font-weight:400;}
.kpi-r .kpi-v{color:var(--cred);}
.clim{background:var(--cforest);border-radius:var(--r);padding:1.2rem 1.4rem;color:#fff;margin-bottom:.6rem;}
.clim-e{font-size:9px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--cglt);margin:0 0 4px;}
.clim-n{font-family:'Playfair Display',serif;font-size:1rem;font-weight:600;margin:0 0 6px;color:#fff;}
.clim-v{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;line-height:1;margin:0;}
.clim-u{font-size:.75rem;color:rgba(247,242,233,.45);font-weight:300;margin-left:2px;}
.clim-d{font-size:.75rem;color:rgba(247,242,233,.55);line-height:1.45;margin-top:5px;}
.clim-s{display:inline-block;margin-top:5px;padding:2px 9px;border-radius:20px;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;}
.s-ok{background:rgba(22,163,74,.2);color:#4ADE80;border:1px solid rgba(22,163,74,.3);}
.s-w{background:rgba(217,119,6,.2);color:#FCD34D;border:1px solid rgba(217,119,6,.3);}
.s-d{background:rgba(185,28,28,.2);color:#FCA5A5;border:1px solid rgba(185,28,28,.3);}
.s-n{background:rgba(107,114,128,.2);color:#9CA3AF;border:1px solid rgba(107,114,128,.3);}
.ftbl-hdr{display:flex;align-items:center;padding:7px 14px;background:var(--csand);border:1px solid var(--cborder);border-radius:var(--r) var(--r) 0 0;}
.ftbl-row{display:flex;align-items:center;padding:8px 14px;background:#fff;border-left:1px solid var(--cborder);border-right:1px solid var(--cborder);border-bottom:1px solid var(--csand);}
.ftbl-last{border-radius:0 0 var(--r) var(--r);border-bottom:1px solid var(--cborder);}
.fc1{flex:2;font-size:.82rem;color:var(--cslate);}
.fc1h{font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--cmuted);}
.fc2{flex:1;font-family:'Playfair Display',serif;font-size:.88rem;font-weight:600;color:var(--cgdk);text-align:right;}
.fc2h{font-family:'DM Sans',sans-serif;font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--cmuted);text-align:right;}
.fc3{flex:1.5;text-align:right;}
.ftag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}
.t-p{background:#F0FDF4;color:#15803D;border:1px solid #BBF7D0;}
.t-n{background:#F9FAFB;color:#4B5563;border:1px solid #E5E7EB;}
.t-w{background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;}
.t-c{background:#FEF2F2;color:#991B1B;border:1px solid #FECACA;}
.pb-wrap{background:#fff;border:1px solid var(--cborder);border-radius:var(--r);padding:1.4rem 1.6rem;margin-bottom:.6rem;}
.pb-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}
.pb-nm{font-size:.82rem;font-weight:600;color:var(--cslate);}
.pb-pc{font-family:'Playfair Display',serif;font-size:.9rem;font-weight:600;}
.pb-tr{height:10px;background:var(--csand);border-radius:10px;overflow:hidden;margin-bottom:12px;}
.pb-tr:last-child{margin-bottom:0;}
.pb-fl{height:100%;border-radius:10px;}
.fl-g{background:#16A34A;} .fl-a{background:#D97706;} .fl-r{background:#B91C1C;}
.etag-ok{display:inline-block;background:#F0FDF4;color:#15803D;border:1px solid rgba(22,163,74,.25);padding:4px 12px;border-radius:4px;font-size:.78rem;font-weight:500;margin:3px 4px 3px 0;}
.etag-ko{display:inline-block;background:#FEF2F2;color:var(--cred);border:1px solid rgba(185,28,28,.25);padding:4px 12px;border-radius:4px;font-size:.78rem;font-weight:500;margin:3px 4px 3px 0;}
.advice{background:#fff;border:1px solid var(--cborder);border-left:4px solid var(--cgmd);border-radius:var(--r);padding:1.4rem 1.6rem;font-size:.9rem;line-height:1.75;color:var(--cslate);margin-top:.6rem;}
.info-box{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:var(--r);padding:.9rem 1.2rem;font-size:.82rem;color:var(--cmuted);margin-bottom:.8rem;line-height:1.5;}
.avr-ftr{border-top:1px solid var(--cborder);padding-top:1.4rem;margin-top:2rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;}
.ftr-nm{font-family:'Playfair Display',serif;font-size:.95rem;font-weight:600;color:var(--cforest);}
.ftr-mt{font-size:.72rem;color:var(--cmuted);letter-spacing:.06em;}
section[data-testid="stSidebar"]{background:var(--cforest) !important;border-right:1px solid rgba(42,157,92,.2) !important;}
section[data-testid="stSidebar"] *{color:var(--cream) !important;}
section[data-testid="stSidebar"] .stButton>button{background:rgba(30,122,68,.3) !important;border:1px solid rgba(42,157,92,.4) !important;color:var(--cream) !important;border-radius:6px !important;}
div[data-testid="stButton"]>button[kind="primary"]{background:var(--cgdk) !important;border:none !important;color:#fff !important;font-family:'DM Sans',sans-serif !important;font-weight:600 !important;font-size:.9rem !important;letter-spacing:.04em !important;padding:.65rem 2rem !important;border-radius:6px !important;box-shadow:0 2px 8px rgba(20,90,47,.35) !important;}
div[data-testid="stButton"]>button[kind="primary"]:hover{background:var(--cgmd) !important;}
div[data-testid="stButton"]>button[kind="secondary"]{background:transparent !important;border:1.5px solid var(--cgmd) !important;color:var(--cgdk) !important;font-family:'DM Sans',sans-serif !important;border-radius:6px !important;}
.stSelectbox label,.stNumberInput label,.stCheckbox label{font-family:'DM Sans',sans-serif !important;font-size:.82rem !important;font-weight:600 !important;color:var(--cslate) !important;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def vpd_status(v):
    if v is None: return "N/A", "s-n"
    if v < 1.0:   return "Optimal", "s-ok"
    if v < 2.5:   return "Modere",  "s-w"
    return "Critique", "s-d"

def ndwi_status(v):
    if v is None:  return "N/A", "s-n"
    if v > 0.2:    return "Vegetation Saine", "s-ok"
    if v > 0.0:    return "Stress Modere", "s-w"
    return "Secheresse", "s-d"

def ratio_tag(r):
    if r is None: return "N/A", "t-n"
    if r < 0.5:   return "Faible risque", "t-p"
    if r < 0.7:   return "Acceptable",    "t-n"
    if r < 1.0:   return "Eleve",         "t-w"
    return "Insuffisant", "t-c"

def score_band(s):
    if s >= 70: return "Profil Eligible",     "sb-g"
    if s >= 45: return "Profil Conditionnel", "sb-a"
    return "Profil Risque", "sb-r"

def fmt(v, f="{:,.0f}", na="—"):
    return f.format(v) if v is not None else na

def h(content):
    """Wrapper — single-line HTML safe for st.markdown"""
    st.markdown(content, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    h('<div style="padding:1.2rem 0 .8rem"><p style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:600;color:#F7F2E9;margin:0">AvoRuby Cash</p><p style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:rgba(42,157,92,.7);font-weight:500;margin:4px 0 0">Scoring Credit Agricole</p></div>')
    if st.button("Initialiser le Conseil", use_container_width=True):
        with st.spinner("Chargement..."):
            try:
                init_rag(force_rebuild_kb=True)
                st.success("Module RAG initialise.")
            except Exception as e:
                st.error(f"Erreur: {e}")
    st.markdown("---")
    h('<div style="margin-top:.8rem"><p style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:rgba(42,157,92,.6);font-weight:600;margin:0 0 8px">SEUILS SABC</p><p style="font-size:.75rem;color:rgba(247,242,233,.4);line-height:1.7;margin:0">Approuve &nbsp;&ge;&nbsp; 70<br>Conditionnel &nbsp; 45 &ndash; 69<br>Refuse &nbsp;&lt;&nbsp; 45</p></div>')
    st.markdown("---")
    h('<div style="text-align:center;padding-bottom:.5rem"><p style="font-family:Playfair Display,serif;font-size:.85rem;color:rgba(247,242,233,.35);margin:0">AvoRuby Cash</p></div>')

# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
h('<div class="avr-hdr"><p class="avr-eye">Systeme de Scoring de Credit Agricole</p><h1>Avoru<span>by</span> Cash</h1><p class="avr-sub">Evaluation intelligente du risque agricole &mdash; Scoring dynamique &middot; Analyse climatique &middot; Conseil personnalise</p></div>')

# ══════════════════════════════════════════════════════
# FORM
# ══════════════════════════════════════════════════════
h('<p class="sec">Profil de l\'Agriculteur</p>')

col1, col2, col3 = st.columns([1.2, 1.2, 1])
with col1:
    h('<div class="fcard"><p class="fcard-t">Exploitation</p></div>')
    culture     = st.selectbox("Culture", ["avocat","fruits_rouges"], format_func=lambda x: "Avocat (Persea americana)" if x=="avocat" else "Fruits Rouges (Fragaria / Rubus)")
    superficie  = st.number_input("Superficie (ha)", 0.1, 500.0, 5.0, 0.5)
    experience  = st.number_input("Annees d'experience", 0, 50, 5, 1)
with col2:
    h('<div class="fcard"><p class="fcard-t">Localisation &amp; Sol</p></div>')
    localisation   = st.selectbox("Localisation", ["Souss","Agadir","Kenitra","Larache","Marrakech","Moulay Bousselham","Tadla"])
    type_sol       = st.selectbox("Type de sol", ["Limoneux","Argileux","Sableux"])
    aversion_risque = st.slider("Coefficient d'aversion au risque (Gamma)", 0.0, 1.0, 0.5, 0.05, help="Gamma eleve attenue l'impact des chocs exogenes.")
with col3:
    h('<div class="fcard"><p class="fcard-t">Equipements</p></div>')
    irrigation    = st.checkbox("Irrigation goutte-a-goutte")
    solaire       = st.checkbox("Panneaux solaires")
    iot_sondes    = st.checkbox("Sondes IoT / Meteo")
    filets_serres = st.checkbox("Filets / Serres")

h('<p class="sec sec-red">Demande de Credit</p>')
h('<div class="cbox"><p class="cbox-t">Montant du credit demande</p></div>')

col_c, _ = st.columns([1, 2])
with col_c:
    montant_credit = st.number_input("Montant (MAD)", 0.0, 10_000_000.0, 50_000.0, 5_000.0, format="%.0f")

h('<hr class="avr-div">')
col_b, _ = st.columns([1, 3])
with col_b:
    run = st.button("Analyser le Dossier", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════
if run:
    form_data = {
        "culture": culture, "localisation": localisation, "type_sol": type_sol,
        "superficie_ha": superficie, "annees_experience": experience,
        "aversion_risque": aversion_risque, "irrigation": int(irrigation),
        "solaire": int(solaire), "iot_sondes": int(iot_sondes),
        "filets_serres": int(filets_serres), "montant_credit_demande": montant_credit,
    }
    with st.spinner("Execution du pipeline d'analyse..."):
        try:
            result = traiter_dossier_agriculteur(form_data)
        except Exception as e:
            st.error(f"Erreur : {e}"); st.stop()

    # credit override
    rev = result.get("revenu_brut") or (
        (result.get("production_estimee") or 0) * (result.get("prix_estime") or 0)
    )
    result["revenu_brut"] = rev
    result["montant_credit"] = montant_credit
    credit_msg = None
    credit_ok  = True
    if montant_credit > 0 and rev > 0:
        ratio = montant_credit / rev
        result["ratio_credit"] = ratio
        if ratio > 1.0:
            result["decision"] = "REFUSE"
            credit_ok  = False
            credit_msg = f"Credit demande ({montant_credit:,.0f} MAD) superieur au revenu brut estime ({rev:,.0f} MAD) — Taux : {ratio:.1%}. Capacite de remboursement insuffisante. Decision : REFUSE."
        elif ratio > 0.7 and result.get("decision") == "APPROUVE":
            result["decision"] = "CONDITIONNEL"
            credit_ok  = False
            credit_msg = f"Taux credit/revenu : {ratio:.1%} (seuil 70% depasse). Decision rebascule vers CONDITIONNEL. Garanties supplementaires requises."
        else:
            credit_msg = f"Credit demande : <strong>{montant_credit:,.0f} MAD</strong> &middot; Revenu brut estime : <strong>{rev:,.0f} MAD</strong> &middot; Taux de couverture : <strong>{ratio:.1%}</strong> — capacite adequate."
    else:
        result["ratio_credit"] = None

    st.session_state["result"]      = result
    st.session_state["form_data"]   = form_data
    st.session_state["credit_msg"]  = credit_msg
    st.session_state["credit_ok"]   = credit_ok
    st.session_state.pop("conseil_ia", None)

# ══════════════════════════════════════════════════════
# RÉSULTATS
# ══════════════════════════════════════════════════════
if "result" in st.session_state:
    result     = st.session_state["result"]
    form_data  = st.session_state["form_data"]
    credit_msg = st.session_state.get("credit_msg")
    credit_ok  = st.session_state.get("credit_ok", True)

    decision = result.get("decision", "CONDITIONNEL")
    score    = result.get("score_sabc") or 0

    h('<p class="sec">Resultats de l\'Evaluation</p>')

    # — Decision + Score —
    dcls = {"APPROUVE":"dec-app","CONDITIONNEL":"dec-con","REFUSE":"dec-ref"}
    dlbl = {"APPROUVE":"Decision Favorable","CONDITIONNEL":"Decision Conditionnelle","REFUSE":"Decision Defavorable"}
    dtit = {"APPROUVE":"Credit Approuve","CONDITIONNEL":"Credit Conditionnel","REFUSE":"Credit Refuse"}
    c_dec, c_sc = st.columns([1.6, 1])
    with c_dec:
        h(f'<div class="dec {dcls.get(decision,"dec-con")}"><p class="dec-lbl">{dlbl.get(decision,"")}</p><h2>{dtit.get(decision,"")}</h2><p class="dec-note">Analyse basee sur le score SABC, les predictions climatiques, le profil socio-economique et la demande de credit.</p></div>')
        if credit_msg:
            cls_a = "alert-ok" if credit_ok else "alert-ko"
            h(f'<div class="{cls_a}">{credit_msg}</div>')
    with c_sc:
        bl, bc = score_band(score)
        h(f'<div class="scard"><p class="scard-l">Score SABC</p><p class="scard-v">{score}<span class="scard-mx">/100</span></p><span class="sband {bc}">{bl}</span></div>')

    # — KPIs —
    h('<p class="sec">Estimations Intermediaires</p>')
    rendement  = result.get("rendement_estime")
    prix_est   = result.get("prix_estime")
    production = result.get("production_estimee")
    rev_brut   = result.get("revenu_brut")
    ratio_c    = result.get("ratio_credit")

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: h(f'<div class="kpi"><p class="kpi-l">Rendement estime</p><p class="kpi-v">{fmt(rendement)}<span class="kpi-u"> kg/ha</span></p></div>')
    with k2: h(f'<div class="kpi"><p class="kpi-l">Prix estime</p><p class="kpi-v">{fmt(prix_est,"{:.2f}")}<span class="kpi-u"> MAD/kg</span></p></div>')
    with k3: h(f'<div class="kpi"><p class="kpi-l">Production totale</p><p class="kpi-v">{fmt(production)}<span class="kpi-u"> kg</span></p></div>')
    with k4: h(f'<div class="kpi"><p class="kpi-l">Revenu brut estime</p><p class="kpi-v">{fmt(rev_brut)}<span class="kpi-u"> MAD</span></p></div>')
    with k5:
        rd = f"{ratio_c:.1%}" if ratio_c is not None else "—"
        rc = "kpi-r" if (ratio_c or 0) > 0.7 else ""
        h(f'<div class="kpi {rc}"><p class="kpi-l">Taux credit / revenu</p><p class="kpi-v">{rd}</p></div>')

    # — Indicateurs Climatiques — 2 colonnes natives Streamlit —
    h('<p class="sec">Indicateurs Agro-Climatiques Predits</p>')
    vpd_val  = result.get("vpd_j7")
    ndwi_val = result.get("ndwi_j30")
    vs_t, vs_c  = vpd_status(vpd_val)
    ns_t, ns_c  = ndwi_status(ndwi_val)
    vd = f"{vpd_val:.3f}"  if vpd_val  is not None else "&mdash;"
    nd = f"{ndwi_val:.3f}" if ndwi_val is not None else "&mdash;"
    vu = "kPa" if vpd_val is not None else "non integre"
    nu = "indice" if ndwi_val is not None else "non integre"

    cl1, cl2 = st.columns(2)
    with cl1:
        h(f'<div class="clim"><p class="clim-e">Indicateur Climatique &middot; Horizon J+7</p><p class="clim-n">Deficit de Pression de Vapeur</p><p class="clim-v">{vd}<span class="clim-u">{vu}</span></p><p class="clim-d">Mesure du stress hydrique physiologique de la culture. Seuil critique : VPD &gt; 2.5 kPa.</p><span class="clim-s {vs_c}">{vs_t}</span></div>')
    with cl2:
        h(f'<div class="clim"><p class="clim-e">Indicateur Vegetation &middot; Horizon J+30</p><p class="clim-n">Teneur en Eau de la Vegetation</p><p class="clim-v">{nd}<span class="clim-u">{nu}</span></p><p class="clim-d">Etat hydrique de la vegetation et risque de secheresse sur la parcelle. Alerte : NDWI &lt; -0.15.</p><span class="clim-s {ns_c}">{ns_t}</span></div>')



    # — Features Table — header + lignes séparées —
    h('<p class="sec">Analyse Detaillee de l\'Exploitation</p>')
    stress_t = result.get("stress_thermique")
    stress_h = result.get("stress_hydrique")
    risque   = result.get("risque_brut")

    def feat_row(name, value, tag, tcls, last=False):
        lc = " ftbl-last" if last else ""
        h(f'<div class="ftbl-row{lc}"><span class="fc1">{name}</span><span class="fc2">{value}</span><span class="fc3"><span class="ftag {tcls}">{tag}</span></span></div>')

    h('<div class="ftbl-hdr"><span class="fc1 fc1h">Indicateur</span><span class="fc2 fc2h">Valeur</span><span class="fc3 fc2h">Interpretation</span></div>')

    if vpd_val is not None:
        vt = "Optimal" if vpd_val<1 else ("Modere" if vpd_val<2.5 else "Critique")
        vc = "t-p" if vpd_val<1 else ("t-w" if vpd_val<2.5 else "t-c")
    else: vt,vc = "Non integre","t-n"
    feat_row("VPD Moyen (J+7)", f"{vpd_val:.3f} kPa" if vpd_val else "—", vt, vc)

    if ndwi_val is not None:
        nt2 = "Saine" if ndwi_val>0.2 else ("Modere" if ndwi_val>0 else "Secheresse")
        nc2 = "t-p" if ndwi_val>0.2 else ("t-w" if ndwi_val>0 else "t-c")
    else: nt2,nc2 = "Non integre","t-n"
    feat_row("NDWI Moyen (J+30)", f"{ndwi_val:.3f}" if ndwi_val is not None else "—", nt2, nc2)

    st_t = "Actif" if stress_t==1 else ("Inactif" if stress_t==0 else "—")
    sc_t = "t-c" if stress_t==1 else ("t-p" if stress_t==0 else "t-n")
    tl_t = "Critique" if stress_t==1 else ("Nominal" if stress_t==0 else "Non integre")
    feat_row("Stress Thermique", st_t, tl_t, sc_t)

    st_h = "Actif" if stress_h==1 else ("Inactif" if stress_h==0 else "—")
    sc_h = "t-c" if stress_h==1 else ("t-p" if stress_h==0 else "t-n")
    tl_h = "Critique" if stress_h==1 else ("Nominal" if stress_h==0 else "Non integre")
    feat_row("Stress Hydrique", st_h, tl_h, sc_h)

    if risque is not None:
        rl = "Risque Eleve" if risque>0.6 else ("Risque Modere" if risque>0.35 else "Risque Faible")
        rc2 = "t-c" if risque>0.6 else ("t-w" if risque>0.35 else "t-p")
        feat_row("Risque Brut Final", f"{risque:.4f}", rl, rc2)
    else:
        feat_row("Risque Brut Final", "—", "Non integre", "t-n")

    if rev_brut:
        rv_l = "Solide" if rev_brut>200000 else ("Correct" if rev_brut>80000 else "Faible")
        rv_c = "t-p" if rev_brut>200000 else ("t-n" if rev_brut>80000 else "t-w")
        feat_row("Revenu Brut Estime", f"{rev_brut:,.0f} MAD", rv_l, rv_c)
    else:
        feat_row("Revenu Brut Estime", "—", "Non integre", "t-n")

    rt_l, rt_c = ratio_tag(ratio_c)
    feat_row("Taux Credit / Revenu", f"{ratio_c:.1%}" if ratio_c else "—", rt_l, rt_c, last=True)

    # — Probabilités — une ligne à la fois —
    h('<p class="sec">Distribution des Probabilites</p>')
    probas = result.get("probabilites", {})
    pmap   = {"APPROUVE":("fl-g","#16A34A"), "CONDITIONNEL":("fl-a","#D97706"), "REFUSE":("fl-r","#B91C1C")}

    h('<div class="pb-wrap">')
    for cls in ["APPROUVE","CONDITIONNEL","REFUSE"]:
        pct     = probas.get(cls, 0)
        fc, col = pmap[cls]
        w       = max(pct, 1)
        h(f'<div class="pb-hdr"><span class="pb-nm">{cls}</span><span class="pb-pc" style="color:{col}">{pct}%</span></div><div class="pb-tr"><div class="pb-fl {fc}" style="width:{w}%"></div></div>')
    h('</div>')

    # — Equipements —
    h('<p class="sec">Equipements</p>')
    equip = {
        "Irrigation Goutte-a-goutte": int(form_data.get("irrigation",0)),
        "Panneaux Solaires":          int(form_data.get("solaire",0)),
        "Sondes IoT / Meteo":         int(form_data.get("iot_sondes",0)),
        "Filets / Serres":            int(form_data.get("filets_serres",0)),
    }
    tags = "".join(
        f'<span class="etag-ok">{n}</span>' if v else f'<span class="etag-ko">{n} (manquant)</span>'
        for n, v in equip.items()
    )
    h(f'<div style="margin-bottom:1rem">{tags}</div>')

    # — Conseil RAG —
    h('<hr class="avr-div">')
    h('<p class="sec">Recommandations</p>')
    cb, _ = st.columns([1, 3])
    with cb:
        if st.button("Obtenir des Recommandations", use_container_width=True):
            with st.spinner("Generation du conseil..."):
                try:
                    profil_rag = {
                        "culture": form_data.get("culture"), "localisation": form_data.get("localisation"),
                        "superficie": form_data.get("superficie_ha"), "sabc": score,
                        "decision": decision, "equipements_manquants": result.get("equipements_manquants",[]),
                        "vpd_j7": vpd_val, "ndwi_j30": ndwi_val, "ratio_credit": ratio_c,
                    }
                    st.session_state["conseil_ia"] = get_assistant_advice(profil_rag)
                except Exception as e:
                    st.error(f"Erreur RAG : {e}")

    if "conseil_ia" in st.session_state:
        h(f'<div class="advice">{st.session_state["conseil_ia"]}</div>')

    # — Footer —
    h('<div class="avr-ftr"><span class="ftr-nm">AvoRuby Cash</span><span class="ftr-mt">Scoring &middot; Analyse Climatique &middot; Conseil Agricole</span></div>')