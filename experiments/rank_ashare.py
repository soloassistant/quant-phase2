"""A股「黄金坑」筛选 v3：绝对趋势 + 跑赢大盘（修正"池内截面排名"的伪筛选）。

v2 批评：在"一群快溺死的便宜股"里做截面百分位，选出的只是"挣扎最剧烈的"，不是"活下来的"。
v3 修正：绝对趋势(站上自己200日线) + 跑赢大盘(近120日跑赢沪深300) 双硬门槛；排序列后。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"
DATA = ROOT / "data_ashare"

FUND = {
    '000737':('peg',0.16),'603799':('peg',0.19),'605117':('peg',0.20),'300972':('peg',0.21),
    '000657':('peg',0.25),'601066':('peg',0.26),'300037':('peg',0.28),'000807':('peg',0.30),
    '600026':('peg',0.35),'300059':('peg',0.36),'300432':('peg',0.37),'003022':('peg',0.41),
    '002463':('peg',0.43),'600415':('peg',0.47),'603225':('peg',0.48),'688183':('peg',0.50),
    '001965':('peg',0.55),'300953':('peg',0.57),'600522':('peg',0.58),'600489':('peg',0.61),
    '603338':('peg',0.62),'000938':('peg',0.70),'300033':('peg',0.76),'688297':('peg',0.82),
    '603345':('peg',0.84),'688099':('peg',0.89),'688019':('peg',0.94),'300623':('peg',0.98),
    '688578':('pe',2.0),'300251':('pe',10.0),'600519':('pe',15.0),
    '600816':('pb',0.0),'601136':('pb',2.0),'601601':('pb',6.0),
    '002821':('disc',152.68,154.77),'601866':('disc',2.35,2.66),'002078':('disc',13.28,17.71),
    '600547':('disc',24.91,30.41),'600085':('disc',23.96,25.18),'002487':('disc',32.02,42.30),
    '601018':('disc',3.51,4.82),'601099':('disc',3.37,4.18),'002044':('disc',4.84,5.93),
    '600008':('disc',2.76,3.73),'002603':('disc',15.48,20.39),'000725':('disc',4.41,4.71),
    '603529':('disc',21.34,22.58),'000937':('disc',4.19,4.25),'600426':('disc',18.48,21.40),
    '600685':('disc',25.37,32.03),'000651':('disc',38.89,61.80),'603786':('disc',39.11,67.74),
    '002294':('disc',29.78,34.03),'000425':('disc',6.54,8.02),'600486':('disc',54.38,64.18),
    '600031':('disc',18.27,23.26),'600862':('disc',18.35,28.22),'600050':('disc',4.23,4.97),
    '002311':('disc',47.67,55.62),
}
CHEAP_MIN=0.30; BLACKSWAN_1D=-0.15; BLACKSWAN_5D=-0.25; TOP_CORE=12; RS_WIN=120

def cheap_raw(code):
    f=FUND[code]; t=f[0]
    if t in ('peg','pe','pb'): return -f[1]
    price,low=f[1],f[2]; return (low-price)/low

def load_idx():
    idx={}
    for sym in ('sh000300','sh000905'):
        p=DATA/f"_index_{sym}.csv"
        if p.exists():
            idx[sym]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
    return idx

def diag_one(code, idx):
    raw=pd.read_csv(DATA/f"{code}.csv",parse_dates=['date']).set_index('date').sort_index()
    c=raw['close'].astype(float).ffill(); v=raw['volume'].astype(float); n=len(c)
    d={'price':float(c.iloc[-1])}
    d['ret_60']=float(c.iloc[-1]/c.iloc[-min(61,n)]-1)
    d['ret_120']=float(c.iloc[-1]/c.iloc[-min(121,n)]-1)
    d['ret_250']=float(c.iloc[-1]/c.iloc[-min(251,n)]-1)
    sma=c.rolling(200,min_periods=200).mean(); sv=float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else float('nan')
    d['dist_sma200']=float(c.iloc[-1]/sv-1) if sv==sv else float('nan')
    w=min(250,n)
    d['max_1d']=float(c.pct_change().iloc[-w:].min())
    d['max_5d']=float((c/c.shift(5)-1).iloc[-w:].min())
    seg=c.iloc[-w:]; d['max_dd_250']=float((seg/seg.cummax()-1).min())
    vv=v.iloc[-w:]; med=float(vv.median()) if vv.median()>0 else np.nan
    d['vol_spike']=float(vv.max()/med) if med==med and med>0 else float('nan')
    d['vol20']=float(c.pct_change().iloc[-20:].std(ddof=1)*np.sqrt(252))
    for sym,s in idx.items():
        both=pd.concat([c,s],axis=1).dropna(); key='excess300' if sym=='sh000300' else 'excess500'
        if len(both)>RS_WIN:
            d[key]=float((both.iloc[-1,0]/both.iloc[-RS_WIN-1,0]-1)-(both.iloc[-1,1]/both.iloc[-RS_WIN-1,1]-1))
        else: d[key]=float('nan')
    return d

def main():
    A=json.loads((OUT/'ashare_screen.json').read_text(encoding='utf-8'))
    rows={r['code']:dict(r) for r in A['rows']}
    idx=load_idx()
    dfc=pd.DataFrame([{'code':c,'type':FUND[c][0],'cheap_raw':cheap_raw(c)} for c in FUND])
    dfc['cheap_pct']=dfc.groupby('type')['cheap_raw'].rank(pct=True)
    cp=dict(zip(dfc['code'],dfc['cheap_pct']))
    for c in FUND:
        rows[c].update(diag_one(c,idx)); rows[c]['cheap_pct']=float(cp[c])
    for c in FUND:
        r=rows[c]; rs=[]
        if r['cheap_pct']<CHEAP_MIN: rs.append('不够便宜')
        if not (np.isfinite(r.get('dist_sma200',np.nan)) and r['dist_sma200']>0): rs.append('未站上200日线')
        if not (np.isfinite(r.get('excess300',np.nan)) and r['excess300']>0): rs.append('跑输沪深300')
        if (np.isfinite(r.get('max_1d',np.nan)) and r['max_1d']<BLACKSWAN_1D) or (np.isfinite(r.get('max_5d',np.nan)) and r['max_5d']<BLACKSWAN_5D): rs.append('黑天鹅')
        r['veto']='; '.join(rs); r['flag_volume']=bool(np.isfinite(r.get('vol_spike',np.nan)) and r['vol_spike']>5.0)
    surv=[c for c in FUND if not rows[c]['veto']]
    if surv:
        ex=pd.Series({c:rows[c]['excess300'] for c in surv}); m6=pd.Series({c:rows[c]['ret_60'] for c in surv})
        exr=ex.rank(pct=True); m6r=m6.rank(pct=True)
        for c in surv: rows[c]['score']=float(0.40*rows[c]['cheap_pct']+0.35*exr[c]+0.25*m6r[c])
    surv.sort(key=lambda c:-rows[c].get('score',0))
    rk={c:i+1 for i,c in enumerate(surv)}
    for c in FUND:
        r=rows[c]; r['rank']=rk.get(c); r['suggest']='否决' if r['veto'] else ('核心' if rk[c]<=TOP_CORE else '观察')
    from collections import Counter
    vc=Counter()
    for c in FUND:
        for x in rows[c]['veto'].split('; '):
            if x: vc[x]+=1
    out={'gates':{'cheap_min':CHEAP_MIN,'abs_trend':'dist_sma200>0','beat_index':'excess300>0','bs1':BLACKSWAN_1D,'bs5':BLACKSWAN_5D},
         'n_total':len(FUND),'n_veto':len(FUND)-len(surv),'veto_breakdown':dict(vc),'n_survivor':len(surv),
         'n_core':min(TOP_CORE,len(surv)),
         'ranking':sorted([rows[c] for c in FUND],key=lambda r:(r['rank'] is None,r['rank'] or 999))}
    (OUT/'ashare_rank.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(f"总59  否决 {len(FUND)-len(surv)}  幸存 {len(surv)}  明细 {dict(vc)}")
    if surv:
        print(f"{'#':>2}{'代码':7s}{'名称':10s}{'体系':5s}{'便宜':>6s}{'距200':>7s}{'120日':>7s}{'超300':>7s}{'超500':>7s}")
        for c in surv[:TOP_CORE]:
            r=rows[c]; print(f"{r['rank']:>2}{c:7s}{r['name']:10s}{r['system']:5s}{r['cheap_pct']:6.2f}"
                f"{r['dist_sma200']*100:6.1f}%{r['ret_120']*100:6.0f}%{r['excess300']*100:+6.0f}%{r.get('excess500',0)*100:+6.0f}%")

if __name__=='__main__':
    main()
