import json, numpy as np
from scipy.optimize import brentq
def conc_errors(rows, sel):
    """Leave-one-channel-out concentration prediction: calibrate ln G = a + b u + c u^2 (u = ln c)
    on the other channels of the same gap and design, invert on the held-out channel."""
    errs = []
    R = [r for r in rows if sel(r)]
    for g in sorted(set(r['gap'] for r in R)):
        Rg = [r for r in R if r['gap'] == g]
        for ch in sorted(set(r['ch'] for r in Rg)):
            tr = [r for r in Rg if r['ch'] != ch]; te = [r for r in Rg if r['ch'] == ch]
            u = np.log([r['c'] for r in tr]); y = np.array([r['lnG'] for r in tr])
            co = np.polyfit(u, y, 2)
            for r in te:
                fz = lambda uu: np.polyval(co, uu) - r['lnG']
                lo, hi = np.log(0.03), np.log(15)
                try:
                    uh = brentq(fz, lo, hi)
                except ValueError:
                    uh = lo if abs(fz(lo)) < abs(fz(hi)) else hi
                errs.append(dict(gap=g, c=r['c'], e=uh - np.log(r['c'])))
    return errs
if __name__ == '__main__':
    o = json.load(open('../results/realdata_results.json')); rows = o['rows']
    for m in [4,5,6]:
        for dn in ['log-uniform','heuristic','nominal','robust']:
            E = conc_errors(rows, lambda r: r['m']==m and r['design']==dn and r['op']=='single')
            e = np.array([x['e'] for x in E])
            wc = max(np.sqrt(np.mean([x['e']**2 for x in E if x['c']==c])) for c in [0.1,0.5,1,2,4])
            wg = max(np.sqrt(np.mean([x['e']**2 for x in E if x['gap']==g])) for g in ['80','120','160','200','240'])
            print(m, dn, 'RMS rel conc err %.3f  worst-conc %.3f  worst-gap %.3f'%(np.sqrt(np.mean(e**2)), wc, wg))
    E = conc_errors(rows, lambda r: r['m']==0)
    # full spectrum reference
    full=[dict(r, lnG=r['lnG_full']) for r in rows if r['m']==6 and r['design']=='robust' and r['op']=='single']
    E = conc_errors(full, lambda r: True); e=np.array([x['e'] for x in E]); print('full-spectrum RMS %.3f'%np.sqrt(np.mean(e**2)))
