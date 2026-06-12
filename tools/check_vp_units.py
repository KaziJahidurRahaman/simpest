import csv, math, statistics

def es(t):
    return 0.6108 * math.exp((17.27 * t) / (t + 237.3))

def classify(rh):
    # physically plausible RH range: 20-100%
    return 20.0 <= rh <= 100.0

datasets = {
    'indiana':      r'c:\ParamVC\Research\simpest\docs\examples\data\weather_sim\indiana.txt',
    'indiana_sim':  r'c:\ParamVC\Research\simpest\docs\examples\data\weather_sim\indiana_sim.txt',
    'sevilla':      r'c:\ParamVC\Research\simpest\docs\examples\data\weather_sim\sevilla.txt',
    'wageningen':   r'c:\ParamVC\Research\simpest\docs\examples\data\weather_sim\wageningen.txt',
}

for name, path in datasets.items():
    try:
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter='\t')
            reader.fieldnames = [h.lstrip('\ufeff').strip() for h in reader.fieldnames]
            rows = list(reader)

        rh_kpa_all, rh_hpa_all = [], []
        ok_kpa = ok_hpa = 0

        for r in rows:
            vp = float(r['VapourPressure'])
            tmin = float(r['Tmin'])
            es_min = es(tmin)
            if es_min <= 0:
                continue
            rh_kpa = 100.0 * vp / es_min          # assume VP already in kPa
            rh_hpa = 100.0 * (vp / 10.0) / es_min # assume VP in hPa → /10 to get kPa
            rh_kpa_all.append(rh_kpa)
            rh_hpa_all.append(rh_hpa)
            if classify(rh_kpa): ok_kpa += 1
            if classify(rh_hpa): ok_hpa += 1

        n = len(rh_kpa_all)
        vp_vals  = [float(r['VapourPressure']) for r in rows[:5]]
        esmin_5  = [round(es(float(r['Tmin'])), 4) for r in rows[:5]]
        rh_kpa_5 = [round(100.0 * vp_vals[i] / esmin_5[i], 1) for i in range(5)]
        rh_hpa_5 = [round(100.0 * (vp_vals[i] / 10.0) / esmin_5[i], 1) for i in range(5)]

        print(f"\n=== {name} ({len(rows)} rows) ===")
        print(f"  VP (first 5):              {[round(v, 4) for v in vp_vals]}")
        print(f"  es(Tmin) (first 5):        {esmin_5}")
        print(f"  RH if kPa (first 5):       {rh_kpa_5}%")
        print(f"  RH if hPa→/10 (first 5):  {rh_hpa_5}%")
        print(f"  VP median: {round(statistics.median(float(r['VapourPressure']) for r in rows), 4)}")
        print(f"  Plausible rows (20-100%) if kPa: {ok_kpa}/{n}  ({100*ok_kpa/n:.1f}%)")
        print(f"  Plausible rows (20-100%) if hPa: {ok_hpa}/{n}  ({100*ok_hpa/n:.1f}%)")
        med_kpa = round(statistics.median(rh_kpa_all), 1)
        med_hpa = round(statistics.median(rh_hpa_all), 1)
        print(f"  Median RH if kPa: {med_kpa}%  |  Median RH if hPa: {med_hpa}%")
        if ok_kpa > ok_hpa:
            print(f"  --> VERDICT: kPa more likely")
        elif ok_hpa > ok_kpa:
            print(f"  --> VERDICT: hPa more likely")
        else:
            print(f"  --> VERDICT: AMBIGUOUS")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
