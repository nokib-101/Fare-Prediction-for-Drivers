"""
Loads the v3 artifact from garibook_driver_bid_model_v3.ipynb and turns a booking into
a bid band.

v3 keeps one model per trip type. BidModel.quote() routes on trip type and runs that
segment's own feature list, rate-card encoder, ensemble and conformal band.

The feature-building code below (maps, parsers, standardize, add_features,
RouteRateEncoder) is a copy of the notebook's
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from catboost import CatBoostRegressor, Pool
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


def _blob_to_file(blob, suffix):
    mode = "w" if isinstance(blob, str) else "wb"
    fh = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False)
    fh.write(blob)
    fh.close()
    return fh.name


def restore(entry):
    """Rebuild a live booster from the portable blob the notebook saved.

    Each library reads back its own stable export format, so the serving
    versions no longer have to match the versions that trained the model.
    """
    if "model" in entry:
        return entry                       # already a live object
    fmt = entry.get("fmt")
    e = dict(entry)
    if fmt == "lgb_txt":
        e["model"] = lgb.Booster(model_str=entry["blob"])
    elif fmt == "xgb_json":
        p = _blob_to_file(entry["blob"], ".json")
        try:
            m = xgb.XGBRegressor(enable_categorical=True)
            m.load_model(p)
        finally:
            os.unlink(p)
        e["model"] = m
    elif fmt == "cat_cbm":
        if not HAS_CATBOOST:
            raise ImportError("This model includes a CatBoost member but catboost is "
                              "not installed. Run: python -m pip install catboost")
        p = _blob_to_file(entry["blob"], ".cbm")
        try:
            m = CatBoostRegressor()
            m.load_model(p)
        finally:
            os.unlink(p)
        e["model"] = m
    else:
        raise ValueError(f"unknown serialisation format: {fmt!r}")
    e.pop("blob", None)
    return e

# --------------------------------------------------------------------------- maps

DIVISION_MAP = {
    "dhaka": "Dhaka", "ঢাকা": "Dhaka", "dhaka division": "Dhaka", "ঢাকা বিভাগ": "Dhaka",
    "ダッカ": "Dhaka", "ঢাকা-১১৮৮০": "Dhaka", "default pickup division": np.nan,
    "chattogram": "Chattogram", "chittagong": "Chattogram", "চট্টগ্রাম": "Chattogram",
    "chittagong division": "Chattogram", "চট্টগ্রাম বিভাগ": "Chattogram",
    "chattogram division": "Chattogram",
    "mymensingh": "Mymensingh", "ময়মনসিংহ": "Mymensingh", "ময়মনসিংহ বিভাগ": "Mymensingh",
    "rajshahi": "Rajshahi", "রাজশাহী": "Rajshahi", "rajshahi division": "Rajshahi",
    "khulna": "Khulna", "খুলনা": "Khulna", "khulna division": "Khulna",
    "sylhet": "Sylhet", "সিলেট": "Sylhet", "sylhet division": "Sylhet",
    "rangpur": "Rangpur", "রংপুর": "Rangpur", "rangpur division": "Rangpur",
    "barisal": "Barishal", "barishal": "Barishal", "বরিশাল": "Barishal",
    "バリサル": "Barishal", "barishal division": "Barishal",
}

DISTRICT_MAP = {
    "ঢাকা": "Dhaka", "চট্টগ্রাম": "Chattogram", "chittagong": "Chattogram",
    "গাজীপুর": "Gazipur", "নারায়ণগঞ্জ": "Narayanganj", "কুমিল্লা": "Cumilla",
    "comilla": "Cumilla", "ময়মনসিংহ": "Mymensingh", "টাঙ্গাইল": "Tangail",
    "চাঁদপুর": "Chandpur", "কিশোরগঞ্জ": "Kishoreganj", "সিলেট": "Sylhet",
    "খুলনা": "Khulna", "রাজশাহী": "Rajshahi", "রংপুর": "Rangpur", "বরিশাল": "Barishal",
    "noakhali": "Noakhali", "feni": "Feni", "bogra": "Bogura", "bogura": "Bogura",
    "jessore": "Jashore", "jashore": "Jashore", "barisal": "Barishal",
}

AC_MAP = {"ac": "AC", "এসি": "AC", "dual-ac": "Dual-AC", "ডুয়েল এসি": "Dual-AC",
          "dual ac": "Dual-AC", "non-ac": "Non-AC", "non ac": "Non-AC", "0": "Unknown"}

CAR_MODEL_KEYWORDS = [
    ("hiace", "HiAce"), ("hi-ace", "HiAce"), ("hi ace", "HiAce"),
    ("noah", "Noah"), ("voxy", "Voxy"), ("esquire", "Esquire"),
    ("axio", "Axio"), ("fielder", "Fielder"), ("corolla", "Corolla"),
    ("allion", "Allion"), ("premio", "Premio"), ("probox", "Probox"),
    ("belta", "Belta"), ("vitz", "Vitz"), ("aqua", "Aqua"), ("prius", "Prius"),
    ("carina", "Carina"), ("sienta", "Sienta"), ("rush", "Rush"), ("harrier", "Harrier"),
    ("crown", "Crown"), ("micro", "Microbus"), ("coaster", "Coaster"),
    ("pajero", "Pajero"), ("cr-v", "CRV"), ("crv", "CRV"), ("x-trail", "XTrail"),
    ("hybrid", "OtherHybrid"),
]

DHAKA = (23.8103, 90.4125)
BD_BBOX = dict(lat=(20.0, 27.0), lon=(87.0, 93.0))

ROUTE_COLS = [
    "route_ppk", "route_ppk_n", "baseline_fare", "log_baseline_fare",
    "kmband_ppk", "kmband_baseline_fare", "car_rate_mult",
    "blend_baseline_fare", "log_blend_baseline_fare", "pickup_district_volume",
]

# ---------------------------------------------------------------------- parsers


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


def parse_dt(s):
    out = pd.to_datetime(s, format="%B %d, %Y, %I:%M %p", errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():
        out.loc[miss] = pd.to_datetime(s[miss], errors="coerce", format="mixed")
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def norm_label(s, mapping):
    mapped = s.astype(str).str.strip().str.lower().map(mapping)
    out = mapped.fillna(s.astype(str).str.strip().str.title())
    return out.replace({"Nan": np.nan, "None": np.nan, "": np.nan})


def group_car_model(s):
    low = s.astype(str).str.lower()
    out = pd.Series("Other", index=s.index, dtype=object)
    for kw, label in CAR_MODEL_KEYWORDS:
        hit = low.str.contains(kw, regex=False, na=False) & (out == "Other")
        out.loc[hit] = label
    out.loc[s.isna()] = "Unknown"
    return out


def clean_coord(lat, lon):
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    ok = lat.between(*BD_BBOX["lat"]) & lon.between(*BD_BBOX["lon"])
    return lat.where(ok), lon.where(ok)


# --------------------------------------------------------------- feature pipeline


def standardize(raw, target_raw="Driver Fare"):
    d = pd.DataFrame(index=raw.index)
    g = lambda c: raw[c] if c in raw.columns else pd.Series(np.nan, index=raw.index, dtype=object)

    d["fare"] = to_num(g(target_raw))
    d["km"] = to_num(g("Total Km"))
    d["hours"] = pd.to_numeric(g("Hours"), errors="coerce").fillna(0.0).clip(0, 24)

    d["trip_type"] = (g("Trip Type").astype(str).str.strip().str.lower()
                      .str.replace(r"[\s\-]+", "_", regex=True))
    d["car_type"] = g("Car Type").astype(str).str.strip()
    d["seats"] = pd.to_numeric(g("No Of Seat").astype(str).str.extract(r"(\d+)")[0],
                               errors="coerce")

    d["pickup_dt"] = parse_dt(g("Pickup Date Time"))
    d["created_dt"] = parse_dt(g("Created At"))
    d["return_dt"] = parse_dt(g("Return Date Time"))

    d["pickup_division"] = norm_label(g("Pickup Division"), DIVISION_MAP)
    d["dropoff_division"] = norm_label(g("Dropoff Division"), DIVISION_MAP)
    d["pickup_district"] = norm_label(g("Pickup District"), DISTRICT_MAP)
    d["dropoff_district"] = norm_label(g("Dropoff District"), DISTRICT_MAP)
    d["pickup_lat"], d["pickup_long"] = clean_coord(g("Pickup Lat"), g("Pickup Long"))
    d["dropoff_lat"], d["dropoff_long"] = clean_coord(g("Dropoff Lat"), g("Dropoff Long"))

    ac = norm_label(g("Driver Car Ac Status"), AC_MAP).fillna("Unknown")
    d["ac_status"] = ac.where(ac.isin(["AC", "Dual-AC", "Non-AC"]), "Unknown")
    my = pd.to_numeric(g("Driver Car Model Year"), errors="coerce")
    d["car_model_year"] = my.where(my.between(1980, 2027))
    d["car_model_group"] = group_car_model(g("Driver Car Model Name"))
    d["driver_rating"] = pd.to_numeric(g("Driver Rating"), errors="coerce").replace(0, np.nan)
    d["driver_trips"] = pd.to_numeric(g("Driver No Of Trips"), errors="coerce")
    d["driver_car_rating"] = pd.to_numeric(g("Driver Car Rating"), errors="coerce").replace(0, np.nan)

    d["promo_amount_raw"] = pd.to_numeric(g("Promo Code Amount"), errors="coerce")
    d["promo_discount_raw"] = pd.to_numeric(g("Promo Discount Amount (BDT)"), errors="coerce")

    d["trip_status"] = g("Trip Status").astype(str).str.strip().str.lower()
    d["booking_id"] = g("Booking ID")
    return d


def add_features(df):
    d = df.copy()

    d["log_km"] = np.log1p(d["km"])
    d["haversine_km"] = haversine_km(d["pickup_lat"], d["pickup_long"],
                                     d["dropoff_lat"], d["dropoff_long"])
    d["log_haversine_km"] = np.log1p(d["haversine_km"])
    d["detour_ratio"] = (d["km"] / d["haversine_km"].replace(0, np.nan)).clip(0, 20)
    d["bearing"] = bearing_deg(d["pickup_lat"], d["pickup_long"],
                               d["dropoff_lat"], d["dropoff_long"])

    d["is_round"] = d["trip_type"].eq("round_way").astype(int)
    d["km_per_leg"] = np.where(d["is_round"] == 1, d["km"] / 2, d["km"])

    d["pickup_dist_from_dhaka"] = haversine_km(d["pickup_lat"], d["pickup_long"], *DHAKA)
    d["dropoff_dist_from_dhaka"] = haversine_km(d["dropoff_lat"], d["dropoff_long"], *DHAKA)
    d["lat_diff"] = d["dropoff_lat"] - d["pickup_lat"]
    d["long_diff"] = d["dropoff_long"] - d["pickup_long"]
    d["same_district"] = (d["pickup_district"] == d["dropoff_district"]).astype(int)
    d["same_division"] = (d["pickup_division"] == d["dropoff_division"]).astype(int)
    d["pickup_is_dhaka"] = d["pickup_district"].eq("Dhaka").astype(int)

    p = d["pickup_dt"]
    d["pickup_hour"] = p.dt.hour
    d["pickup_dow"] = p.dt.dayofweek
    d["pickup_month"] = p.dt.month
    d["pickup_year"] = p.dt.year
    d["pickup_doy"] = p.dt.dayofyear
    d["is_weekend"] = p.dt.dayofweek.isin([4, 5]).astype(int)
    d["is_night"] = p.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    d["is_early_morning"] = p.dt.hour.between(4, 7).astype(int)
    for col, period in [("pickup_hour", 24), ("pickup_dow", 7), ("pickup_month", 12)]:
        d[f"{col}_sin"] = np.sin(2 * np.pi * d[col] / period)
        d[f"{col}_cos"] = np.cos(2 * np.pi * d[col] / period)
    d["days_since_start"] = (p - p.min()).dt.total_seconds() / 86400

    lead = (p - d["created_dt"]).dt.total_seconds() / 3600
    d["lead_hours"] = lead.where(lead.between(0, 24 * 120))
    d["log_lead_hours"] = np.log1p(d["lead_hours"])
    d["is_immediate"] = (d["lead_hours"].fillna(99) < 2).astype(int)

    gap = (d["return_dt"] - p).dt.total_seconds() / 3600
    d["return_gap_hours"] = gap.where(gap.between(0, 24 * 60))
    d["return_gap_days"] = np.ceil(d["return_gap_hours"] / 24)
    d["is_multiday"] = (d["return_gap_hours"] > 20).astype(int)

    d["car_age"] = (d["pickup_year"] - d["car_model_year"]).where(lambda s: s.between(0, 45))
    d["log_driver_trips"] = np.log1p(d["driver_trips"])
    d["driver_is_new"] = (d["driver_trips"].fillna(0) < 3).astype(int)

    d["promo_amount"] = d["promo_amount_raw"].fillna(0)
    d["promo_discount"] = d["promo_discount_raw"].fillna(0)
    return d


class RouteRateEncoder:
    """Must stay importable under this name — the notebook pickled instances of it.

    Structure must match the v3 notebook exactly: district hierarchy, distance-band
    hierarchy, and a shrunk car-type rate multiplier.
    """

    def __init__(self, min_count=8, n_km_bands=10):
        self.min_count = min_count
        self.n_km_bands = n_km_bands

    def _bands(self, km):
        return pd.Series(np.digitize(km.to_numpy(dtype="float64"), self.km_edges_),
                         index=km.index).astype(str)

    def fit(self, d, y):
        rate = (pd.Series(np.asarray(y, dtype="float64"), index=d.index)
                / d["km"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        qs = np.linspace(0, 1, self.n_km_bands + 1)[1:-1]
        self.km_edges_ = np.unique(d["km"].quantile(qs).to_numpy())
        t = pd.DataFrame({
            "rate": rate,
            "ct": d.car_type.astype(str),
            "pd_": d.pickup_district.astype(str), "dd": d.dropoff_district.astype(str),
            "pdiv": d.pickup_division.astype(str), "ddiv": d.dropoff_division.astype(str),
            "band": self._bands(d["km"]),
        }).dropna(subset=["rate"])
        self.global_ = float(t["rate"].median())

        def agg(keys):
            g = t.groupby(keys)["rate"].agg(["median", "size"])
            return g[g["size"] >= self.min_count]

        self.lv1_ = agg(["pd_", "dd", "ct"])
        self.lv2_ = agg(["pd_", "dd"])
        self.lv3_ = agg(["pd_"])
        self.lv4_ = agg(["pdiv", "ddiv"])
        self.kb1_ = agg(["band", "ct"])
        self.kb2_ = agg(["band"])
        cm = t.groupby("ct")["rate"].agg(["median", "size"])
        w = cm["size"] / (cm["size"] + 20)
        self.car_mult_ = (w * (cm["median"] / self.global_) + (1 - w) * 1.0)
        self.volume_ = t.groupby("pd_").size()
        return self

    def transform(self, d):
        idx = d.index
        band = self._bands(d["km"])
        mk = lambda arrs: pd.MultiIndex.from_arrays(arrs)

        def look(tab, keys):
            return (pd.Series(tab["median"].reindex(keys).to_numpy(), index=idx),
                    pd.Series(tab["size"].reindex(keys).to_numpy(), index=idx))

        ct = d.car_type.astype(str)
        r1, n1 = look(self.lv1_, mk([d.pickup_district.astype(str),
                                     d.dropoff_district.astype(str), ct]))
        r2, n2 = look(self.lv2_, mk([d.pickup_district.astype(str),
                                     d.dropoff_district.astype(str)]))
        r3, n3 = look(self.lv3_, pd.Index(d.pickup_district.astype(str)))
        r4, n4 = look(self.lv4_, mk([d.pickup_division.astype(str),
                                     d.dropoff_division.astype(str)]))
        b1, m1 = look(self.kb1_, mk([band, ct]))
        b2, m2 = look(self.kb2_, pd.Index(band))

        route = r1.fillna(r2).fillna(r3).fillna(r4).fillna(self.global_)
        nobs = n1.fillna(n2).fillna(n3).fillna(n4).fillna(0)
        kmrate = b1.fillna(b2).fillna(self.global_)
        unit = d["km"].replace(0, np.nan)

        out = pd.DataFrame(index=idx)
        out["route_ppk"] = route.to_numpy()
        out["route_ppk_n"] = nobs.to_numpy()
        out["baseline_fare"] = (route * unit).to_numpy()
        out["log_baseline_fare"] = np.log1p(out.baseline_fare.clip(lower=0))
        out["kmband_ppk"] = kmrate.to_numpy()
        out["kmband_baseline_fare"] = (kmrate * unit).to_numpy()
        out["car_rate_mult"] = ct.map(self.car_mult_).fillna(1.0).to_numpy()
        blend = np.sqrt(out.baseline_fare.clip(lower=1) *
                        out.kmband_baseline_fare.clip(lower=1))
        out["blend_baseline_fare"] = blend
        out["log_blend_baseline_fare"] = np.log1p(blend)
        out["pickup_district_volume"] = (d.pickup_district.astype(str)
                                         .map(self.volume_).fillna(0).to_numpy())
        return out


TARGETS = {
    "logfare": dict(
        label="log1p(fare)",
        inv=lambda z, d: np.expm1(np.asarray(z, dtype="float64")),
    ),
    "logppk": dict(
        label="log(fare per km)",
        inv=lambda z, d: (np.exp(np.asarray(z, dtype="float64"))
                          * d["km"].to_numpy(dtype="float64")),
    ),
}

# joblib pickles the encoder by reference as "__main__.RouteRateEncoder" because the
# notebook defined it at top level. Alias it so unpickling resolves.
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "RouteRateEncoder", RouteRateEncoder)


# ------------------------------------------------------------------ the service


class UnsupportedTripType(ValueError):
    pass


class BidModel:
    """Routes a booking to its trip-type specialist and returns the band."""

    def __init__(self, artifact_path):
        self.path = Path(artifact_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.path}. Run the v3 notebook's "
                f"'Save the artifact' cell and copy the .joblib file next to app.py."
            )
        a = joblib.load(self.path)

        version = a.get("version")
        if version != 3:
            raise ValueError(
                f"This app expects a v3 artifact but {self.path.name} reports "
                f"version={version!r}. Re-run garibook_driver_bid_model_v3.ipynb and use "
                f"the garibook_bid_model_v3.joblib it produces."
            )

        self.cfg = a["cfg"]
        self.segments = a["segments"]
        self.trip_types = a["supported_trip_types"]
        self.car_types = [c for c in a.get("car_types", [])
                          if c and str(c).lower() not in ("nan", "none", "")]
        self.ac_levels = a.get("ac_levels") or ["AC", "Non-AC", "Dual-AC", "Unknown"]
        self.comparison = a.get("comparison", [])

        # rebuild every booster from its portable blob
        for tt, sg in self.segments.items():
            sg["prod_models"] = {k: restore(e) for k, e in sg["prod_models"].items()}
            sg["quantile_models"] = {
                t: (m if not isinstance(m, str) else lgb.Booster(model_str=m))
                for t, m in sg["quantile_models"].items()
            }

        needs_cat = any("catboost" in s["member_order"] for s in self.segments.values())
        if needs_cat and not HAS_CATBOOST:
            raise ImportError(
                "This model includes a CatBoost member but catboost is not installed. "
                "Run: python -m pip install catboost"
            )

    # -- metadata -----------------------------------------------------------
    def coverage_pct(self, seg=None):
        s = self.segments[seg] if seg else next(iter(self.segments.values()))
        return int(round(100 * (1 - float(s["band_alpha"]))))

    def segment_metrics(self):
        """Holdout scores for each segment's shipped ensemble."""
        out = {}
        for tt, s in self.segments.items():
            rows = s.get("scores", [])
            row = next((r for r in rows if r.get("model") == "ENSEMBLE"), None)
            if row is None and rows:
                row = min(rows, key=lambda r: r["MedAPE"])
            row = row or {}
            out[tt] = {
                "n_rows": s.get("n_rows"),
                "MedAPE": row.get("MedAPE"),
                "within_20pct": row.get("within_20pct"),
                "MAE": row.get("MAE"),
                "coverage": (None if s.get("coverage_conformal") is None
                             else round(100 * float(s["coverage_conformal"]), 1)),
                "band_width_pct": (None if s.get("band_width") is None
                                   else round(100 * float(s["band_width"]))),
                "n_members": len(s["member_order"]),
                "weights": {k: v for k, v in (s.get("weights") or {}).items() if v},
            }
        return out

    # -- prediction ---------------------------------------------------------
    def _prepare(self, seg, f):
        """Build this segment's design matrix, mapping unseen levels to the NA bucket.

        A category this segment never saw in training (a district it has no trips for,
        say) would otherwise silently become NaN. LightGBM tolerates that; CatBoost
        raises. Either way the model has no information for the value, so route it to
        the explicit "NA" level and report it to the caller.
        """
        if seg.get("route_encoder") is not None:
            f = f.copy()
            f[ROUTE_COLS] = seg["route_encoder"].transform(f).to_numpy()
        X = f.reindex(columns=seg["num_cols"]).astype("float64")
        unseen = {}
        for c in seg["cat_cols"]:
            dt = seg["cat_dtypes"][c]
            v = f[c].astype("string").fillna("NA")
            known = v.isin(list(dt.categories))
            if not known.all():
                unseen[c] = str(v[~known].iloc[0])
            X[c] = v.where(known, "NA").astype(dt)
        return f, X.reindex(columns=seg["fit_cols"]), unseen

    def _member_fare(self, seg, key, X, f):
        e = seg["prod_models"][key]
        m, kind = e["model"], e["kind"]
        if kind == "lgb":
            return np.clip(TARGETS[e["target"]]["inv"](m.predict(X), f), 1, None)
        if kind == "lgb_resid":
            anchor = np.log1p(f["blend_baseline_fare"].clip(lower=1))
            return np.clip(np.expm1(m.predict(X) + anchor), 1, None)
        if kind == "xgb":
            return np.clip(np.expm1(m.predict(X)), 1, None)
        if kind == "cat":
            Xc = X.assign(**{c: X[c].astype(str).fillna("NA")
                             for c in seg["cat_cols"]})
            return np.clip(np.expm1(m.predict(Pool(Xc, cat_features=e["cat_idx"]))), 1, None)
        raise ValueError(f"unknown member kind: {kind}")

    def quote(self, booking, round_to=50):
        f = add_features(standardize(pd.DataFrame([booking]),
                                     target_raw=self.cfg.get("TARGET_RAW", "Driver Fare")))

        tt = f["trip_type"].iloc[0]
        if tt not in self.segments:
            raise UnsupportedTripType(
                f"This model covers {', '.join(self.trip_types)} only. "
                f"It cannot price a {tt} booking.")

        km = f["km"].iloc[0]
        if pd.isna(km) or km <= 0:
            raise ValueError("Distance must be a positive number of kilometres.")

        seg = self.segments[tt]
        f, X, unseen = self._prepare(seg, f)

        members = {k: float(self._member_fare(seg, k, X, f)[0])
                   for k in seg["member_order"]}
        z = np.log1p(np.array([members[k] for k in seg["member_order"]]))
        rec = float(np.clip(np.expm1(float(z @ np.array(seg["stack_coef"]))
                                     + seg["stack_intercept"]), 1, None))

        q = float(seg["q_conformal"])
        lo = float(np.expm1(seg["quantile_models"]["lo"].predict(X)[0] - q))
        hi = float(np.expm1(seg["quantile_models"]["hi"].predict(X)[0] + q))
        lo, hi = min(max(lo, 1.0), rec), max(hi, rec)

        rnd = lambda v: int(round(v / round_to) * round_to)
        evidence = int(f.route_ppk_n.iloc[0])

        return {
            "segment": tt,
            "low": rnd(lo),
            "recommended": rnd(rec),
            "high": rnd(hi),
            "km": round(float(km), 1),
            "bdt_per_km": round(rec / float(km), 1),
            "route_rate_bdt_per_km": round(float(f.route_ppk.iloc[0]), 1),
            "kmband_rate_bdt_per_km": round(float(f.kmband_ppk.iloc[0]), 1),
            "rate_card_anchor": rnd(float(f.blend_baseline_fare.iloc[0])),
            "evidence_n": evidence,
            "evidence_label": _evidence_label(evidence),
            "coverage_pct": self.coverage_pct(tt),
            "members": {k: rnd(v) for k, v in members.items()},
            "weights": {k: v for k, v in (seg.get("weights") or {}).items() if v},
            "recipe": f"{len(members)}-model ensemble, {tt} specialist",
            "unseen_categories": unseen,
        }


def _evidence_label(n):
    if n >= 100:
        return "strong"
    if n >= 25:
        return "moderate"
    if n >= 8:
        return "thin"
    return "no comparable trips"
