"""
Thin wrapper around the cloudpickle bundle written by export_model.py.

One BidModel holds one FarePredictor per trip type. Nothing is refitted here;
this validates inputs, shapes the response, and scores confidence by looking up
empirical accuracy tables built at export time.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cloudpickle


class UnsupportedTripType(ValueError):
    pass


def _local(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


class BidModel:
    # Tier thresholds sit on the empirical hit-rate scale. Global within-20%
    # is ~89% for one-way and ~83% for round trips, so these describe how a
    # quote compares with what the model reliably achieves.
    CONF_TIERS = [(85, "High"), (75, "Moderate"), (60, "Low"), (0, "Very low")]

    # How much the marginal tables are allowed to move the cell estimate.
    # Car type carries independent signal (Sedan Economy really is worse).
    # Distance is damped harder because it partly overlaps with band width.
    DAMP_CAR, DAMP_KM = 0.6, 0.4

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"model bundle not found: {path}\n"
                "Run export_model.py at the end of the notebook and copy the "
                ".pkl next to app.py, or set BID_MODEL to its path.")
        try:
            with open(path, "rb") as f:
                b = cloudpickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"\nCould not load {path.name}: {type(e).__name__}: {e}\n\n"
                "This almost always means your local Python or scikit-learn "
                "version differs from the Colab session that exported the bundle.\n"
                f"Local: Python {sys.version_info.major}.{sys.version_info.minor}, "
                f"scikit-learn {_local('sklearn')}.\n"
                "Fix: use the requirements.txt produced by the Colab export, in a "
                "venv built with the Python version written at its top.") from e
        self.versions = b.get("versions", {})
        self._check_versions()
        self.fuel_table = b.get("fuel_table")
        self.fuel_grade_by_car = b.get("fuel_grade_by_car", {})
        self.default_fuel_grade = b.get("default_fuel_grade", "petrol_octane")
        self.predictors = b["predictors"]
        self.meta = b["meta"]
        self.coverage = b.get("coverage", 0.80)
        self.calibration = b.get("calibration", {})
        self.path = path
        if not self.calibration:
            print("WARNING: bundle has no calibration tables — confidence "
                  "scores will be unavailable. Re-export with the current "
                  "export_model.py to enable them.")

    # ------------------------------------------------------------- versions
    def _check_versions(self):
        """
        cloudpickle stores the pipeline's functions as compiled bytecode, which
        only runs on the SAME Python minor version. A mismatch loads fine and
        then crashes mid-prediction (`SystemError: no locals when deleting
        'dict'`), so fail loudly here instead.
        """
        v = self.versions
        if not v:
            print("note: bundle has no version stamp (older export) — skipping check")
            return
        local_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        if v.get("python") and v["python"] != local_py:
            raise RuntimeError(
                f"\nPython mismatch: bundle was built with Python {v['python']}, "
                f"this venv runs {local_py}.\n"
                f"Rebuild the venv:  py -{v['python']} -m venv .venv  "
                "then pip install -r requirements.txt")
        sk = _local("sklearn")
        if v.get("scikit-learn") and sk and v["scikit-learn"] != sk:
            print(f"WARNING: scikit-learn {sk} locally vs {v['scikit-learn']} in the "
                  "bundle. If quotes fail, install the exported requirements.txt.")

    # ----------------------------------------------------------------- fuel
    def fuel_price_on(self, car_type: str, when: datetime | None = None):
        """Pump price (Tk/L) from the bundled table for a car's fuel grade."""
        if not self.fuel_table:
            return None
        when = when or datetime.now()
        row = self.fuel_table[0]
        for r in self.fuel_table:
            if datetime.fromisoformat(str(r[0])) <= when:
                row = r
        diesel, octane, petrol, kerosene = row[1:5]
        grade = self.fuel_grade_by_car.get(car_type, self.default_fuel_grade)
        return {"petrol_octane": (octane + petrol) / 2, "diesel": diesel,
                "octane": octane, "petrol": petrol, "kerosene": kerosene}.get(
                    grade, (octane + petrol) / 2)

    def fuel_status(self) -> dict | None:
        if not self.fuel_table:
            return None
        last = self.fuel_table[-1]
        return {"effective_from": str(last[0]),
                "current_price": self.fuel_price_on("Sedan"),
                "grade": self.default_fuel_grade,
                "table_rows": len(self.fuel_table)}

    # ------------------------------------------------------------ vocabulary
    @property
    def trip_types(self) -> list[str]:
        return sorted(self.predictors)

    def car_types(self, trip_type: str | None = None) -> list[str]:
        if trip_type and trip_type in self.meta:
            return list(self.meta[trip_type]["car_types"])
        seen: set[str] = set()
        for m in self.meta.values():
            seen |= set(m["car_types"])
        return sorted(seen)

    def seats_for(self, trip_type: str, car_type: str) -> int:
        return int(self.meta.get(trip_type, {})
                   .get("seats_by_car", {}).get(car_type, 4))

    def segment_metrics(self) -> dict:
        out = {}
        for tt, m in self.meta.items():
            cal = self.calibration.get(tt, {})
            out[tt] = {
                "selected_model": m["selected_model"],
                "n_train": m["n_train"], "n_test": m["n_test"],
                "trained_to": m["trained_to"],
                "km_p99": round(m["km_p99"], 0),
                "MAE": round(m["metrics"]["MAE"], 0),
                "SMAPE": round(m["metrics"]["SMAPE%"], 2),
                "within10": round(m["metrics"]["within10%"], 1),
                "within20": round(m["metrics"]["within20%"], 1),
                "bias": round(m["metrics"]["bias"], 0),
                "band_coverage": round(m["interval"]["coverage%"], 1),
                "calibrated_on": cal.get("n_scored"),
                "trained_from": m.get("trained_from"),
                "fuel": m.get("fuel"),
                "support_from": cal.get("n_train_support"),
            }
        return out

    # ------------------------------------------------------------ confidence
    @staticmethod
    def _shrink(rate, n, prior, k):
        """Pull a thin cell's rate toward the global rate."""
        if rate is None or n is None:
            return prior
        return (n * rate + k * prior) / (n + k)

    @staticmethod
    def _bucket(value, buckets):
        for lo, hi, label in buckets:
            if lo <= value <= hi:
                return label
        return buckets[-1][2]

    def route_support(self, tt, pu, do, car):
        """
        How many TRAINING trips share this district-pair and car type.
        Read straight off the corridor encoder the model itself uses, so the
        number reflects the evidence behind the prediction.
        """
        try:
            tbl = self.predictors[tt].enc.maps_.get("enc_dist_pair_car")
            key = (pu, do, car)
            if tbl is None or key not in tbl.index:
                return 0.0
            return float(np.expm1(tbl.loc[key, "cnt"]))
        except Exception:
            return None

    def confidence(self, tt, car, km, low, typical, high, pu, do):
        """
        Expected probability that this quote lands within +/-20% of the fare
        actually paid, read from held-out accuracy tables. Not a model
        internal -- a measured frequency for trips that look like this one.
        """
        cal = self.calibration.get(tt)
        if not cal:
            return None

        g = cal["global_hit"]
        k = cal.get("shrink", 25)
        sup_b = [list(x) for x in cal["support_buckets"]]
        wid_b = [list(x) for x in cal["width_buckets"]]
        km_b = [list(x) for x in cal["km_bins"]]

        n_sup = self.route_support(tt, pu, do, car)
        rel = (high - low) / typical if typical else 1.0
        sb = self._bucket(n_sup if n_sup is not None else 0, sup_b)
        wb = self._bucket(rel, wid_b)
        kb = self._bucket(km, km_b)

        cell = cal["cell"].get(f"{sb}|{wb}")
        base = self._shrink(cell["hit"] if cell else None,
                            cell["n"] if cell else None, g, k)

        car_e = cal["by_car"].get(car)
        km_e = cal["by_km"].get(kb)
        adj_car = self.DAMP_CAR * (
            self._shrink(car_e["hit"] if car_e else None,
                         car_e["n"] if car_e else None, g, k) - g)
        adj_km = self.DAMP_KM * (
            self._shrink(km_e["hit"] if km_e else None,
                         km_e["n"] if km_e else None, g, k) - g)

        rate = base + adj_car + adj_km
        notes, penalties = [], []

        # Things the calibration tables cannot speak to, applied on top.
        p99 = self.meta.get(tt, {}).get("km_p99")
        if p99 and km > p99:
            factor = 0.75 if km > 1.5 * p99 else 0.85
            rate *= factor
            penalties.append(
                f"{km:.0f} km is past the 99th percentile of training distance "
                f"({p99:.0f} km), where too few held-out trips exist to measure "
                "accuracy — score reduced as a precaution")

        if typical and high <= typical * 1.001:
            rate *= 0.85
            penalties.append(
                "the point estimate hit its own upper bound, so the low-high "
                "range is not usable for this trip")

        score = int(round(max(5.0, min(97.0, rate * 100))))

        notes.append(
            f"{n_sup:.0f} comparable {car} trip(s) on {pu} to {do} in "
            f"{cal['n_train_support']:,} training trips"
            if n_sup is not None else "route support unavailable")
        notes.append(f"predicted range spans {rel * 100:.0f}% of the fare "
                     f"({wb} band)")
        if cell:
            notes.append(f"trips in this support/band group landed within "
                         f"\u00b120% {cell['hit'] * 100:.0f}% of the time "
                         f"({cell['n']:,} held-out trips)")
        if car_e:
            notes.append(f"{car} overall: {car_e['hit'] * 100:.0f}% within "
                         f"\u00b120% ({car_e['n']:,} trips)")

        return {
            "score": score,
            "label": next(l for t, l in self.CONF_TIERS if score >= t),
            "basis": (f"measured on {cal['n_scored']:,} held-out {tt} trips; "
                      f"route support from {cal['n_train_support']:,} training trips"),
            "route_support": None if n_sup is None else int(n_sup),
            "band_bucket": wb,
            "support_bucket": sb,
            "global_rate": round(g * 100),
            "notes": notes,
            "penalties": penalties,
        }

    # --------------------------------------------------------------- quoting
    def quote(self, booking: dict) -> dict:
        tt = str(booking.get("trip_type", "")).strip().lower()
        if tt not in self.predictors:
            raise UnsupportedTripType(
                f"No model for trip type '{tt}'. Trained types: "
                f"{', '.join(self.trip_types)}. Hourly and airport-rental trips "
                "were never in the training data and cannot be quoted.")

        car = str(booking.get("car_type", "")).strip()
        if car not in self.meta[tt]["car_types"]:
            raise ValueError(
                f"'{car}' was not seen for {tt} trips in training. "
                f"Available: {', '.join(self.meta[tt]['car_types'])}")

        km = float(booking.get("total_km") or 0)
        if km <= 0:
            raise ValueError("total_km must be greater than zero")
        if tt == "round_way" and not booking.get("return_datetime"):
            raise ValueError("round trips need a return date/time — trip "
                             "duration is one of the strongest features")

        req = {
            "car_type": car,
            "seats": int(booking.get("seats") or self.seats_for(tt, car)),
            "total_km": km,
            "pickup_lat": float(booking["pickup_lat"]),
            "pickup_long": float(booking["pickup_long"]),
            "dropoff_lat": float(booking["dropoff_lat"]),
            "dropoff_long": float(booking["dropoff_long"]),
            "pickup_datetime": booking["pickup_datetime"],
            "booking_datetime": booking["booking_datetime"],
        }
        if booking.get("return_datetime"):
            req["return_datetime"] = booking["return_datetime"]
        if booking.get("car_year"):
            req["car_year"] = float(booking["car_year"])
        if booking.get("fuel_price"):
            req["fuel_price"] = float(booking["fuel_price"])

        out = self.predictors[tt].predict(req)
        low, typical, high = out["low"], out["typical"], out["high"]

        # Resolve districts through the SAME resolver the model uses, so the
        # support lookup matches the features the prediction was built from.
        try:
            pu_d = self.predictors[tt].resolver.resolve(
                req["pickup_lat"], req["pickup_long"])[0]
            do_d = self.predictors[tt].resolver.resolve(
                req["dropoff_lat"], req["dropoff_long"])[0]
        except Exception:
            pu_d = do_d = "?"

        try:
            conf = self.confidence(tt, car, km, low, typical, high, pu_d, do_d)
        except Exception as e:          # never let scoring break a good quote
            print(f"confidence scoring failed: {e}")
            conf = None

        m = self.meta[tt]
        warnings = []
        fuel = None
        if out.get("fuel_price") is not None and out.get("fuel_reference"):
            fuel = {"price": out["fuel_price"], "reference": out["fuel_reference"],
                    "effect_pct": out["fuel_effect_pct"],
                    "overridden": bool(booking.get("fuel_price"))}
            seen = [self.fuel_price_on(car, datetime.fromisoformat(str(r[0])))
                    for r in (self.fuel_table or [])
                    if str(r[0]) <= m.get("validated_to", m.get("trained_to", "9999"))]
            top = max(seen) if seen else None
            if top and out["fuel_price"] > top * 1.001:
                warnings.append(
                    f"Fuel at Tk {out['fuel_price']:.0f}/L is above any price in the "
                    f"data the model was built and tested on (max Tk {top:.0f}). The fuel effect "
                    f"({out['fuel_effect_pct']:+.1f}%) comes from the cost-share "
                    "formula rather than observed fares, so check it against the "
                    "first few real bids at this price.")
        if km > m["km_p99"]:
            warnings.append(
                f"{km:.0f} km is beyond the 99th percentile of {tt} training "
                f"data ({m['km_p99']:.0f} km) — treat this quote as an "
                "extrapolation.")
        if high <= typical * 1.001:
            warnings.append(
                "The point estimate hit the top of its own predicted range, so "
                "the upper bound is not meaningful here. The typical figure is "
                "still usable; the band is not.")
        if m["metrics"]["bias"] < -150:
            warnings.append(
                f"{tt} predictions run about {abs(m['metrics']['bias']):.0f} BDT "
                "low on average in backtesting — consider it a floor, not a "
                "target.")

        return {
            "trip_type": tt, "car_type": car, "total_km": round(km, 1),
            "low": low, "typical": typical, "high": high,
            "per_km": round(typical / km, 1),
            "band_pct": round(100 * (high - low) / typical, 1) if typical else None,
            "coverage_target": round(self.coverage * 100),
            "model": m["selected_model"],
            "confidence": conf,
            "fuel": fuel,
            "accuracy": {"SMAPE": round(m["metrics"]["SMAPE%"], 2),
                         "within10": round(m["metrics"]["within10%"], 1),
                         "within20": round(m["metrics"]["within20%"], 1),
                         "MAE": round(m["metrics"]["MAE"], 0)},
            "warnings": warnings,
        }
