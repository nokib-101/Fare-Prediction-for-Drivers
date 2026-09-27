"""
Bid fare recommender — local web app.

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:8000

Expects `bid_model_bundle.pkl` next to this file (see export_model.py).
Override with:  BID_MODEL=/path/to/bundle.pkl python app.py
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import bd_places
from model_service import BidModel, UnsupportedTripType

HERE = Path(__file__).parent
ARTIFACT = os.environ.get("BID_MODEL", str(HERE / "bid_model_bundle.pkl"))

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM = "https://router.project-osrm.org/route/v1/driving"
UA = {"User-Agent": "bid-recommender/1.0 (local development tool)"}

# Median road/straight-line ratio measured on the training data (one-way trips).
# Used only when OSRM is unreachable.
ROAD_FACTOR = 1.32

app = FastAPI(title="Bid fare recommender")
model = BidModel(ARTIFACT)


class Place(BaseModel):
    label: str = ""
    lat: float
    lon: float
    district: str | None = None
    division: str | None = None


class QuoteRequest(BaseModel):
    trip_type: str
    car_type: str
    pickup: Place
    dropoff: Place
    seats: int = 4
    pickup_datetime: str | None = None
    return_datetime: str | None = None
    car_year: int | None = None
    km_override: float | None = Field(default=None, gt=0)
    fuel_price: float | None = Field(default=None, gt=0)   # Tk/L override


async def _road_km(a: Place, b: Place):
    """Road distance in km, with a measured fallback if OSRM is unreachable."""
    url = f"{OSRM}/{a.lon},{a.lat};{b.lon},{b.lat}"
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=UA) as c:
            r = await c.get(url, params={"overview": "simplified", "geometries": "geojson"})
            r.raise_for_status()
            data = r.json()
        route = data["routes"][0]
        return (route["distance"] / 1000.0, "road network (OSRM)",
                [[lat, lon] for lon, lat in route["geometry"]["coordinates"]])
    except Exception:
        straight = bd_places.haversine_km(a.lat, a.lon, b.lat, b.lon)
        return (straight * ROAD_FACTOR,
                f"estimated ({ROAD_FACTOR}x straight line — routing unreachable)", None)


@app.get("/api/meta")
def meta():
    return {
        "trip_types": model.trip_types,
        "car_types": {tt: model.car_types(tt) for tt in model.trip_types},
        "seats_by_car": {tt: model.meta[tt]["seats_by_car"] for tt in model.trip_types},
        "segments": model.segment_metrics(),
        "coverage_pct": round(model.coverage * 100),
        "fuel": model.fuel_status(),
        "versions": model.versions,
        "bundle": Path(ARTIFACT).name,
        "default_pickup": (datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M"),
        "default_return": (datetime.now() + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M"),
    }


@app.get("/api/places")
async def places(q: str):
    q = q.strip()
    if len(q) < 3:
        return {"results": [], "source": "none"}
    params = {"q": q, "format": "jsonv2", "addressdetails": 1,
              "countrycodes": "bd", "limit": 6}
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=UA) as c:
            r = await c.get(NOMINATIM, params=params)
            r.raise_for_status()
            rows = r.json()
    except Exception:
        return {"results": bd_places.search(q), "source": "offline district list"}

    out = []
    for row in rows:
        lat, lon = float(row["lat"]), float(row["lon"])
        d, div, _ = bd_places.nearest_district(lat, lon)
        out.append({"label": row.get("display_name", "")[:110], "lat": lat, "lon": lon,
                    "district": d, "division": div})
    return {"results": out or bd_places.search(q),
            "source": "OpenStreetMap" if out else "offline district list"}


@app.post("/api/quote")
async def quote(req: QuoteRequest):
    if req.km_override:
        km, km_source, geometry = float(req.km_override), "entered manually", None
    else:
        km, km_source, geometry = await _road_km(req.pickup, req.dropoff)
        if req.trip_type == "round_way":
            # Total Km in the training data covers BOTH legs for round trips
            # (measured ratio to straight-line was 2.75x vs 1.32x one-way)
            km *= 2
            km_source += ", doubled for the return leg"

    snapped = []
    for place, name in ((req.pickup, "pickup"), (req.dropoff, "dropoff")):
        if not place.district:
            d, div, _ = bd_places.nearest_district(place.lat, place.lon)
            place.district, place.division = d, div
            snapped.append(f"{name} → {d}")

    pickup_dt = req.pickup_datetime or (
        datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")

    booking = {
        "trip_type": req.trip_type, "car_type": req.car_type, "seats": req.seats,
        "total_km": round(km, 2),
        "pickup_lat": req.pickup.lat, "pickup_long": req.pickup.lon,
        "dropoff_lat": req.dropoff.lat, "dropoff_long": req.dropoff.lon,
        "pickup_datetime": pickup_dt.replace("T", " "),
        "booking_datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "return_datetime": (req.return_datetime or "").replace("T", " ") or None,
        "car_year": req.car_year,
        "fuel_price": req.fuel_price,
    }

    try:
        result = model.quote(booking)
    except UnsupportedTripType as e:
        raise HTTPException(400, str(e))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))

    result.update(km_source=km_source, geometry=geometry, snapped=snapped,
                  pickup_datetime=pickup_dt,
                  pickup_district=req.pickup.district,
                  dropoff_district=req.dropoff.district,
                  straight_km=round(bd_places.haversine_km(
                      req.pickup.lat, req.pickup.lon,
                      req.dropoff.lat, req.dropoff.lon), 1))
    return result


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    print(f"\n  bundle : {Path(ARTIFACT).name}")
    v = model.versions or {}
    if v:
        print(f"  built  : Python {v.get('python_full', '?')}, scikit-learn "
              f"{v.get('scikit-learn', '?')}")
    fs = model.fuel_status()
    if fs:
        print(f"  fuel   : Tk {fs['current_price']:.1f}/L ({fs['grade']}) in force since "
              f"{fs['effective_from']}")
    for tt, m in model.segment_metrics().items():
        print(f"  {tt:10s}: {m['selected_model']:6s} | {m['n_train']:,} train rows "
              f"| SMAPE {m['SMAPE']}% | within20 {m['within20']}% | bias {m['bias']:+.0f}")
    print("\n  open http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
