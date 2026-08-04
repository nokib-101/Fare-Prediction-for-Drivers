"""
Garibook bid recommender — local web app.

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:8000


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
from model_service import (
    DIVISION_MAP,
    DISTRICT_MAP,
    BidModel,
    UnsupportedTripType,
    haversine_km,
)

HERE = Path(__file__).parent
ARTIFACT = os.environ.get("GARIBOOK_ARTIFACT", str(HERE / "garibook_bid_model_v3.joblib"))

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM = "https://router.project-osrm.org/route/v1/driving"
UA = {"User-Agent": "garibook-bid-recommender/1.0 (local development tool)"}

# straight-line -> road distance fudge factor, used only when OSRM is unreachable
ROAD_FACTOR = 1.35

app = FastAPI(title="Garibook bid recommender")
model = BidModel(ARTIFACT)


# ------------------------------------------------------------------ schemas


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
    ac_status: str = "AC"
    pickup_datetime: str | None = None          # ISO, from <input type=datetime-local>
    return_datetime: str | None = None
    km_override: float | None = Field(default=None, gt=0)


# ------------------------------------------------------------------ helpers


def _title(v: str | None) -> str | None:
    if not v:
        return None
    key = v.strip().lower()
    return DISTRICT_MAP.get(key) or DIVISION_MAP.get(key) or v.strip().title()


def _address_to_admin(addr: dict) -> tuple[str | None, str | None]:
    """Pull (district, division) out of a Nominatim address block."""
    district = (addr.get("state_district") or addr.get("district")
                or addr.get("county") or addr.get("city") or addr.get("town"))
    division = addr.get("state") or addr.get("region")

    def strip_suffix(v, suffix):
        if not v:
            return None
        v = v.strip()
        if v.lower().endswith(suffix):
            v = v[: -len(suffix)].strip()
        return v or None

    # Nominatim sometimes puts "Dhaka Division" where a district is expected
    if district and district.lower().endswith(" division") and not division:
        district, division = None, district

    return (_title(strip_suffix(district, " district")),
            _title(strip_suffix(division, " division")))


async def _road_km(a: Place, b: Place) -> tuple[float, str, list | None]:
    """Road distance in km. Falls back to scaled great-circle if OSRM is unreachable."""
    url = f"{OSRM}/{a.lon},{a.lat};{b.lon},{b.lat}"
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=UA) as c:
            r = await c.get(url, params={"overview": "simplified",
                                         "geometries": "geojson"})
            r.raise_for_status()
            data = r.json()
        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]
        return (route["distance"] / 1000.0, "road network (OSRM)",
                [[lat, lon] for lon, lat in coords])
    except Exception:
        straight = float(haversine_km(a.lat, a.lon, b.lat, b.lon))
        return (straight * ROAD_FACTOR,
                f"estimated ({ROAD_FACTOR}x straight line — routing service unreachable)",
                None)


# ------------------------------------------------------------------- routes


@app.get("/api/meta")
def meta():
    segs = model.segment_metrics()
    return {
        "trip_types": model.trip_types,
        "car_types": model.car_types,
        "ac_levels": model.ac_levels,
        "coverage_pct": model.coverage_pct(),
        "segments": segs,
        "default_pickup": (datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M"),
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
        # offline / rate-limited: use the built-in district list instead of failing
        return {"results": bd_places.search(q), "source": "offline district list"}

    out = []
    for row in rows:
        lat, lon = float(row["lat"]), float(row["lon"])
        district, division = _address_to_admin(row.get("address", {}))
        if not district or not division:
            snap_d, snap_div, _ = bd_places.nearest_district(lat, lon)
            district = district or snap_d
            division = division or snap_div
        out.append({
            "label": row.get("display_name", "")[:110],
            "lat": lat, "lon": lon,
            "district": district, "division": division,
        })
    if not out:
        return {"results": bd_places.search(q), "source": "offline district list"}
    return {"results": out, "source": "OpenStreetMap"}


@app.post("/api/quote")
async def quote(req: QuoteRequest):
    if req.km_override:
        km, km_source, geometry = float(req.km_override), "entered manually", None
    else:
        km, km_source, geometry = await _road_km(req.pickup, req.dropoff)
        if req.trip_type == "round_way":
            # Total Km in training data covers both legs for round trips
            km *= 2
            km_source += ", doubled for the return leg"

    # the rate card is keyed on district; snap any missing one to the nearest centroid
    snapped = []
    for place, name in ((req.pickup, "pickup"), (req.dropoff, "dropoff")):
        if not place.district or not place.division:
            d, div, _ = bd_places.nearest_district(place.lat, place.lon)
            place.district = place.district or d
            place.division = place.division or div
            snapped.append(f"{name} -> {place.district}")

    pickup_dt = req.pickup_datetime or (datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
    booking = {
        "Trip Type": req.trip_type,
        "Car Type": req.car_type,
        "No Of Seat": f"{req.seats} Seats",
        "Total Km": round(km, 2),
        "Pickup Lat": req.pickup.lat, "Pickup Long": req.pickup.lon,
        "Dropoff Lat": req.dropoff.lat, "Dropoff Long": req.dropoff.lon,
        "Pickup District": req.pickup.district, "Pickup Division": req.pickup.division,
        "Dropoff District": req.dropoff.district, "Dropoff Division": req.dropoff.division,
        "Pickup Date Time": pickup_dt.replace("T", " "),
        "Created At": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Return Date Time": (req.return_datetime or "").replace("T", " ") or None,
        "Driver Car Ac Status": req.ac_status,
    }

    try:
        result = model.quote(booking)
    except UnsupportedTripType as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    result["km_source"] = km_source
    result["pickup_district"] = req.pickup.district
    result["dropoff_district"] = req.dropoff.district
    result["snapped"] = snapped
    result["geometry"] = geometry
    result["pickup_datetime"] = pickup_dt
    return result


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    print(f"\n  model      : {Path(ARTIFACT).name}  (v3, one model per trip type)")
    for tt, m in model.segment_metrics().items():
        med = m["MedAPE"]
        print(f"  {tt:11s}: {m['n_rows']:,} rows | {m['n_members']} members"
              + (f" | median error {med:.1f}%" if med is not None else ""))
    print("\n  open http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
