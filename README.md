# Bid recommender — local web app

A small localhost site that wraps the model from `driver_bid_model_v3.ipynb`.
Pick a trip type and car type, search a pickup and dropoff, and it returns the bid band.

**Requires a v3 artifact.** v3 trains one model per trip type, and the app routes on trip
type internally.

## Setup

```bash
cd garibook_ui
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the artifact the notebook saves into this folder:

```bash
cp /path/to/garibook_bid_model_v3.joblib .
```

```

## Run

```bash
python app.py
```

Open <http://127.0.0.1:8000>.

## What each file does

| File | Role |
|---|---|
| `app.py` | FastAPI server: page, place search, road distance, quote endpoint |
| `model_service.py` | Loads the artifact, routes to the right trip-type model, returns a band. Contains a copy of the notebook's feature code |
| `bd_places.py` | Offline list of the 64 districts — powers search when the geocoder is down, and fills in missing districts |
| `static/index.html` | The whole frontend, one file |

## How a quote is built

1. You pick two places. The app geocodes them through OpenStreetMap Nominatim and reads
   the district and division out of the result.
2. Road distance comes from the public OSRM routing service. For a round trip it is
   doubled, because `Total Km` in the training data covers both legs.
3. That becomes a booking dict keyed by the **raw CSV column names**, which goes through
   the same `standardize → add_features → route encoder` path the notebook used for
   training. Same code, so training and serving cannot drift.
4. The booking is routed to its **trip-type specialist** — one-way and round-way are
   separate models with separate features, tuning and calibration.
5. That segment's six ensemble members each predict a fare; a non-negative ridge blend
   (weights learned on its calibration slice) combines them into the recommended figure.
   The band comes from its conformalised P10/P90 quantile models.

## Things worth knowing

**Two network services are involved.** Nominatim for place search, OSRM for road
distance. Both are free public endpoints with no key, and both are rate-limited — fine
for one person clicking around, not for production. If either is unreachable the app
degrades rather than failing: search falls back to the built-in district list, and
distance falls back to straight-line × 1.35. The result panel always tells you which
distance source was used.

**You can skip search entirely.** Paste coordinates straight into either location box as
`23.8103, 90.4125`. Missing districts get snapped to the nearest of the 64 district
centroids, so the route rate card still works.

**Only `one_way` and `round_way` are supported**

**Unseen categories are flagged.** If you pick a district or division that a segment has
no training data for, the app routes it to the model's unknown bucket and says so in the
result panel. The prediction then rests on distance and the wider rate card, which is a
much weaker basis — worth heeding.

**Boosters are stored in portable formats**, not pickled. LightGBM as text, XGBoost as
JSON, CatBoost as .cbm — each library's own version-stable export. That means the
library versions here don't have to match the versions that trained the model. Pickling
them (the obvious approach) breaks with `input stream corrupted` whenever Colab upgrades
XGBoost, which it does every few weeks.

**The trim affects what the band promises.** v3 trains on the middle 96% of ৳/km per
segment, so the 80% coverage figure applies to that population. On a genuinely unusual
booking the real coverage is lower.

**Watch the evidence count.** It is how many comparable historical trips backed the rate
card for that corridor. Under 8 means the model is extrapolating from division-level or
global averages, and the number deserves much less trust than the confident typography
suggests.
