"""
Offline fallback: centroids of all 64 Bangladesh districts.

Used for two things: typeahead search when Nominatim is unreachable or
rate-limited, and snapping a dropped map pin to a district for display.
The model itself resolves districts from coordinates internally, so this is
convenience rather than a dependency.
"""
from __future__ import annotations
import math

DISTRICTS: dict[str, tuple[float, float, str]] = {
    # Barishal
    "Barguna": (22.16, 90.11, "Barishal"), "Barishal": (22.70, 90.35, "Barishal"),
    "Bhola": (22.69, 90.65, "Barishal"), "Jhalokati": (22.64, 90.20, "Barishal"),
    "Patuakhali": (22.36, 90.33, "Barishal"), "Pirojpur": (22.58, 89.98, "Barishal"),
    # Chattogram
    "Bandarban": (22.20, 92.22, "Chattogram"), "Brahmanbaria": (23.96, 91.11, "Chattogram"),
    "Chandpur": (23.23, 90.66, "Chattogram"), "Chattogram": (22.36, 91.83, "Chattogram"),
    "Cumilla": (23.46, 91.18, "Chattogram"), "Cox's Bazar": (21.44, 92.00, "Chattogram"),
    "Feni": (23.02, 91.40, "Chattogram"), "Khagrachhari": (23.10, 91.98, "Chattogram"),
    "Lakshmipur": (22.94, 90.83, "Chattogram"), "Noakhali": (22.87, 91.10, "Chattogram"),
    "Rangamati": (22.65, 92.17, "Chattogram"),
    # Dhaka
    "Dhaka": (23.81, 90.41, "Dhaka"), "Faridpur": (23.60, 89.83, "Dhaka"),
    "Gazipur": (23.99, 90.42, "Dhaka"), "Gopalganj": (23.01, 89.83, "Dhaka"),
    "Kishoreganj": (24.44, 90.78, "Dhaka"), "Madaripur": (23.16, 90.19, "Dhaka"),
    "Manikganj": (23.86, 90.00, "Dhaka"), "Munshiganj": (23.54, 90.53, "Dhaka"),
    "Narayanganj": (23.62, 90.50, "Dhaka"), "Narsingdi": (23.92, 90.72, "Dhaka"),
    "Rajbari": (23.76, 89.64, "Dhaka"), "Shariatpur": (23.20, 90.35, "Dhaka"),
    "Tangail": (24.25, 89.92, "Dhaka"),
    # Khulna
    "Bagerhat": (22.65, 89.79, "Khulna"), "Chuadanga": (23.64, 88.84, "Khulna"),
    "Jashore": (23.17, 89.21, "Khulna"), "Jhenaidah": (23.54, 89.17, "Khulna"),
    "Khulna": (22.81, 89.56, "Khulna"), "Kushtia": (23.90, 89.12, "Khulna"),
    "Magura": (23.49, 89.42, "Khulna"), "Meherpur": (23.76, 88.63, "Khulna"),
    "Narail": (23.17, 89.50, "Khulna"), "Satkhira": (22.71, 89.07, "Khulna"),
    # Mymensingh
    "Jamalpur": (24.92, 89.94, "Mymensingh"), "Mymensingh": (24.75, 90.40, "Mymensingh"),
    "Netrokona": (24.87, 90.73, "Mymensingh"), "Sherpur": (25.02, 90.02, "Mymensingh"),
    # Rajshahi
    "Bogura": (24.85, 89.37, "Rajshahi"), "Chapai Nawabganj": (24.60, 88.28, "Rajshahi"),
    "Joypurhat": (25.10, 89.02, "Rajshahi"), "Naogaon": (24.80, 88.94, "Rajshahi"),
    "Natore": (24.42, 89.00, "Rajshahi"), "Pabna": (24.00, 89.24, "Rajshahi"),
    "Rajshahi": (24.37, 88.60, "Rajshahi"), "Sirajganj": (24.45, 89.70, "Rajshahi"),
    # Rangpur
    "Dinajpur": (25.63, 88.64, "Rangpur"), "Gaibandha": (25.33, 89.53, "Rangpur"),
    "Kurigram": (25.81, 89.64, "Rangpur"), "Lalmonirhat": (25.92, 89.45, "Rangpur"),
    "Nilphamari": (25.93, 88.86, "Rangpur"), "Panchagarh": (26.33, 88.55, "Rangpur"),
    "Rangpur": (25.75, 89.24, "Rangpur"), "Thakurgaon": (26.03, 88.46, "Rangpur"),
    # Sylhet
    "Habiganj": (24.38, 91.42, "Sylhet"), "Moulvibazar": (24.48, 91.78, "Sylhet"),
    "Sunamganj": (25.07, 91.40, "Sylhet"), "Sylhet": (24.90, 91.87, "Sylhet"),
}
assert len(DISTRICTS) == 64, len(DISTRICTS)

# a few landmarks people actually type
LANDMARKS = {
    "Hazrat Shahjalal International Airport": (23.8433, 90.4004, "Dhaka", "Dhaka"),
    "Shah Amanat International Airport": (22.2496, 91.8133, "Chattogram", "Chattogram"),
    "Osmani International Airport": (24.9634, 91.8668, "Sylhet", "Sylhet"),
    "Cox's Bazar Beach": (21.4272, 92.0058, "Cox's Bazar", "Chattogram"),
    "Kuakata Beach": (21.8184, 90.1195, "Patuakhali", "Barishal"),
    "Sreemangal": (24.3065, 91.7296, "Moulvibazar", "Sylhet"),
    "Padma Bridge (Mawa)": (23.4180, 90.2620, "Munshiganj", "Dhaka"),
    "Sundarbans (Mongla)": (22.4900, 89.6000, "Bagerhat", "Khulna"),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def search(q: str, limit: int = 8) -> list[dict]:
    q = (q or "").strip().lower()
    if not q:
        return []
    out = []
    for name, (lat, lon, dist, div) in LANDMARKS.items():
        if q in name.lower():
            out.append({"label": name, "lat": lat, "lon": lon,
                        "district": dist, "division": div})
    for name, (lat, lon, div) in DISTRICTS.items():
        if q in name.lower():
            out.append({"label": f"{name}, {div} Division", "lat": lat, "lon": lon,
                        "district": name, "division": div})
    out.sort(key=lambda r: (not r["label"].lower().startswith(q), len(r["label"])))
    return out[:limit]


def nearest_district(lat: float, lon: float) -> tuple[str, str, float]:
    best, bd = None, 1e18
    for name, (dlat, dlon, div) in DISTRICTS.items():
        d = haversine_km(lat, lon, dlat, dlon)
        if d < bd:
            best, bd = (name, div), d
    return best[0], best[1], bd
