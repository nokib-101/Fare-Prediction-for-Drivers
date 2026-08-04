"""
Approximate centroids for the 64 districts of Bangladesh.

Two jobs:

1.  **District snapping.** The route rate card is keyed on
    `trip_type x pickup_district x dropoff_district x car_type`. If a geocoded place
    comes back without an administrative district, the model falls all the way down
    to the division-pair or global rate, which is a much blunter anchor. Snapping the
    coordinates to the nearest district centroid recovers most of that precision.

2.  **Offline place search.** When the geocoding service is unreachable the app falls
    back to matching against this list, so the UI still works.

Centroids are approximate — good to a few kilometres, which is all that nearest-district
lookup needs. They are not used for distance calculation.
"""
from __future__ import annotations

import math

# (district, division, lat, lon)
DISTRICTS: list[tuple[str, str, float, float]] = [
    # Dhaka
    ("Dhaka", "Dhaka", 23.8103, 90.4125),
    ("Gazipur", "Dhaka", 24.0023, 90.4264),
    ("Narayanganj", "Dhaka", 23.6238, 90.5000),
    ("Narsingdi", "Dhaka", 23.9200, 90.7180),
    ("Munshiganj", "Dhaka", 23.5422, 90.5305),
    ("Manikganj", "Dhaka", 23.8617, 90.0003),
    ("Tangail", "Dhaka", 24.2513, 89.9167),
    ("Kishoreganj", "Dhaka", 24.4449, 90.7766),
    ("Faridpur", "Dhaka", 23.6070, 89.8429),
    ("Gopalganj", "Dhaka", 23.0050, 89.8266),
    ("Madaripur", "Dhaka", 23.1641, 90.1897),
    ("Shariatpur", "Dhaka", 23.2423, 90.4348),
    ("Rajbari", "Dhaka", 23.7574, 89.6444),
    # Chattogram
    ("Chattogram", "Chattogram", 22.3569, 91.7832),
    ("Cox's Bazar", "Chattogram", 21.4272, 92.0058),
    ("Cumilla", "Chattogram", 23.4607, 91.1809),
    ("Brahmanbaria", "Chattogram", 23.9571, 91.1115),
    ("Chandpur", "Chattogram", 23.2333, 90.6712),
    ("Feni", "Chattogram", 23.0159, 91.3976),
    ("Noakhali", "Chattogram", 22.8696, 91.0995),
    ("Lakshmipur", "Chattogram", 22.9447, 90.8282),
    ("Khagrachhari", "Chattogram", 23.1193, 91.9847),
    ("Rangamati", "Chattogram", 22.6533, 92.1751),
    ("Bandarban", "Chattogram", 22.1953, 92.2184),
    # Rajshahi
    ("Rajshahi", "Rajshahi", 24.3745, 88.6042),
    ("Bogura", "Rajshahi", 24.8465, 89.3773),
    ("Pabna", "Rajshahi", 24.0064, 89.2372),
    ("Sirajganj", "Rajshahi", 24.4534, 89.7007),
    ("Natore", "Rajshahi", 24.4206, 88.9414),
    ("Naogaon", "Rajshahi", 24.7936, 88.9318),
    ("Joypurhat", "Rajshahi", 25.0968, 89.0227),
    ("Chapainawabganj", "Rajshahi", 24.5965, 88.2775),
    # Khulna
    ("Khulna", "Khulna", 22.8456, 89.5403),
    ("Jashore", "Khulna", 23.1667, 89.2081),
    ("Satkhira", "Khulna", 22.7185, 89.0705),
    ("Bagerhat", "Khulna", 22.6516, 89.7859),
    ("Narail", "Khulna", 23.1725, 89.5125),
    ("Magura", "Khulna", 23.4855, 89.4198),
    ("Jhenaidah", "Khulna", 23.5450, 89.1726),
    ("Kushtia", "Khulna", 23.9013, 89.1206),
    ("Chuadanga", "Khulna", 23.6402, 88.8412),
    ("Meherpur", "Khulna", 23.7622, 88.6318),
    # Barishal
    ("Barishal", "Barishal", 22.7010, 90.3535),
    ("Patuakhali", "Barishal", 22.3596, 90.3299),
    ("Bhola", "Barishal", 22.6859, 90.6482),
    ("Pirojpur", "Barishal", 22.5841, 89.9720),
    ("Barguna", "Barishal", 22.0953, 90.1121),
    ("Jhalokati", "Barishal", 22.6406, 90.1987),
    # Sylhet
    ("Sylhet", "Sylhet", 24.8949, 91.8687),
    ("Moulvibazar", "Sylhet", 24.4829, 91.7774),
    ("Habiganj", "Sylhet", 24.3745, 91.4155),
    ("Sunamganj", "Sylhet", 25.0658, 91.3950),
    # Rangpur
    ("Rangpur", "Rangpur", 25.7439, 89.2752),
    ("Dinajpur", "Rangpur", 25.6217, 88.6354),
    ("Kurigram", "Rangpur", 25.8072, 89.6295),
    ("Gaibandha", "Rangpur", 25.3288, 89.5281),
    ("Nilphamari", "Rangpur", 25.9317, 88.8560),
    ("Panchagarh", "Rangpur", 26.3411, 88.5542),
    ("Thakurgaon", "Rangpur", 26.0337, 88.4616),
    ("Lalmonirhat", "Rangpur", 25.9923, 89.2847),
    # Mymensingh
    ("Mymensingh", "Mymensingh", 24.7471, 90.4203),
    ("Jamalpur", "Mymensingh", 24.9375, 89.9372),
    ("Netrokona", "Mymensingh", 24.8709, 90.7279),
    ("Sherpur", "Mymensingh", 25.0205, 90.0153),
]

# a few well-known places that are not district seats, so search feels less bare
LANDMARKS: list[tuple[str, str, str, float, float]] = [
    ("Hazrat Shahjalal International Airport", "Dhaka", "Dhaka", 23.8433, 90.3978),
    ("Uttara, Dhaka", "Dhaka", "Dhaka", 23.8759, 90.3795),
    ("Gulshan, Dhaka", "Dhaka", "Dhaka", 23.7925, 90.4078),
    ("Dhanmondi, Dhaka", "Dhaka", "Dhaka", 23.7461, 90.3742),
    ("Motijheel, Dhaka", "Dhaka", "Dhaka", 23.7330, 90.4172),
    ("Savar", "Dhaka", "Dhaka", 23.8583, 90.2667),
    ("Shah Amanat International Airport", "Chattogram", "Chattogram", 22.2496, 91.8133),
    ("Kuakata", "Patuakhali", "Barishal", 21.8172, 90.1195),
    ("Sreemangal", "Moulvibazar", "Sylhet", 24.3065, 91.7296),
    ("Sundarbans (Mongla)", "Bagerhat", "Khulna", 22.4900, 89.5900),
]


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def nearest_district(lat: float, lon: float) -> tuple[str, str, float]:
    """Return (district, division, km_to_centroid) for the closest district centroid."""
    best = min(DISTRICTS, key=lambda d: _haversine(lat, lon, d[2], d[3]))
    return best[0], best[1], round(_haversine(lat, lon, best[2], best[3]), 1)


def search(query: str, limit: int = 6) -> list[dict]:
    """Offline substring search over districts and landmarks."""
    q = query.strip().lower()
    if not q:
        return []
    hits: list[tuple[int, dict]] = []

    for name, div, lat, lon in DISTRICTS:
        low = name.lower()
        if q in low:
            hits.append((0 if low.startswith(q) else 1, {
                "label": f"{name}, {div} Division",
                "lat": lat, "lon": lon, "district": name, "division": div,
            }))

    for label, dist, div, lat, lon in LANDMARKS:
        low = label.lower()
        if q in low:
            hits.append((0 if low.startswith(q) else 1, {
                "label": f"{label} — {dist}, {div} Division",
                "lat": lat, "lon": lon, "district": dist, "division": div,
            }))

    hits.sort(key=lambda h: (h[0], h[1]["label"]))
    return [h[1] for h in hits[:limit]]
