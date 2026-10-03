# Bid fare recommender — local app (fuel-aware)

Enter trip type, car type, seats, pickup and dropoff; get a suggested driver bid
with a low/high range, a confidence score, and the fuel effect applied.

## Fresh setup (Windows, VS Code)

### 1. Export from Colab
Run the notebook to the end. The export cell writes two files:

* `bid_model_bundle.pkl`
* `requirements.txt` — exact library versions, **Python version on line 2**

It also prints e.g. `your local venv must use Python 3.12`.



## Fuel price

The model predicts a fuel-neutral fare, then scales it by the pump price in
force on the booking date (Tk/L, average of octane and petrol). The fuel table
travels inside the bundle.

* **Prices changed?** Add a row to `FUEL_TABLE` in Stage B, re-run the export.
  No retraining needed if the session is still alive (just re-run Stage B's
  code cell, the predictor cells, and the export — or retrain fully later).
* **Before re-exporting**, type the new price into the app's *Fuel price*
  field. Blank = table price; a number = that price. Also useful for
  "what if fuel goes to Tk 180?".

