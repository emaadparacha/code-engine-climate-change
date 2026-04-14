"""
IBM Cloud Code Engine Demo – Climate Impact Simulator
======================================================
Everyone picks a Canadian province. Code Engine spins up 16 parallel
workers (4 cities × 4 temperature deltas) and texts back what each
city looks like over the next 20 years under that temperature shift.

Environment variables:
  PROVINCE        – one of: Ontario, Manitoba, Nova Scotia, Quebec,
                    Alberta, British Columbia
  TEMP_DELTAS     – comma-separated temp changes, e.g. "+2,+3,-1,-5"
  TWILIO_TOKEN
  PHONE_NUMBER
  JOB_INDEX       – set automatically by Code Engine (0-15)
"""

import os
import numpy as np
from twilio.rest import Client


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000
YEARS = 20

# ── City Data ────────────────────────────────────────────────────────
# Each city: population, annual avg temp (°C), latitude,
#            baseline heating degree days (HDD), baseline cooling degree days (CDD),
#            avg annual precipitation (mm), baseline frost-free days

PROVINCES = {
    "Ontario": [
        {
            "name": "Toronto",
            "population": 2_794_356,
            "avg_temp": 9.4,
            "latitude": 43.65,
            "hdd": 3520,
            "cdd": 350,
            "precip_mm": 831,
            "frost_free_days": 203,
        },
        {
            "name": "Ottawa",
            "population": 1_017_449,
            "avg_temp": 6.6,
            "latitude": 45.42,
            "hdd": 4440,
            "cdd": 250,
            "precip_mm": 943,
            "frost_free_days": 170,
        },
        {
            "name": "Thunder Bay",
            "population": 108_843,
            "avg_temp": 2.6,
            "latitude": 48.38,
            "hdd": 5650,
            "cdd": 80,
            "precip_mm": 712,
            "frost_free_days": 120,
        },
        {
            "name": "Windsor",
            "population": 229_660,
            "avg_temp": 9.9,
            "latitude": 42.30,
            "hdd": 3400,
            "cdd": 420,
            "precip_mm": 918,
            "frost_free_days": 210,
        },
    ],
    "Manitoba": [
        {
            "name": "Winnipeg",
            "population": 749_607,
            "avg_temp": 3.0,
            "latitude": 49.90,
            "hdd": 5670,
            "cdd": 170,
            "precip_mm": 521,
            "frost_free_days": 122,
        },
        {
            "name": "Brandon",
            "population": 51_313,
            "avg_temp": 2.3,
            "latitude": 49.84,
            "hdd": 5900,
            "cdd": 130,
            "precip_mm": 474,
            "frost_free_days": 118,
        },
        {
            "name": "Thompson",
            "population": 13_678,
            "avg_temp": -3.3,
            "latitude": 55.74,
            "hdd": 7800,
            "cdd": 30,
            "precip_mm": 517,
            "frost_free_days": 85,
        },
        {
            "name": "Churchill",
            "population": 899,
            "avg_temp": -6.5,
            "latitude": 58.77,
            "hdd": 9200,
            "cdd": 5,
            "precip_mm": 432,
            "frost_free_days": 60,
        },
    ],
    "Nova Scotia": [
        {
            "name": "Halifax",
            "population": 439_819,
            "avg_temp": 7.5,
            "latitude": 44.65,
            "hdd": 4000,
            "cdd": 150,
            "precip_mm": 1508,
            "frost_free_days": 175,
        },
        {
            "name": "Sydney",
            "population": 29_904,
            "avg_temp": 6.2,
            "latitude": 46.14,
            "hdd": 4500,
            "cdd": 80,
            "precip_mm": 1530,
            "frost_free_days": 160,
        },
        {
            "name": "Truro",
            "population": 12_826,
            "avg_temp": 6.3,
            "latitude": 45.36,
            "hdd": 4400,
            "cdd": 100,
            "precip_mm": 1180,
            "frost_free_days": 155,
        },
        {
            "name": "Yarmouth",
            "population": 6_772,
            "avg_temp": 7.0,
            "latitude": 43.84,
            "hdd": 4100,
            "cdd": 60,
            "precip_mm": 1340,
            "frost_free_days": 170,
        },
    ],
    "Quebec": [
        {
            "name": "Montreal",
            "population": 1_762_949,
            "avg_temp": 6.8,
            "latitude": 45.50,
            "hdd": 4300,
            "cdd": 280,
            "precip_mm": 1000,
            "frost_free_days": 178,
        },
        {
            "name": "Quebec City",
            "population": 549_459,
            "avg_temp": 4.9,
            "latitude": 46.81,
            "hdd": 5000,
            "cdd": 160,
            "precip_mm": 1190,
            "frost_free_days": 150,
        },
        {
            "name": "Gatineau",
            "population": 291_000,
            "avg_temp": 6.4,
            "latitude": 45.48,
            "hdd": 4500,
            "cdd": 240,
            "precip_mm": 920,
            "frost_free_days": 168,
        },
        {
            "name": "Saguenay",
            "population": 148_050,
            "avg_temp": 2.8,
            "latitude": 48.43,
            "hdd": 5600,
            "cdd": 100,
            "precip_mm": 960,
            "frost_free_days": 130,
        },
    ],
    "Alberta": [
        {
            "name": "Calgary",
            "population": 1_306_784,
            "avg_temp": 4.4,
            "latitude": 51.05,
            "hdd": 5100,
            "cdd": 80,
            "precip_mm": 419,
            "frost_free_days": 114,
        },
        {
            "name": "Edmonton",
            "population": 1_010_899,
            "avg_temp": 2.9,
            "latitude": 53.55,
            "hdd": 5600,
            "cdd": 100,
            "precip_mm": 477,
            "frost_free_days": 108,
        },
        {
            "name": "Red Deer",
            "population": 100_844,
            "avg_temp": 3.3,
            "latitude": 52.27,
            "hdd": 5400,
            "cdd": 70,
            "precip_mm": 486,
            "frost_free_days": 105,
        },
        {
            "name": "Lethbridge",
            "population": 101_482,
            "avg_temp": 5.8,
            "latitude": 49.69,
            "hdd": 4700,
            "cdd": 110,
            "precip_mm": 386,
            "frost_free_days": 118,
        },
    ],
    "British Columbia": [
        {
            "name": "Vancouver",
            "population": 662_248,
            "avg_temp": 10.4,
            "latitude": 49.28,
            "hdd": 2900,
            "cdd": 120,
            "precip_mm": 1189,
            "frost_free_days": 222,
        },
        {
            "name": "Victoria",
            "population": 91_867,
            "avg_temp": 10.2,
            "latitude": 48.43,
            "hdd": 2900,
            "cdd": 50,
            "precip_mm": 608,
            "frost_free_days": 240,
        },
        {
            "name": "Kelowna",
            "population": 144_576,
            "avg_temp": 8.8,
            "latitude": 49.89,
            "hdd": 3600,
            "cdd": 200,
            "precip_mm": 380,
            "frost_free_days": 165,
        },
        {
            "name": "Prince George",
            "population": 74_003,
            "avg_temp": 4.1,
            "latitude": 53.92,
            "hdd": 5200,
            "cdd": 40,
            "precip_mm": 617,
            "frost_free_days": 95,
        },
    ],
}


# ── Simulation Engine ────────────────────────────────────────────────

def simulate_climate_impact(city: dict, temp_delta: float) -> dict:
    """
    Monte Carlo simulation of a city's climate impact over 20 years
    given a temperature shift (°C).

    Models:
      1. Energy demand  – change in heating/cooling degree days → cost
      2. Health impact   – excess heat/cold-related hospitalizations
      3. Growing season  – frost-free day shift
      4. Water stress    – precipitation & drought probability
      5. Infrastructure  – freeze-thaw cycle damage (cold climates)
    """
    pop = city["population"]
    base_temp = city["avg_temp"]
    new_temp = base_temp + temp_delta

    # ── 1. Energy Demand ──────────────────────────────────────────
    # Each +1°C reduces HDD by ~250-350 and increases CDD by ~100-200
    # Add Monte Carlo noise for year-to-year weather variability
    hdd_shift_per_deg = np.random.normal(300, 40, NUM_SIMULATIONS) * (-temp_delta)
    cdd_shift_per_deg = np.random.normal(150, 30, NUM_SIMULATIONS) * temp_delta

    new_hdd = city["hdd"] + hdd_shift_per_deg
    new_cdd = city["cdd"] + cdd_shift_per_deg
    new_hdd = np.clip(new_hdd, 0, None)
    new_cdd = np.clip(new_cdd, 0, None)

    heating_cost_change_pct = ((new_hdd - city["hdd"]) / city["hdd"]) * 100
    if city["cdd"] > 20:
        cooling_cost_change_pct = ((new_cdd - city["cdd"]) / city["cdd"]) * 100
    else:
        cooling_cost_change_pct = new_cdd * np.random.normal(5, 1, NUM_SIMULATIONS)

    # Average household energy cost (CAD/year)
    avg_heating_cost = 2200  # baseline per household
    avg_cooling_cost = 600
    households = pop / 2.5   # avg household size

    annual_heating_delta = np.median(heating_cost_change_pct) / 100 * avg_heating_cost
    annual_cooling_delta = np.median(cooling_cost_change_pct) / 100 * avg_cooling_cost
    total_energy_delta_per_household = annual_heating_delta + annual_cooling_delta
    city_energy_impact_20yr = total_energy_delta_per_household * households * YEARS

    # ── 2. Health Impact ──────────────────────────────────────────
    # Excess heat-related hospitalizations per 100k per +1°C above 25°C
    # Cold-related deaths decrease with warming, increase with cooling
    # Base rates: ~12 heat-related per 100k per summer, ~18 cold-related per winter

    heat_rate_per_100k = 12.0
    cold_rate_per_100k = 18.0

    if temp_delta > 0:
        # More extreme heat days
        heat_increase = np.random.normal(
            temp_delta * 3.5, temp_delta * 1.2, NUM_SIMULATIONS
        )
        cold_decrease = np.random.normal(
            temp_delta * 2.0, temp_delta * 0.8, NUM_SIMULATIONS
        )
        heat_increase = np.clip(heat_increase, 0, None)
        cold_decrease = np.clip(cold_decrease, 0, None)
    else:
        heat_increase = np.zeros(NUM_SIMULATIONS)
        cold_decrease = np.zeros(NUM_SIMULATIONS)
        # More cold days
        cold_increase = np.random.normal(
            abs(temp_delta) * 2.5, abs(temp_delta) * 1.0, NUM_SIMULATIONS
        )
        cold_decrease = -np.clip(cold_increase, 0, None)

    net_health_change_per_100k = np.median(heat_increase) - np.median(cold_decrease)
    annual_excess_incidents = net_health_change_per_100k * (pop / 100_000)
    total_health_incidents_20yr = annual_excess_incidents * YEARS

    # ── 3. Growing Season ─────────────────────────────────────────
    # Roughly +7 to +10 frost-free days per +1°C
    frost_free_shift = np.random.normal(8.5 * temp_delta, 2.5, NUM_SIMULATIONS)
    new_frost_free = city["frost_free_days"] + frost_free_shift

    median_frost_free = float(np.median(new_frost_free))
    frost_free_change = median_frost_free - city["frost_free_days"]

    # What this means for agriculture
    if median_frost_free > 200:
        crop_note = "Long enough for corn, soybeans, grapes, and most warm-season crops."
    elif median_frost_free > 150:
        crop_note = "Supports wheat, canola, potatoes, and some hardier fruit trees."
    elif median_frost_free > 100:
        crop_note = "Limited to short-season crops: barley, oats, root vegetables."
    else:
        crop_note = "Extremely short. Only cold-hardy greens and root crops viable."

    # ── 4. Water Stress ───────────────────────────────────────────
    # Warmer = more evaporation, altered precip patterns
    # Model: precip changes are noisy; evaporation scales with temp
    precip_change_pct = np.random.normal(temp_delta * 2.0, 5.0, NUM_SIMULATIONS)
    evap_increase_pct = np.random.normal(temp_delta * 3.5, 2.0, NUM_SIMULATIONS)

    # Net water balance shift
    net_water_pct = precip_change_pct - evap_increase_pct
    median_water = float(np.median(net_water_pct))

    # Drought probability: baseline ~10%, shifts with water balance
    drought_prob_shift = -median_water * 1.5  # less water = more drought
    drought_prob = np.clip(10 + drought_prob_shift, 2, 80)

    if median_water > 5:
        water_note = "Wetter conditions. Flood risk increases."
    elif median_water > -5:
        water_note = "Water balance roughly stable."
    elif median_water > -15:
        water_note = "Drier conditions. Moderate water stress expected."
    else:
        water_note = "Significant water deficit. Drought risk is high."

    # ── 5. Infrastructure Stress ──────────────────────────────────
    # Freeze-thaw cycles damage roads, pipes, foundations
    # Warming in cold climates can INCREASE freeze-thaw near 0°C
    if -5 < new_temp < 5:
        freeze_thaw_note = "High freeze-thaw zone. Expect accelerated road and pipe damage."
        infra_risk = "High"
    elif new_temp >= 5:
        freeze_thaw_note = "Fewer freeze-thaw cycles. Less frost damage."
        infra_risk = "Low"
    else:
        freeze_thaw_note = "Stays frozen longer. Permafrost changes are the main risk."
        infra_risk = "Moderate"

    # ── Summary Score ─────────────────────────────────────────────
    # Simple livability impact score: -100 (catastrophic) to +100 (big improvement)
    score = 0
    score += np.clip(-total_energy_delta_per_household / 50, -20, 20)
    score += np.clip(-annual_excess_incidents / (pop / 100_000), -25, 25)
    score += np.clip(frost_free_change * 0.5, -15, 15)
    score += np.clip(median_water * 0.8, -20, 20)
    score += {"High": -10, "Moderate": -5, "Low": 5}[infra_risk]
    score = np.clip(score, -100, 100)

    return {
        "new_avg_temp": new_temp,
        # Energy
        "heating_cost_change_pct": float(np.median(heating_cost_change_pct)),
        "cooling_cost_change_pct": float(np.median(cooling_cost_change_pct)),
        "energy_delta_per_household": total_energy_delta_per_household,
        "city_energy_impact_20yr": city_energy_impact_20yr,
        # Health
        "annual_excess_incidents": annual_excess_incidents,
        "total_health_incidents_20yr": total_health_incidents_20yr,
        # Growing season
        "frost_free_change": frost_free_change,
        "new_frost_free_days": median_frost_free,
        "crop_note": crop_note,
        # Water
        "water_balance_pct": median_water,
        "drought_prob": drought_prob,
        "water_note": water_note,
        # Infrastructure
        "freeze_thaw_note": freeze_thaw_note,
        "infra_risk": infra_risk,
        # Overall
        "livability_score": float(score),
    }


# ── SMS Formatting ───────────────────────────────────────────────────

def format_sms(city: dict, temp_delta: float, results: dict) -> str:
    """Build the SMS body for one city × temperature scenario."""
    name = city["name"]
    pop = city["population"]
    delta_str = f"{temp_delta:+.1f}" if temp_delta != int(temp_delta) else f"{temp_delta:+.0f}"

    # Livability verdict
    score = results["livability_score"]
    if score > 15:
        verdict = f"Overall outlook: Net positive. Livability improves modestly."
    elif score > -5:
        verdict = f"Overall outlook: Mixed bag. Some gains, some losses. Adaptation needed."
    elif score > -25:
        verdict = f"Overall outlook: Net negative. Significant adaptation costs ahead."
    else:
        verdict = f"Overall outlook: Severe impact. Major infrastructure and health challenges."

    energy_dir = "savings" if results["energy_delta_per_household"] < 0 else "increase"
    energy_abs = abs(results["energy_delta_per_household"])

    health_dir = "additional" if results["annual_excess_incidents"] > 0 else "fewer"
    health_abs = abs(results["annual_excess_incidents"])

    frost_dir = "more" if results["frost_free_change"] > 0 else "fewer"
    frost_abs = abs(results["frost_free_change"])

    return (
        f"Simulation: {name} at {delta_str}C\n"
        f"----\n"
        f"Population: {pop:,}\n"
        f"New avg temp: {results['new_avg_temp']:.1f}C "
        f"(was {city['avg_temp']:.1f}C)\n"
        f"\n"
        f"-- Energy (20 yr) --\n"
        f"Heating costs: {results['heating_cost_change_pct']:+.0f}%\n"
        f"Cooling costs: {results['cooling_cost_change_pct']:+.0f}%\n"
        f"Per household: ${energy_abs:,.0f}/yr {energy_dir}\n"
        f"City-wide 20yr impact: ${abs(results['city_energy_impact_20yr']):,.0f}\n"
        f"\n"
        f"-- Health --\n"
        f"{health_abs:,.0f} {health_dir} heat/cold incidents per year\n"
        f"{abs(results['total_health_incidents_20yr']):,.0f} total over 20 years\n"
        f"\n"
        f"-- Growing Season --\n"
        f"{frost_abs:.0f} {frost_dir} frost-free days "
        f"({results['new_frost_free_days']:.0f} total)\n"
        f"{results['crop_note']}\n"
        f"\n"
        f"-- Water --\n"
        f"Water balance: {results['water_balance_pct']:+.1f}%\n"
        f"Drought probability: {results['drought_prob']:.0f}%\n"
        f"{results['water_note']}\n"
        f"\n"
        f"-- Infrastructure --\n"
        f"Freeze-thaw risk: {results['infra_risk']}\n"
        f"{results['freeze_thaw_note']}\n"
        f"\n"
        f"----\n"
        f"{verdict}\n"
        f"\n"
        f"[{NUM_SIMULATIONS:,} simulations | 20-year forecast | "
        f"Worker {int(os.getenv('JOB_INDEX', 0)) + 1} of 16]"
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    job_index = int(os.getenv("JOB_INDEX", "0"))

    # ── Parse province ───────────────────────────────────────────
    province = os.environ.get("PROVINCE", "Ontario").strip()
    if province not in PROVINCES:
        print(f"Unknown province: {province}")
        print(f"Available: {', '.join(PROVINCES.keys())}")
        return

    cities = PROVINCES[province]

    # ── Parse temperature deltas ─────────────────────────────────
    deltas_raw = os.environ.get("TEMP_DELTAS", "+2,+3,-1,-5")
    temp_deltas = [float(d.strip()) for d in deltas_raw.split(",") if d.strip()]

    if len(temp_deltas) != 4:
        print(f"Expected 4 temperature deltas, got {len(temp_deltas)}")
        return

    num_workers = len(cities) * len(temp_deltas)  # should be 16
    if job_index >= num_workers:
        print(f"Worker {job_index}: no assignment (only {num_workers} combos). Exiting.")
        return

    # ── Decode assignment ────────────────────────────────────────
    city_index = job_index // len(temp_deltas)
    temp_index = job_index % len(temp_deltas)

    city = cities[city_index]
    temp_delta = temp_deltas[temp_index]

    # ── Twilio setup ─────────────────────────────────────────────
    account_sid = "AC9c4c7cc858805c2d3661e38214d8e505"
    auth_token  = os.environ["TWILIO_TOKEN"]
    from_number = "+18674571410"
    to_number   = os.environ["PHONE_NUMBER"]
    twilio_client = Client(account_sid, auth_token)

    # ── Simulate ─────────────────────────────────────────────────
    print(f"Worker {job_index}: simulating {city['name']} at "
          f"{temp_delta:+.1f}C ({NUM_SIMULATIONS:,} sims, {YEARS} years)...")

    results = simulate_climate_impact(city, temp_delta)

    # ── Send SMS ─────────────────────────────────────────────────
    body = format_sms(city, temp_delta, results)
    message = twilio_client.messages.create(
        body=body,
        from_=from_number,
        to=to_number,
    )

    print(f"Worker {job_index} ({city['name']} @ {temp_delta:+.1f}C): "
          f"SMS sent, SID={message.sid}")
    print(body)


if __name__ == "__main__":
    main()
