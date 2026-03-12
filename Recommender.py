"""
Rule-based appliance recommendations based on usage hours and model prediction.
"""

EFFICIENT_HOURS = {
    "AC / Heater":        6,
    "Water Heater":       1.5,
    "Washing Machine":    1,
    "Refrigerator":       24,
    "Dishwasher":         1,
    "Microwave":          1,
    "Lights":             6,
    "TV / Entertainment": 4,
    "Computer / Laptop":  6,
    "Electric Oven":      1,
}

TIPS = {
    "AC / Heater": [
        "🌡️ Set your thermostat to 24–26°C in summer and 18–20°C in winter to save up to 10% per degree.",
        "🪟 Use ceiling fans alongside AC — it lets you raise the thermostat by 2°C without losing comfort.",
        "🔧 Clean AC filters monthly — dirty filters make it work 15% harder.",
        "⏰ Use a smart timer to turn off AC 30 minutes before you leave home.",
    ],
    "Water Heater": [
        "🚿 Shorter showers (5–7 min) instead of 15+ min saves significant hot water energy.",
        "🌡️ Set your water heater to 50°C — anything higher wastes energy.",
        "⏰ Schedule the water heater to heat only during morning and evening peak use times.",
        "🛁 Fix dripping hot water taps — a slow drip wastes up to 300L of hot water per month.",
    ],
    "Washing Machine": [
        "🧺 Always run full loads — a half-full machine uses the same energy as a full one.",
        "❄️ Wash clothes in cold water — 90% of a washing machine's energy goes to heating water.",
        "☀️ Air-dry clothes instead of using a dryer when possible.",
        "🔄 Use eco/quick wash mode for lightly soiled clothes.",
    ],
    "Refrigerator": [
        "🌡️ Set fridge to 3–5°C and freezer to -18°C — colder than needed wastes energy.",
        "🚪 Don't leave the fridge door open — every 10 seconds open costs extra cooling energy.",
        "📦 Keep fridge well-stocked but not overcrowded — a full fridge retains cold better.",
        "🔧 Check door seals — a loose seal lets cold air escape constantly.",
    ],
    "Dishwasher": [
        "🍽️ Only run the dishwasher when fully loaded.",
        "♨️ Use the eco wash mode — it uses less water and lower temperatures.",
        "🌬️ Skip the heated dry cycle and let dishes air dry instead.",
        "⏰ Run dishwasher at night during off-peak hours.",
    ],
    "Microwave": [
        "✅ Microwaves are already very efficient — use them instead of the oven for small meals.",
        "⏰ Defrost food in the fridge overnight instead of using the microwave defrost cycle.",
        "🫙 Cover food when microwaving — it heats faster and uses less energy.",
        "🫙 Cover food when microwaving — it heats faster and uses less energy.",
    ],
    "Lights": [
        "💡 Switch to LED bulbs — they use 75% less energy than incandescent bulbs.",
        "🌞 Use natural daylight during daytime — open blinds and curtains.",
        "📱 Install smart bulbs or motion sensors to automatically turn off unused lights.",
        "🕯️ Use task lighting (desk lamp) instead of lighting a whole room.",
    ],
    "TV / Entertainment": [
        "📺 Enable auto-brightness on your TV — it adjusts backlight and reduces consumption.",
        "😴 Use sleep timers so the TV automatically turns off when you fall asleep.",
        "🔌 Unplug gaming consoles when not in use — they draw standby power.",
        "📡 Use streaming sticks instead of full gaming consoles for video.",
    ],
    "Computer / Laptop": [
        "💻 Use a laptop instead of a desktop — laptops use up to 80% less energy.",
        "😴 Enable sleep/hibernate mode after 10–15 minutes of inactivity.",
        "🌑 Use dark mode — on OLED screens it significantly reduces power draw.",
        "🔌 Unplug the charger once the laptop is fully charged.",
    ],
    "Electric Oven": [
        "🍳 Use a microwave or air fryer for small meals — they use far less energy.",
        "🚫 Avoid preheating the oven for longer than needed — 10 minutes is usually enough.",
        "🍲 Cook multiple dishes at once to maximise each oven session.",
        "🌡️ Don't open the oven door while cooking — each opening drops temperature by 15°C.",
    ],
}


def get_recommendations(usage_hours: dict, class_id: int) -> list:
    from train_model import APPLIANCE_WATTS, APPLIANCE_NAMES
    watts_map = dict(zip(APPLIANCE_NAMES, APPLIANCE_WATTS))
    results   = []

    for appliance, hours in usage_hours.items():
        threshold = EFFICIENT_HOURS.get(appliance, 6)
        watts     = watts_map.get(appliance, 500)
        daily_kwh = round(hours * watts / 1000, 3)
        over      = hours > threshold
        tips      = TIPS.get(appliance, ["Consider reducing usage time."])
        tip       = tips[min(class_id, len(tips) - 1)]

        results.append({
            "appliance": appliance,
            "hours":     round(hours, 1),
            "daily_kwh": daily_kwh,
            "threshold": threshold,
            "over":      over,
            "tip":       tip,
            "watts":     watts,
        })

    results.sort(key=lambda x: (-int(x["over"]), -x["daily_kwh"]))
    return results


def savings_estimate(usage_hours: dict, class_id: int, cost_per_kwh: float = 0.12) -> dict:
    from train_model import APPLIANCE_WATTS, APPLIANCE_NAMES
    watts_map = dict(zip(APPLIANCE_NAMES, APPLIANCE_WATTS))

    current_kwh   = 0.0
    optimised_kwh = 0.0

    for appliance, hours in usage_hours.items():
        watts     = watts_map.get(appliance, 500)
        threshold = EFFICIENT_HOURS.get(appliance, 6)
        kwh       = hours * watts / 1000
        current_kwh += kwh
        if hours > threshold:
            optimised_kwh += min(hours, threshold) * watts / 1000
        else:
            optimised_kwh += kwh

    monthly_current   = round(current_kwh   * 30, 2)
    monthly_optimised = round(optimised_kwh * 30, 2)
    monthly_saving    = round((monthly_current - monthly_optimised) * cost_per_kwh, 2)
    pct_saving        = round((monthly_current - monthly_optimised) / max(monthly_current, 1) * 100, 1)

    return {
        "current_monthly_kwh":   monthly_current,
        "optimised_monthly_kwh": monthly_optimised,
        "monthly_saving_usd":    monthly_saving,
        "pct_saving":            pct_saving,
        "current_daily_kwh":     round(current_kwh, 3),
    }