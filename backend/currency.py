"""
Currency: the catalog stores IKEA Japan prices (JPY) - filtering and budget checks stay in
JPY so they are exact - while everything shown to the user is in VND (decision 2026-10-15).

One fixed, assumed rate, defined here only. Edit VND_PER_JPY to change it everywhere.
"""

VND_PER_JPY = 170          # assumed fixed rate (1 JPY = 170 VND)
USD_IN_JPY = 150           # for requests that mention dollars

# multiplier from an amount in <currency> to JPY (used when parsing user budgets)
JPY_PER_UNIT = {"JPY": 1.0, "VND": 1 / VND_PER_JPY, "USD": float(USD_IN_JPY)}


def jpy_to_vnd(jpy) -> int:
    """JPY -> VND rounded to the nearest 1,000 đồng (how prices are usually shown)."""
    return int(round(jpy * VND_PER_JPY / 1000.0)) * 1000


def vnd_to_jpy(vnd) -> int:
    return int(round(vnd / VND_PER_JPY))


def fmt_vnd(vnd) -> str:
    """1650000 -> '1.650.000 ₫'"""
    return f"{int(vnd):,}".replace(",", ".") + " ₫"
