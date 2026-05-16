"""
IMC Prosperity Round 3 - Trading Algorithm
==========================================
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_4000..VEV_6500

Key findings from data analysis:
- VEVs are European call options on VELVETFRUIT_EXTRACT (VE)
- TTE = (8 - day - timestamp/10000) / 365  (annualised, 1 sol_day = 1/365 yr)
- Implied vol ~ 0.240 consistently across all strikes (flat smile)
- Realized vol > implied vol in game terms → market overprices time value
- SELL VEVs + delta hedge with VE is the core edge
- HYDROGEL_PACK mean-reverts around 10000, spread ~16 → market make inside it

Strategy:
1. HYDROGEL_PACK  – pure market making around 10000
2. VELVETFRUIT_EXTRACT – delta hedge VEV book + light market making
3. VEVs – sell overpriced time value, delta hedge, focus on VEV_5100–VEV_5400
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
import math


# Pure-math replacements for scipy.stats.norm.cdf and norm.pdf
# (scipy is not available in the competition Lambda environment)
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc – accurate to ~1e-15."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class _Norm:
    """Minimal drop-in for scipy.stats.norm (cdf and pdf only)."""
    @staticmethod
    def cdf(x: float) -> float:
        return _norm_cdf(x)

    @staticmethod
    def pdf(x: float) -> float:
        return _norm_pdf(x)


norm = _Norm()


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

# Annualised implied vol calibrated from cross-strike fit at Day 0
SIGMA_BASE = 0.240

# Position limits (adjust if competition specifies different values)
POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 75,
    "VELVETFRUIT_EXTRACT": 200,  # actual exchange limit (was incorrectly 350)
    "VEV_4000": 200,
    "VEV_4500": 200,
    "VEV_5000": 200,
    "VEV_5100": 200,
    "VEV_5200": 200,
    "VEV_5300": 200,
    "VEV_5400": 200,
    "VEV_5500": 200,
    "VEV_6000": 200,
    "VEV_6500": 200,
}

VEV_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

# Strikes where we have the most edge (biggest time value to sell)
PRIMARY_VEV_TARGETS = {"VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400"}
# Secondary – still profitable but smaller edge
SECONDARY_VEV_TARGETS = {"VEV_5000", "VEV_5500"}

HP_FAIR_VALUE = 10000
HP_MM_OFFSET = 4        # quote ±4 inside the ±8 market spread
HP_MAX_ORDER_SIZE = 8

VE_MM_OFFSET = 2        # quote ±2 inside the ±3 VE spread
VE_MAX_ORDER_SIZE = 15

# Minimum edge required before we sell a VEV (in SeaShells)
VEV_SELL_THRESHOLD = 0.5   # sell if bid > fair + threshold (tightened: market IV ~24% = fair)
# Opportunistic buy disabled: OTM options appear cheap vs BS fair due to
# vol-skew but the signal was spurious and caused large long positions.
# Set a very high threshold so it never fires.
VEV_BUY_THRESHOLD  = 999.0  # effectively disabled
VEV_MAX_ORDER_SIZE = 10

# Rolling vol window (number of VE mid-price observations)
VOL_WINDOW = 200


# ─────────────────────────────────────────────────────────────
#  Black-Scholes helpers  (no r, no dividends – game world)
# ─────────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    if T <= 1e-9:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Delta of a European call."""
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Vega (dC/d_sigma) – used for IV Newton step."""
    if T <= 1e-9:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return S * float(norm.pdf(d1)) * math.sqrt(T)


def implied_vol(C: float, S: float, K: float, T: float,
                lo: float = 0.01, hi: float = 3.0) -> Optional[float]:
    """Bisection implied vol. Returns None if option has no time value."""
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-3:
        return None
    if C >= S:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        price = bs_call(S, K, T, mid)
        if price > C:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def get_tte(day: int, timestamp: int) -> float:
    """
    Time-to-expiry in annualised years.
    TTE = 7 Solvenarian days at day=1, ts=0.
    We treat day 0 as TTE=8, day 1 as TTE=7, day 2 as TTE=6.
    Each game-day has 100_000 timestamps (NOT 10_000 — confirmed from replay data
    which shows timestamps running 0..99900 within a single day).
    """
    sol_days = (8 - day) - timestamp / 100_000
    return max(sol_days / 365.0, 1e-9)


# ─────────────────────────────────────────────────────────────
#  Utility: order book helpers
# ─────────────────────────────────────────────────────────────

def best_bid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders:
        return None
    return float(max(depth.buy_orders.keys()))


def best_ask(depth: OrderDepth) -> Optional[float]:
    if not depth.sell_orders:
        return None
    return float(min(depth.sell_orders.keys()))


def mid_price(depth: OrderDepth) -> Optional[float]:
    bb = best_bid(depth)
    ba = best_ask(depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def clamp_quantity(qty: int, pos: int, limit: int, side: str) -> int:
    """Clamp order quantity so we never exceed position limit."""
    if side == "buy":
        max_buy = limit - pos
        return min(qty, max_buy)
    else:
        max_sell = limit + pos
        return min(qty, max_sell)


# ─────────────────────────────────────────────────────────────
#  Main Trader class
# ─────────────────────────────────────────────────────────────

class Trader:
    def __init__(self):
        # State preserved across calls via trader_data JSON string
        # (re-initialised in run() if empty)
        self.ve_price_history: List[float] = []
        self.current_sigma: float = SIGMA_BASE

    # ── Load / save persistent state ────────────────────────

    def _load_state(self, trader_data: str) -> None:
        if not trader_data:
            self.ve_price_history = []
            self.current_sigma = SIGMA_BASE
            return
        try:
            state = json.loads(trader_data)
            self.ve_price_history = state.get("ve_history", [])
            self.current_sigma = state.get("sigma", SIGMA_BASE)
        except Exception:
            self.ve_price_history = []
            self.current_sigma = SIGMA_BASE

    def _save_state(self) -> str:
        # Keep only the last VOL_WINDOW prices to avoid bloat
        trimmed = self.ve_price_history[-VOL_WINDOW:]
        return json.dumps({
            "ve_history": trimmed,
            "sigma": self.current_sigma,
        })

    # ── Realized vol estimator ───────────────────────────────

    def _update_sigma(self, new_ve_price: float) -> None:
        """Update rolling realized vol from VE log-returns and clamp to [0.10, 0.60]."""
        self.ve_price_history.append(new_ve_price)
        if len(self.ve_price_history) < 20:
            return
        recent = self.ve_price_history[-VOL_WINDOW:]
        log_rets = [
            math.log(recent[i] / recent[i - 1])
            for i in range(1, len(recent))
            if recent[i - 1] > 0
        ]
        if len(log_rets) < 10:
            return
        # Per-timestamp std → annualise: multiply by sqrt(100_000 * 365)
        std_ts = float(np.std(log_rets))
        annualised = std_ts * math.sqrt(100_000 * 365)
        # Blend 80% rolling, 20% base to avoid wild swings
        blended = 0.80 * annualised + 0.20 * SIGMA_BASE
        self.current_sigma = max(0.10, min(0.60, blended))

    # ── Hydrogel Pack market making ──────────────────────────

    def _trade_hydrogel(
        self,
        depth: OrderDepth,
        position: int,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = POSITION_LIMITS["HYDROGEL_PACK"]

        # Inventory skew: if long, lower both quotes to offload; if short, raise
        skew = -position // 10  # gentle skew

        bid_price = HP_FAIR_VALUE - HP_MM_OFFSET + skew
        ask_price = HP_FAIR_VALUE + HP_MM_OFFSET + skew

        # Buy side
        buy_qty = clamp_quantity(HP_MAX_ORDER_SIZE, position, limit, "buy")
        if buy_qty > 0:
            orders.append(Order("HYDROGEL_PACK", bid_price, buy_qty))

        # Sell side
        sell_qty = clamp_quantity(HP_MAX_ORDER_SIZE, position, limit, "sell")
        if sell_qty > 0:
            orders.append(Order("HYDROGEL_PACK", ask_price, -sell_qty))

        # Opportunistic: hit existing bids below fair or lift asks above fair if
        # they give us a clear edge and help our inventory
        bb = best_bid(depth)
        ba = best_ask(depth)
        if ba is not None and ba < HP_FAIR_VALUE - HP_MM_OFFSET - 1 and position < limit:
            # Cheap ask – lift it
            ask_vol = abs(depth.sell_orders.get(int(ba), 0))
            lift_qty = clamp_quantity(min(ask_vol, HP_MAX_ORDER_SIZE), position, limit, "buy")
            if lift_qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(ba), lift_qty))

        if bb is not None and bb > HP_FAIR_VALUE + HP_MM_OFFSET + 1 and position > -limit:
            # Rich bid – hit it
            bid_vol = abs(depth.buy_orders.get(int(bb), 0))
            hit_qty = clamp_quantity(min(bid_vol, HP_MAX_ORDER_SIZE), position, limit, "sell")
            if hit_qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(bb), -hit_qty))

        return orders

    # ── VEV options book ─────────────────────────────────────

    def _trade_vevs(
        self,
        state: TradingState,
        ve_mid: float,
        T: float,
        sigma: float,
    ) -> Tuple[Dict[str, List[Order]], float]:
        """
        For each VEV:
          - Compute BS fair value
          - SELL if bid > fair + threshold (overpriced time value)
          - BUY  if ask < fair - threshold (opportunistic; rarer)
        Returns orders dict and the *net delta* of resulting VEV book.
        """
        all_orders: Dict[str, List[Order]] = {}
        net_delta = 0.0  # sum of delta * position across all VEVs

        for product, K in VEV_STRIKES.items():
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]
            fair = bs_call(ve_mid, K, T, sigma)
            delta = bs_delta(ve_mid, K, T, sigma)

            orders: List[Order] = []

            bb = best_bid(depth)
            ba = best_ask(depth)

            # ── SELL overpriced VEVs ──────────────────────────────────────
            if bb is not None and bb > fair + VEV_SELL_THRESHOLD:
                # How much can we sell?
                max_sell_qty = clamp_quantity(VEV_MAX_ORDER_SIZE, pos, limit, "sell")
                avail_sell = abs(depth.buy_orders.get(int(bb), 0))
                sell_qty = min(max_sell_qty, avail_sell)

                # Larger positions for the most overpriced strikes
                if product in PRIMARY_VEV_TARGETS:
                    sell_qty = min(sell_qty, VEV_MAX_ORDER_SIZE)
                elif product in SECONDARY_VEV_TARGETS:
                    sell_qty = min(sell_qty, VEV_MAX_ORDER_SIZE // 2)
                else:
                    sell_qty = min(sell_qty, 2)

                if sell_qty > 0:
                    orders.append(Order(product, int(bb), -sell_qty))

            # ── BUY underpriced VEVs (opportunistic) ─────────────────────
            # Only buy if the ask is dramatically below fair AND we are already
            # short (helping us cover). VEV_BUY_THRESHOLD is set very high by
            # default so this block almost never fires.
            if ba is not None and ba < fair - VEV_BUY_THRESHOLD and pos < 0:
                max_buy_qty = clamp_quantity(VEV_MAX_ORDER_SIZE, pos, limit, "buy")
                avail_buy = abs(depth.sell_orders.get(int(ba), 0))
                buy_qty = min(max_buy_qty, avail_buy)

                if product in PRIMARY_VEV_TARGETS:
                    buy_qty = min(buy_qty, VEV_MAX_ORDER_SIZE)
                elif product in SECONDARY_VEV_TARGETS:
                    buy_qty = min(buy_qty, VEV_MAX_ORDER_SIZE // 2)
                else:
                    buy_qty = min(buy_qty, 2)

                if buy_qty > 0:
                    orders.append(Order(product, int(ba), buy_qty))

            # ── Passive quotes for VEVs with enough spread ───────────────
            # Post passive SELL one tick below the market ask so we actually
            # get lifted, and strictly above our fair value.
            # Never post passive BUY unless we are already short and covering.
            if bb is not None and ba is not None:
                spread = ba - bb
                if spread >= 3:
                    # Passive ask: one tick below market ask, must be above fair
                    passive_ask = int(ba) - 1
                    if passive_ask > bb and passive_ask > fair:
                        pq = clamp_quantity(3, pos, limit, "sell")
                        if pq > 0:
                            orders.append(Order(product, passive_ask, -pq))

                    # Passive buy only to cover an existing short position
                    if pos < 0:
                        passive_bid = int(fair - 1.0)
                        if passive_bid >= bb and passive_bid < ba:
                            pq = clamp_quantity(3, pos, limit, "buy")
                            if pq > 0:
                                orders.append(Order(product, passive_bid, pq))

            all_orders[product] = orders

            # Accumulate net delta from CURRENT position (orders will shift it)
            # We compute delta on current pos; hedge will correct for order fills
            net_delta += pos * delta

        return all_orders, net_delta

    # ── VE market making + delta hedge ───────────────────────

    def _trade_ve(
        self,
        depth: OrderDepth,
        position: int,
        ve_mid: float,
        net_vev_delta: float,
    ) -> List[Order]:
        """
        Two objectives:
        1. Market make VE at ±2 around mid (capture half-spread)
        2. Delta hedge: target VE position = -net_vev_delta
           (short calls → positive delta → hold long VE to hedge)
        """
        orders: List[Order] = []
        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]

        bb = best_bid(depth)
        ba = best_ask(depth)
        if bb is None or ba is None:
            return orders

        # Target position for delta neutrality
        target_pos = round(-net_vev_delta)
        target_pos = max(-limit, min(limit, target_pos))

        delta_needed = target_pos - position

        # Execute hedge aggressively (cross spread if necessary)
        if delta_needed > 0 and position < limit:
            # Need to buy VE
            buy_qty = clamp_quantity(min(abs(delta_needed), VE_MAX_ORDER_SIZE), position, limit, "buy")
            if buy_qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(ba), buy_qty))

        elif delta_needed < 0 and position > -limit:
            # Need to sell VE
            sell_qty = clamp_quantity(min(abs(delta_needed), VE_MAX_ORDER_SIZE), position, limit, "sell")
            if sell_qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(bb), -sell_qty))

        # Passive market-making quotes around mid (inventory-skewed)
        skew = -(position - target_pos) // 20  # gentle skew toward target
        mm_bid = int(ve_mid) - VE_MM_OFFSET + skew
        mm_ask = int(ve_mid) + VE_MM_OFFSET + skew

        # Don't double-post if we already sent aggressive hedge orders
        # Just add passive quotes if we still have room
        if delta_needed >= 0:
            passive_buy = clamp_quantity(5, position, limit, "buy")
            if passive_buy > 0 and mm_bid < int(ba):
                orders.append(Order("VELVETFRUIT_EXTRACT", mm_bid, passive_buy))

        if delta_needed <= 0:
            passive_sell = clamp_quantity(5, position, limit, "sell")
            if passive_sell > 0 and mm_ask > int(bb):
                orders.append(Order("VELVETFRUIT_EXTRACT", mm_ask, -passive_sell))

        return orders

    # ── Main entry point ─────────────────────────────────────

    def run(self, state: TradingState):
        self._load_state(state.traderData)

        result: Dict[str, List[Order]] = {}

        # ── Determine time context ──────────────────────────────────────
        # state.timestamp is the global timestamp within the current day
        # state.timestamp goes 0..9999 each day; day number from state
        day = getattr(state, "day", 0)
        ts = state.timestamp
        T = get_tte(day, ts)

        # ── Get VE mid price ────────────────────────────────────────────
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid: Optional[float] = None
        if ve_depth:
            ve_mid = mid_price(ve_depth)

        if ve_mid is not None:
            self._update_sigma(ve_mid)

        sigma = self.current_sigma

        # ── 1. HYDROGEL_PACK ────────────────────────────────────────────
        hp_depth = state.order_depths.get("HYDROGEL_PACK")
        if hp_depth:
            hp_pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hydrogel(hp_depth, hp_pos)

        # ── 2. VEV options book ─────────────────────────────────────────
        if ve_mid is not None:
            vev_orders, net_vev_delta = self._trade_vevs(state, ve_mid, T, sigma)
            result.update(vev_orders)

            # ── 3. VELVETFRUIT_EXTRACT (hedge + MM) ─────────────────────
            if ve_depth:
                ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
                result["VELVETFRUIT_EXTRACT"] = self._trade_ve(
                    ve_depth, ve_pos, ve_mid, net_vev_delta
                )

        trader_data = self._save_state()
        conversions = 0
        return result, conversions, trader_data
