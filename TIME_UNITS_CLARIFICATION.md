# Time and Price Units Clarification

## Paper Parameters

The Avellaneda-Stoikov paper (Section 3.3) states:
> "we chose the following parameters: s=100, T=1, σ=2, dt=0.005"

## Interpretation

The given implementation (https://github.com/julian-hilg/avellenda_stoikov) uses the paper's exact parameters with the following interpretation:

### Units Convention
- **Time**: DAILY units (T=1 means 1 trading day)
- **Price**: ABSOLUTE units (σ=2 means ±$2 price moves per day)

### Parameter Values
| Parameter | Paper Value | Interpretation |
|-----------|------------|----------------|
| s | 100 | Initial price = $100 |
| T | 1 | Time horizon = 1 trading day |
| σ | 2 | Volatility = $2 daily price movement |
| dt | 0.005 | Time step = 0.005 days ≈ 7.2 minutes |
| γ | 0.1 | Risk aversion coefficient |
| k | 1.5 | Order arrival decay |
| A | 140 | Base arrival rate (orders/day) |

## Why Absolute Price Units?

### With σ=2 in absolute units:
- **Reservation price clearly adjusts for inventory**
  - With q=5: r = 100 - 5×0.1×4×1 = 98 (2% below mid)
  - With q=-5: r = 100 + 2 = 102 (2% above mid)
- **Spread narrows over time** from ~1.69 to ~1.29 as T approaches
- **Results match paper's figures**

### If σ=0.02 (percentage):
- Reservation price ≈ mid-price (adjustment only $0.002)
- No visible inventory effect
- Doesn't match paper's behaviour

## Implications

For a $100 stock:
- σ=2 implies 2% daily volatility
- This is ~32% annual volatility (reasonable for liquid stocks)
- Daily price range: typically $96-$104