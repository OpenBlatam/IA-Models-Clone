# Price Tracker

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

Automated system for tracking and analyzing product prices on various e-commerce platforms.

## 🚀 Features

- **Multi-Platform Tracking**: Supports major e-commerce sites (Amazon, MercadoLibre, etc.)
- **Price History**: Historical price data visualization
- **Deal Alerts**: Automatic notifications for price drops
- **Competitor Analysis**: Monitor competitor pricing strategies

## 📁 Structure

```
price_tracker/
├── scrapers/              # Platform-specific scrapers
├── analysis/              # Price analysis logic
├── alerts/                # Notification system
└── database/              # Price history storage
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

```python
from price_tracker.core import Tracker

# Initialize tracker
tracker = Tracker()

# Track product
tracker.track_product(
    url="https://example.com/product",
    target_price=100.00
)
```

---

[← Back to Main README](../README.md)
