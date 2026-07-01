# Hardware — Raspberry Pi 4B Deployment Guide

## Target Hardware

- **Board:** Raspberry Pi 4B (4GB RAM recommended)
- **OS:** Raspberry Pi OS Lite (64-bit)
- **Python:** 3.10+
- **TFLite runtime:** tflite-runtime 2.15.0

## Sensor Integration (Optional)

For real-world data collection:

| Sensor | Pin | Function |
|--------|-----|----------|
| DHT11 | GPIO 4 | Temperature + Humidity |
| Soil Moisture | ADC (MCP3008) | Soil moisture % |
| Wind Speed | GPIO 17 | Anemometer pulse count |
| Solar Radiation | ADC (MCP3008) | Pyranometer analog |

## Wiring Diagram

```
Raspberry Pi 4B
┌─────────────────────────┐
│                         │
│  3.3V ──┬── DHT11 VCC  │
│         ├── Soil VCC    │
│         └── Solar VCC   │
│                         │
│  GPIO4 ──── DHT11 DATA  │
│                         │
│  SPI0 ──── MCP3008      │
│    (MOSI, MISO, SCLK, CE0) │
│                         │
│  GND ──┬── DHT11 GND   │
│        ├── Soil GND     │
│        └── Solar GND    │
│                         │
└─────────────────────────┘
```

## BOM (Bill of Materials)

| Component | Qty | Est. Cost |
|-----------|-----|-----------|
| Raspberry Pi 4B 4GB | 1 | $55 |
| DHT11 sensor | 1 | $2 |
| Soil moisture sensor | 1 | $3 |
| MCP3008 ADC | 1 | $4 |
| Jumper wires | 10 | $2 |
| Breadboard | 1 | $3 |
| **Total** | | **~$69** |

## Pi Deploy Script

```bash
#!/bin/bash
# pi_deploy.sh — deploy TFLite model on Raspberry Pi

# Install TFLite runtime
pip install tflite-runtime

# Copy model
scp runs/best_gru_seed42.tflite pi@raspberrypi:~/et0/

# Run inference
ssh pi@raspberrypi "python3 ~/et0/inference.py"
```

## Latency Expectations

| Platform | Latency (ms/sample) | Source |
|----------|-------------------|--------|
| Colab CPU | 0.028 | This reproduction |
| Pi 4B (paper) | 1.33 | Luo et al. 2025 |
| Pi 4B (estimated) | 1.5–2.0 | TFLite INT8 |
