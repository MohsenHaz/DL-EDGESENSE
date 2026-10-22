# Hardware set-up (fill in before release)
- Boards: Raspberry Pi 4B (4 GB), Jetson Nano; power supply, cable and shunt placement (high-side), shunt value, INA219 address, I2C bus speed.
- INA219 configuration (ADC bits, averaging) and the **measured effective sampling rate**: the INA219 conversion time plus I2C transaction time limit the rate;
  record it from the trace header and state it in the paper instead of the nominal 10 kHz if it is lower.
- GPIO marker pin and wiring to the logger.
- Thermal protocol: 5 min stabilisation before each session, 60 s warm-up, fixed CPU governor/clock.
- Ambient temperature and supply voltage per session.
