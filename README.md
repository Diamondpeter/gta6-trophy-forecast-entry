# gta6-trophy-forecast-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Diamondpeter/gta6-trophy-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Diamondpeter/gta6-trophy-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy on PlayStation 5: from
Exophase's earn rates as the house recorded them, it writes five quantiles
(5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at 26 November 2026,
one week after launch — for the lowest pinball loss on the logit scale.

This repository is an entry in make.fun's bounty
[**"Forecast how many GTA VI players earn each trophy in week one"**](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250 in USDC, deadline Sat 21 November 2026, 12:30 Africa/Lagos).

## How it forecasts

The program reads one case (`/input/<id>.in`): every read the house made of
Exophase's GTA VI trophy page before the case's checkpoint, and writes
`/output/<id>.out` with five quantiles per trophy. It runs offline, with no
model calls, and is deterministic — two runs write identical bytes.

- **Counts, not just rates.** A rate is earned players over players tracked,
  and both move: new players joining Exophase dilute every rate, while the
  players already tracked keep earning. So the model fits the earned counts
  and the tracked-player count separately, each as a power law in log time
  since release, and divides them at the target.
- **The stale page.** Exophase refreshes a game's counts on its own schedule,
  hours apart, so a flat stretch of reads is a stale page, not a still game.
  When the history shows no change, a launch-dynamics prior stands in: the
  tracked base keeps growing, and newly tracked players have earned each
  trophy at roughly the square of its current rate (a trophy at 1% is held
  by about 0.01% of newcomers), so rates fall at the next refresh.
- **Honest spread.** The 5–95% band comes from the fit's uncertainty, the
  extrapolation distance and the refresh timing, so about 90% of rates
  should land inside it — the pinball loss pays for true quantiles.
- **Never invalid.** Every row is clamped to 0.01–99.99%, non-decreasing,
  one row per trophy, and the program always writes a complete forecast.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network.

## License

MIT — see [LICENSE](LICENSE).

---

Entered on [make.fun](https://make.fun/?ref=h68hpwev&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
