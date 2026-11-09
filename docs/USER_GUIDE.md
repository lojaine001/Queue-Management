# IQMS — User Guide
### Intelligent Queue Management System

---

## What Is IQMS?

IQMS is an automatic queue monitoring system for the store. A camera placed at the entrance counts every person who enters, stores their data, and uses that history to predict how busy the queue will be over the next hour.

The live dashboard gives store staff a real-time view of:
- How many people are currently in the queue
- How long the estimated wait time is right now and in the next 45 minutes
- How many people have entered the store today
- A visual forecast of wait time for the next 30 minutes

No manual input is needed. The system runs automatically in the background.

---

## Opening the Dashboard

Open a web browser and go to:

```
http://localhost:8501
```

> If the page does not load, the dashboard may not be running. Contact your technical contact.

The dashboard refreshes automatically. You do not need to reload the page manually.

---

## The Live Dashboard — What Each Number Means

When you open the dashboard you will see five key metrics at the top of the screen.

---

### IN QUEUE NOW

The estimated number of people currently waiting in the checkout queue.

This is updated every 10 seconds from the live camera feed.

The small number below it (e.g. **-5 vs 1h ago**) shows the change compared to one hour ago:
- A negative number means the queue is shorter than it was an hour ago.
- A positive number means the queue has grown.

---

### EST. WAIT +15 MIN

The predicted wait time **15 minutes from now**, in minutes.

This tells you how long a customer joining the queue in 15 minutes would likely wait before being served.

---

### EST. WAIT +30 MIN

The predicted wait time **30 minutes from now**, in minutes.

---

### EST. WAIT +45 MIN

The predicted wait time **45 minutes from now**, in minutes.

If this value is high, it means a busy period is approaching and additional lanes may need to be opened.

---

### ENTRIES TODAY

The total number of people who have entered the store since midnight.

The small text below (e.g. **-147 this hr vs 1h ago**) shows whether foot traffic is increasing or decreasing compared to the previous hour.

---

## Queue Status Indicator

Each wait time prediction is colour-coded with a status:

| Status | Colour | What it means |
|--------|--------|---------------|
| **OK** | Green | Wait time under 5 minutes — queue is normal |
| **BUSY** | Orange | Wait time between 5 and 10 minutes — queue is building up |
| **ALERT** | Red | Wait time 10 minutes or more — queue requires attention |

**When you see ALERT:** consider opening an additional checkout lane. The system predicts this based on the current number of people in the store and historical patterns for this time of day.

---

## The Predicted Wait Time Chart

Below the key metrics is a chart titled **PREDICTED WAIT TIME — NEXT 30 MIN**.

This chart shows the forecast wait time minute by minute over the next half hour.

- The **blue line** is the predicted wait time.
- The **orange dotted line** labelled *Busy* marks the 5-minute threshold.
- The **red dotted line** labelled *Alert* marks the 10-minute threshold.

**How to read it:**
- If the blue line is flat and low — the queue is expected to stay calm.
- If the blue line is rising toward or above the orange or red line — a busy period is coming and it may be worth preparing an extra lane.

The forecast is recalculated automatically approximately every 10–15 minutes.

---

## The Training Data Tab

This tab shows the 30-day history of customer arrivals broken down into 3-minute intervals.

It is useful for:
- Seeing typical busy and quiet periods over the past month
- Understanding seasonal or weekly patterns
- Verifying that the system has been collecting data correctly

### Metrics shown at the top of this tab

| Metric | What it means |
|--------|---------------|
| **Training rows** | Total number of individual entry records used to train the prediction model |
| **Real data span** | How many days of real camera data the model has been trained on |
| **Wait @ +15 min** | Latest predicted wait time at the 15-minute horizon |
| **Wait @ +30 min** | Latest predicted wait time at the 30-minute horizon |
| **Bag rate (24h)** | Percentage of people who entered with a bag or backpack in the last 24 hours |

---

## Typical Daily Patterns

The system learns from historical data. Over time the predictions become more accurate as more real data is collected.

**What a normal day looks like on the chart:**
- Low activity in the early morning (before opening)
- Gradual increase from opening time
- Peaks around lunch and late afternoon
- Drop-off in the evening
- Zero activity during closed hours

If the chart shows unexpectedly high spikes (e.g. hundreds of entries in a single 3-minute window), this may indicate a data quality issue — contact your technical contact.

---

## What To Do In Different Situations

### The wait time is showing ALERT

1. Check the **IN QUEUE NOW** number — if it is high, the queue is actively building.
2. Consider opening an additional checkout lane.
3. Monitor the **PREDICTED WAIT TIME** chart — if the blue line is expected to drop soon, the peak may pass without intervention.

### Entries today looks unusually high

If the **ENTRIES TODAY** number seems much higher than a typical day, it may reflect a period when the system was running with a technical issue (see the known data issue note below). Contact your technical contact to verify.

### The dashboard shows "None" for a wait time

This means the prediction model did not have enough recent data to calculate a confident estimate for that horizon. It is not an error — the number will update on the next forecast cycle (approximately every 15 minutes).

### The dashboard is not loading

1. Check that you are connected to the store network.
2. Try refreshing the page.
3. If it still does not load, the dashboard service may need to be restarted — contact your technical contact.

### The numbers have not changed in a long time

The dashboard updates automatically. If the **Queue snapshot** timestamp at the top of the page is more than a few minutes old and not changing, the camera pipeline may have stopped. Contact your technical contact.

---

## What the System Does Not Do

- It does **not** identify individuals or store any personal images.
- It does **not** control the checkout lanes — opening or closing lanes is always a manual decision by store staff.
- It does **not** guarantee exact wait times — predictions are estimates based on historical patterns and current queue state.
- It does **not** track people after they enter the store — only the entrance zone is monitored.

---

## Known Data Note (May 2026)

Between **19 May 2026** and the evening of **20 May 2026**, a technical issue caused the system to count some entries twice. This means the **ENTRIES TODAY** total and historical counts for those two days are approximately double the real values. The issue has been identified and fixed. Historical data for those days will be corrected by the technical team.

Data collected before 19 May 2026 and after the evening of 20 May 2026 is accurate.

---

## Quick Reference Card

| What you see | What it means | What to do |
|---|---|---|
| Status: OK, wait < 5 min | Queue is normal | No action needed |
| Status: BUSY, wait 5–10 min | Queue is building | Monitor closely |
| Status: ALERT, wait ≥ 10 min | Queue needs attention | Consider opening another lane |
| "None" for wait time | Not enough data yet | Wait for next forecast cycle (~15 min) |
| Entries today very high | Possible data issue | Contact technical contact |
| Dashboard not loading | Service may be down | Contact technical contact |
| Snapshot timestamp not updating | Camera pipeline may have stopped | Contact technical contact |

---

## Contact

For any technical issues with the system, contact your technical team.

---

*Last updated: May 2026*