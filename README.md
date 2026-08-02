# IQMS — Intelligent Queue Management System

A system that watches checkout lines in a store and tells you how long the wait actually is — right now, and in the next hour. Built during my internship.

It uses cameras to see how many people are waiting, predicts how busy it's about to get, and gives staff a clear, simple view of what's happening so they can react before a line gets too long.

---

## Where it is now

![Current app]

<img width="777" height="573" alt="image" src="https://github.com/user-attachments/assets/535e1294-7fa3-4672-b960-bf13f01e73ef" />


A live view of every checkout lane, wait-time alerts, and a statistics page to look back at how busy the store was on any given day.

---

## Where it started

![Earlier version]

<img width="968" height="756" alt="Screenshot 2026-07-02 160916" src="https://github.com/user-attachments/assets/4b97d701-02db-4798-84b9-abe1b323b5f9" />

An earlier version of the same idea — the project has evolved a lot since this point.

---

## The operations dashboard

![Dashboard]

<img width="1492" height="758" alt="Screenshot 2026-05-29 152603" src="https://github.com/user-attachments/assets/c97c4cf5-a17f-4700-aea5-2f25311d9126" />

A separate monitoring tool used to run and tune the prediction model directly — queue state, forecasts at multiple time horizons, and live model controls.

---

## What it actually does

- Counts people as they walk in and check out, using cameras
- Predicts how long the wait will be, so staff can open another lane before it gets bad
- Sends an alert when the wait crosses a threshold
- Keeps a history so you can look back at any past day — how many customers, when it was busiest, who visited
