import { useEffect, useRef, useState, useCallback } from 'react';
import { HEADERS, REFRESH_MS } from '../config';

// Kept comfortably under REFRESH_MS (see config.js) so a hung request times
// out and clears isFetchingRef well before the next poll tick is due —
// otherwise a single slow/hung backend response would permanently block
// every future refresh (see fetchOne's AbortController below). If you lower
// REFRESH_MS further, lower this too — it must stay under it.
const FETCH_TIMEOUT_MS = 8000;

function fetchOne(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  return fetch(url, { headers: HEADERS, signal: controller.signal })
    .then(r => (r.ok ? r.json() : null))
    .catch(() => null) // network error, non-JSON body, or our own abort — treat as "no data"
    .finally(() => clearTimeout(timeoutId));
}

export function useApi(urls, interval = REFRESH_MS) {
  const [data, setData] = useState(urls.map(() => null));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Guards against the interval firing a new cycle while the previous one
  // is still in flight (e.g. a slow backend taking longer than `interval`
  // to respond) — without this, two overlapping fetchAll() calls can
  // resolve out of order and let a stale response overwrite a fresher one.
  const isFetchingRef = useRef(false);

  const fetchAll = useCallback(async (isRefresh = false) => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    if (isRefresh) setRefreshing(true);
    try {
      const results = await Promise.all(urls.map(fetchOne));
      setData(results);
      setLastUpdated(new Date());
      // fetchOne never rejects, so a total outage shows up as every slot
      // being null rather than a thrown error — surface that explicitly
      // instead of silently leaving stale data on screen.
      setError(urls.length > 0 && results.every(r => r === null)
        ? 'Impossible de joindre le serveur.'
        : null);
    } catch {
      setError('Impossible de joindre le serveur.');
    } finally {
      setLoading(false);
      setRefreshing(false);
      isFetchingRef.current = false;
    }
  }, [urls.join(',')]);

  useEffect(() => {
    fetchAll();
    const id = setInterval(() => fetchAll(), interval);
    return () => clearInterval(id);
  }, [fetchAll, interval]);

  return { data, loading, error, refreshing, lastUpdated, refresh: () => fetchAll(true) };
}
