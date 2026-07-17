import { useEffect, useState, useCallback } from 'react';
import { HEADERS, REFRESH_MS } from '../config';

export function useApi(urls, interval = REFRESH_MS) {
  const [data, setData] = useState(urls.map(() => null));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchAll = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      setError(null);
      const results = await Promise.all(
        urls.map(u =>
          fetch(u, { headers: HEADERS }).then(r => (r.ok ? r.json() : null)).catch(() => null)
        )
      );
      setData(results);
      setLastUpdated(new Date());
    } catch {
      setError('Impossible de joindre le serveur.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [urls.join(',')]);

  useEffect(() => {
    fetchAll();
    const id = setInterval(() => fetchAll(), interval);
    return () => clearInterval(id);
  }, [fetchAll, interval]);

  return { data, loading, error, refreshing, lastUpdated, refresh: () => fetchAll(true) };
}
