import { useEffect, useState } from 'react';
import { useLang } from '../context/LanguageContext';

export default function UpdatedAgo({ lastUpdated }) {
  const { t } = useLang();
  const [, tick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => tick(n => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  if (!lastUpdated) return null;

  const secs = Math.max(0, Math.round((Date.now() - lastUpdated.getTime()) / 1000));
  const stale = secs > 60;
  const label = stale ? t.staleData(Math.round(secs / 60)) : (secs < 3 ? t.justNow : t.secondsAgo(secs));

  return (
    <span className="mono" style={{ fontSize: 11, color: stale ? 'var(--amber)' : 'var(--text-3)' }}>
      {label}
    </span>
  );
}
