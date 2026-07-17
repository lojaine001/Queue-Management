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
  const label = secs < 3 ? t.justNow : t.secondsAgo(secs);
  const stale = secs > 60;

  return (
    <span style={s.wrap}>
      <span style={{ ...s.dot, background: stale ? '#d29922' : '#3fb950' }} />
      <span style={{ ...s.text, color: stale ? '#d29922' : '#8b949e' }}>{label}</span>
    </span>
  );
}

const s = {
  wrap: { display: 'flex', alignItems: 'center', gap: 5 },
  dot: {
    width: 6, height: 6, borderRadius: '50%',
    boxShadow: '0 0 5px currentColor',
    animation: 'pulse 2s infinite',
  },
  text: { fontSize: 11, fontFamily: 'var(--font-mono)' },
};
