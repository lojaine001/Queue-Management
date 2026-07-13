import { useLang } from '../context/LanguageContext';

export default function CameraPlaceholder({ label, dataUrl }) {
  const { t } = useLang();
  const isLive = !!dataUrl;
  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <span style={{ ...s.dot, background: isLive ? '#3fb950' : '#484f58' }} />
        <span style={s.label}>{label}</span>
      </div>
      {/* Aspect-ratio box: always 16:9, scales with width */}
      <div style={s.frameWrap}>
        <div style={s.frame}>
          <span style={s.icon}>▣</span>
          <span style={s.pendingText}>{t.cameraPending}</span>
        </div>
      </div>
    </div>
  );
}

const s = {
  wrap: {
    flex: 1,
    background: 'var(--card-bg)',
    border: '1px solid var(--card-border)',
    borderRadius: 16,
    overflow: 'hidden',
    minWidth: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '10px 16px',
    borderBottom: '1px solid var(--card-border)',
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    flexShrink: 0,
  },
  label: {
    fontSize: 11,
    fontWeight: 600,
    color: '#8b95a8',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  frameWrap: {
    position: 'relative',
    width: '100%',
    paddingBottom: '56.25%', /* 16:9 */
  },
  frame: {
    position: 'absolute',
    inset: 0,
    background: '#0a0d12',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    overflow: 'hidden',
  },
  icon: {
    fontSize: 34,
    color: '#2a3140',
  },
  pendingText: {
    fontSize: 11,
    fontWeight: 500,
    color: '#484f58',
    letterSpacing: 0.3,
  },
};
