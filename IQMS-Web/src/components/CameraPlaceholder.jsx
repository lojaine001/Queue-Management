import { useLang } from '../context/LanguageContext';

export default function CameraPlaceholder({ label, camId, dataUrl, metric }) {
  const { t } = useLang();
  const isLive = !!dataUrl;

  return (
    <div style={s.wrap}>
      <div style={s.labelRow}>
        <span style={{ ...s.dot, background: isLive ? 'var(--green)' : 'var(--text-off)' }} />
        <span className="micro-label">{label}</span>
      </div>
      <div style={s.frameWrap}>
        <div style={s.frame}>
          {isLive ? (
            <>
              <img src={dataUrl} alt={label} style={s.img} />
              {metric && <span className="mono" style={s.chip}>{metric}</span>}
            </>
          ) : (
            <span style={s.offlineText}>{t.cameraOffline(camId)}</span>
          )}
        </div>
      </div>
    </div>
  );
}

const s = {
  wrap: { flex: 1, minWidth: 0 },
  labelRow: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 },
  dot: { width: 6, height: 6, borderRadius: '50%', flexShrink: 0 },
  frameWrap: { position: 'relative', width: '100%', paddingBottom: '56.25%' },
  frame: {
    position: 'absolute', inset: 0,
    background: 'var(--video-bg)',
    border: '0.5px solid var(--hairline)',
    borderRadius: 'var(--radius)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    overflow: 'hidden',
  },
  img: { width: '100%', height: '100%', objectFit: 'cover', display: 'block' },
  offlineText: { fontSize: 12, color: 'var(--text-3)' },
  chip: {
    position: 'absolute', left: 8, bottom: 8,
    fontSize: 11, color: 'var(--text)',
    background: 'rgba(6, 8, 9, 0.85)',
    padding: '3px 7px', borderRadius: 'var(--radius)',
  },
};
