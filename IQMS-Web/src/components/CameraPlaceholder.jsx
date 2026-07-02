export default function CameraPlaceholder({ label, dataUrl }) {
  const hasImage = !!dataUrl;
  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <span style={{ ...s.dot, background: hasImage ? '#3fb950' : '#484f58' }} />
        <span style={s.label}>{label}</span>
      </div>
      {/* Aspect-ratio box: always 16:9, scales with width */}
      <div style={s.frameWrap}>
        <div style={s.frame}>
          {hasImage ? (
            <img src={dataUrl} alt={label} style={s.img} />
          ) : (
            <span style={s.icon}>▣</span>
          )}
        </div>
      </div>
    </div>
  );
}

const s = {
  wrap: {
    flex: 1,
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 10,
    overflow: 'hidden',
    minWidth: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 12px',
    borderBottom: '1px solid #30363d',
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
    color: '#8b949e',
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
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  img: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    display: 'block',
  },
  icon: {
    fontSize: 40,
    color: '#21262d',
  },
};
