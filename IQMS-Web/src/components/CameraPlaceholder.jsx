export default function CameraPlaceholder({ label, dataUrl }) {
  const hasImage = !!dataUrl;
  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <span style={{ ...s.dot, background: hasImage ? '#3fb950' : '#484f58' }} />
        <span style={s.label}>{label}</span>
      </div>
      <div style={s.frame}>
        {hasImage ? (
          <img src={dataUrl} alt={label} style={s.img} />
        ) : (
          <span style={s.icon}>▣</span>
        )}
      </div>
    </div>
  );
}

const s = {
  wrap: {
    flex: 1,
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 8,
    overflow: 'hidden',
    minWidth: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 10px',
    borderBottom: '1px solid #30363d',
  },
  dot: {
    width: 6,
    height: 6,
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
  frame: {
    height: 130,
    background: '#0d1117',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  img: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  icon: {
    fontSize: 32,
    color: '#30363d',
  },
};
