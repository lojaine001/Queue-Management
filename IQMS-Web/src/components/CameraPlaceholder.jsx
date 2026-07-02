export default function CameraPlaceholder({ label, stat }) {
  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <span style={s.dot} />
        <span style={s.label}>{label}</span>
      </div>
      <div style={s.frame}>
        <span style={s.icon}>▣</span>
      </div>
      {stat && <div style={s.stat}>{stat}</div>}
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
    background: '#484f58',
  },
  label: {
    fontSize: 11,
    fontWeight: 600,
    color: '#8b949e',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  frame: {
    height: 110,
    background: '#0d1117',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    fontSize: 32,
    color: '#30363d',
  },
  stat: {
    padding: '6px 10px',
    fontSize: 11,
    color: '#8b949e',
    borderTop: '1px solid #30363d',
  },
};
