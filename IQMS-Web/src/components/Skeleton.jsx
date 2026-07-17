export default function Skeleton({ width = '100%', height = 16, style }) {
  return (
    <div
      style={{
        width, height,
        borderRadius: 'var(--radius)',
        background: 'var(--raised)',
        animation: 'pulse 1.4s ease-in-out infinite',
        ...style,
      }}
    />
  );
}
