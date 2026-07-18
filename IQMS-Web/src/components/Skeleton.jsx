export default function Skeleton({ width = '100%', height = 16, radius = 8, style }) {
  return (
    <div
      style={{
        width, height, borderRadius: radius,
        background: '#1c2128',
        animation: 'pulse 1.4s ease-in-out infinite',
        ...style,
      }}
    />
  );
}
