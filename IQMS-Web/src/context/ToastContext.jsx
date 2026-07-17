import { createContext, useCallback, useContext, useState } from 'react';

const ToastCtx = createContext(null);
const TONE = { success: 'var(--green)', error: 'var(--red)' };

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const showToast = useCallback((message, tone = 'success') => {
    const id = Date.now() + Math.random();
    setToasts(list => [...list, { id, message, tone }]);
    setTimeout(() => {
      setToasts(list => list.filter(x => x.id !== id));
    }, 2500);
  }, []);

  return (
    <ToastCtx.Provider value={showToast}>
      {children}
      <div style={s.wrap}>
        {toasts.map(t => (
          <div key={t.id} style={s.toast}>
            <span style={{ ...s.dot, background: TONE[t.tone] }} />
            {t.message}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);

const s = {
  wrap: {
    position: 'fixed', bottom: 20, left: '50%', transform: 'translateX(-50%)',
    display: 'flex', flexDirection: 'column', gap: 8, zIndex: 100,
    alignItems: 'center', pointerEvents: 'none',
  },
  toast: {
    background: 'var(--surface)', border: '0.5px solid var(--hairline)', borderRadius: 'var(--radius)',
    padding: '10px 16px', color: 'var(--text)', fontSize: 13, fontWeight: 400,
    display: 'flex', alignItems: 'center', gap: 8,
    animation: 'fade-in 150ms ease-out',
  },
  dot: { width: 6, height: 6, borderRadius: '50%', flexShrink: 0 },
};
