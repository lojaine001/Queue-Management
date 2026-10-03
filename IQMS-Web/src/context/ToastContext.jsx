import { createContext, useCallback, useContext, useState } from 'react';

const ToastCtx = createContext(null);
const TONE = { success: '#3fb950', error: '#f85149' };

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const showToast = useCallback((message, tone = 'success', duration = 2500) => {
    const id = Date.now() + Math.random();
    setToasts(list => [...list, { id, message, tone }]);
    setTimeout(() => {
      setToasts(list => list.filter(x => x.id !== id));
    }, duration);
  }, []);

  const errorToasts = toasts.filter(t => t.tone === 'error');
  const successToasts = toasts.filter(t => t.tone !== 'error');

  return (
    <ToastCtx.Provider value={showToast}>
      {children}

      {/* Alerts/errors — big, top of screen, hard to miss */}
      <div style={s.wrapTop}>
        {errorToasts.map(t => (
          <div key={t.id} style={s.toastBig}>
            <span style={s.dotBig} />
            {t.message}
          </div>
        ))}
      </div>

      {/* Routine confirmations — small, bottom, out of the way */}
      <div style={s.wrapBottom}>
        {successToasts.map(t => (
          <div key={t.id} style={{ ...s.toast, borderColor: TONE[t.tone] }}>
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
  wrapBottom: {
    position: 'fixed', bottom: 20, left: '50%', transform: 'translateX(-50%)',
    display: 'flex', flexDirection: 'column', gap: 8, zIndex: 100,
    alignItems: 'center', pointerEvents: 'none',
  },
  toast: {
    background: '#161b22', border: '1px solid', borderRadius: 10,
    padding: '10px 16px', color: '#e6edf3', fontSize: 13, fontWeight: 500,
    display: 'flex', alignItems: 'center', gap: 8,
    boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
    animation: 'toast-in 0.2s ease-out',
  },
  dot: { width: 6, height: 6, borderRadius: '50%', flexShrink: 0 },

  wrapTop: {
    position: 'fixed', top: 24, left: '50%', transform: 'translateX(-50%)',
    display: 'flex', flexDirection: 'column', gap: 10, zIndex: 200,
    alignItems: 'center', pointerEvents: 'none', width: '90%', maxWidth: 640,
  },
  toastBig: {
    background: '#2d1a1a', border: '2px solid #f85149', borderRadius: 12,
    padding: '18px 24px', color: '#ffffff', fontSize: 18, fontWeight: 700,
    display: 'flex', alignItems: 'center', gap: 12, width: '100%',
    boxShadow: '0 8px 28px rgba(0,0,0,0.55)',
    animation: 'toast-in 0.2s ease-out',
  },
  dotBig: {
    width: 12, height: 12, borderRadius: '50%', background: '#f85149',
    flexShrink: 0, boxShadow: '0 0 10px #f85149',
  },
};
