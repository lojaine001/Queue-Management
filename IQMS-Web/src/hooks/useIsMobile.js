import { useEffect, useState } from 'react';

// Matches the @media (max-width: 899px) breakpoint used across styles.css
export function useIsMobile() {
  const query = '(max-width: 899px)';
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = (e) => setIsMobile(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  return isMobile;
}
