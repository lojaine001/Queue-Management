// Shared notification helper -- one place that actually shows a
// notification, so every alert site uses the same (working) path instead
// of each reimplementing it slightly differently.
//
// Uses the service worker's showNotification() rather than `new
// Notification()` directly: that's what actually reaches the OS
// notification tray/bar on Android Chrome/Edge/Brave, which plain
// `new Notification()` does not do reliably there (desktop was never the
// problem). Falls back to the old direct API if no service worker is
// available (e.g. a browser that doesn't support one at all).

let swRegistration = null;

export function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return Promise.resolve(null);
  return navigator.serviceWorker
    .register('/sw.js')
    .then(reg => {
      swRegistration = reg;
      return reg;
    })
    .catch(err => {
      console.warn('Service worker registration failed:', err);
      return null;
    });
}

// Must be called from a direct user gesture (a click handler), not
// automatically -- browsers increasingly ignore or auto-deny permission
// prompts triggered any other way, especially on mobile.
export function requestNotificationPermission() {
  if (!('Notification' in window)) return Promise.resolve('unsupported');
  if (Notification.permission !== 'default') return Promise.resolve(Notification.permission);
  return Notification.requestPermission();
}

export function notificationPermissionState() {
  if (!('Notification' in window)) return 'unsupported';
  return Notification.permission;
}

// tag: a per-alert-type identifier. Firing the same tag again replaces the
// previous notification instead of stacking a new one next to it --
// e.g. "iqms-wait-alert" so repeated wait-time alerts don't pile up in
// the tray.
export async function showAppNotification(title, body, tag) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  const options = { body, tag, renotify: true };
  try {
    const reg = swRegistration || (await navigator.serviceWorker.getRegistration());
    if (reg) {
      await reg.showNotification(title, options);
      return;
    }
  } catch (err) {
    console.warn('showNotification via service worker failed, falling back:', err);
  }
  // No active service worker (e.g. desktop Safari) -- still works the
  // original way there, so this isn't a regression for those browsers.
  new Notification(title, options);
}
