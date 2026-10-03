// IQMS service worker.
//
// Stage 1: lets registration.showNotification(...) put alerts in the OS
// notification tray/bar even while the site is just open in a background
// tab -- plain `new Notification(...)` can't do that reliably on Android
// Chrome/Edge/Brave, which is the bug this originally existed to fix.
// Stage 2 (this): a 'push' handler, so the same notifications arrive even
// when no tab of the site is open at all -- sent by push_alert_checker.py
// on the server, delivered through the browser's push service.

self.addEventListener('install', () => {
  // Activate this version immediately rather than waiting for every old
  // tab to close -- there's no state here worth preserving across updates.
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});

// payload is {"title", "body", "url"} -- see push_alert_checker.py. Falls
// back to generic text if the payload is missing or isn't valid JSON,
// rather than silently showing nothing (a push with no visible
// notification is also a spec violation browsers can penalize).
self.addEventListener('push', event => {
  let payload = { title: 'IQMS', body: 'Alert update.', url: '/' };
  try {
    if (event.data) payload = { ...payload, ...event.data.json() };
  } catch (err) {
    console.warn('Push payload was not valid JSON:', err);
  }
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      tag: 'iqms-wait-alert',
      renotify: true,
      data: { url: payload.url || '/' },
    })
  );
});

// Tapping a notification should bring the app to the front, not just
// dismiss it -- reuse an already-open tab if there is one, since opening a
// duplicate tab every time would be confusing.
self.addEventListener('notificationclick', event => {
  event.notification.close();
  const url = event.notification.data?.url || '/';
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      for (const client of clientList) {
        if ('focus' in client) return client.focus();
      }
      if (self.clients.openWindow) return self.clients.openWindow(url);
    })
  );
});
