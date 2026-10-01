// IQMS service worker.
//
// Stage 1 (current): lets registration.showNotification(...) put alerts in
// the OS notification tray/bar even while the site is just open in a
// background tab -- plain `new Notification(...)` can't do that reliably
// on Android Chrome/Edge/Brave, which is the bug this exists to fix.
// Stage 2 will add a 'push' handler here for notifications that work even
// when no tab of the site is open at all.

self.addEventListener('install', () => {
  // Activate this version immediately rather than waiting for every old
  // tab to close -- there's no state here worth preserving across updates.
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});

// Tapping a notification should bring the app to the front, not just
// dismiss it -- reuse an already-open tab if there is one, since opening a
// duplicate tab every time would be confusing.
self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      for (const client of clientList) {
        if ('focus' in client) return client.focus();
      }
      if (self.clients.openWindow) return self.clients.openWindow('/');
    })
  );
});
