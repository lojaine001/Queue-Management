import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';
import { registerServiceWorker } from './notify';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Registering doesn't request notification permission or show anything by
// itself -- it just makes the service worker available so showAppNotification
// (notify.js) has something to call once permission is actually granted.
registerServiceWorker();
