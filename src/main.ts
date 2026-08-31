import './styles.css';

import { createApp } from './app';

function start(): void {
  createApp(document);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}
