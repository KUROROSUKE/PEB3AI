const CACHE_NAME = 'my-game-cache-v1';
const urlsToCache = [
  '/',
  '/Code/game.html',
  '/Code/game.css',
  '/Code/game.js',
  '/images/start_screen_mobile.webp',
];

// インストール
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

// リクエストをキャッシュから取得
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
