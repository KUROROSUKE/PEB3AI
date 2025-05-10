const CACHE_NAME = 'my-game-cache-v1';
const urlsToCache = [
  'https://kurorosuke.github.io/PEB3AI/Code/game.min.html',
  'https://kurorosuke.github.io/PEB3AI/Code/game.min.css',
  'https://kurorosuke.github.io/PEB3AI/Code/game.min.js',
  'https://kurorosuke.github.io/PEB3AI/images/start_screen_mobile.webp',
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
