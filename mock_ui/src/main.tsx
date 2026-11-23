import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/App';
import '@/index.css';
import './mock.css';
import { PipelinePanel } from './PipelinePanel';
import { resetMockState } from './mockApi';
import { DEMO_PROFILES } from './scenario';

// 이 값이 저장된 값과 다르면 브라우저에 남은 데모 상태를 지우고 처음 상태로 맞춘다.
// 프로필 구성을 바꾸면 값을 올린다.
const SEED_KEY = 'tone-mock:seed';
const SEED_VERSION = '1';

const AVATARS = ['happy', 'surprised', 'neutral'] as const;

function clearDemoStorage() {
  const keys = Object.keys(window.localStorage).filter(
    (k) => k.startsWith('tone:') || k.startsWith('tone-mock:'),
  );
  keys.forEach((k) => window.localStorage.removeItem(k));
}

/** 프로필 3명이 있는 처음 상태를 만든다. 실제 앱이 읽는 키를 그대로 쓴다. */
function seedDemoStorage() {
  if (window.localStorage.getItem(SEED_KEY) === SEED_VERSION) return;
  clearDemoStorage();
  const profiles = DEMO_PROFILES.map((p, i) => ({
    id: p.id,
    name: p.name,
    avatar: { type: 'emotion', emotion: AVATARS[i % AVATARS.length] },
    backendId: p.backendId,
  }));
  window.localStorage.setItem('tone:profiles:v3', JSON.stringify(profiles));
  window.localStorage.setItem(SEED_KEY, SEED_VERSION);
}

function restart() {
  resetMockState();
  clearDemoStorage();
  window.location.hash = '';
  window.location.reload();
}

seedDemoStorage();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <div className="mock-root">
      <div className="mock-app">
        <App />
      </div>
      <PipelinePanel onReset={restart} />
    </div>
  </React.StrictMode>,
);
