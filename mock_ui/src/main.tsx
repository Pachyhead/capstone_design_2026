import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '@/App';
import '@/index.css';
import './mock.css';
import './autoplay.css';
import { AutoplayBar, ClickShield, StartOverlay } from './AutoplayBar';
import { PipelinePanel } from './PipelinePanel';
import { seedProfiles, stopAutoplay } from './autoplay';
import { resetMockState } from './mockApi';
import { DEMO_PROFILES } from './scenario';

// 이 값이 저장된 값과 다르면 브라우저에 남은 데모 상태를 지우고 처음 상태로 맞춘다.
// 프로필 구성을 바꾸면 값을 올린다.
const SEED_KEY = 'tone-mock:seed';
const SEED_VERSION = '2';

/** 직접 조작할 때의 처음 상태: 프로필 3명이 모두 있다. (자동 시연은 시작할 때 따로 맞춘다.) */
function seedDemoStorage() {
  if (window.localStorage.getItem(SEED_KEY) === SEED_VERSION) return;
  resetMockState();
  seedProfiles(DEMO_PROFILES);
  window.localStorage.setItem(SEED_KEY, SEED_VERSION);
}

function restart() {
  stopAutoplay();
  resetMockState();
  window.localStorage.removeItem(SEED_KEY);
  window.location.hash = '';
  window.location.reload();
}

seedDemoStorage();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <div className="mock-root">
      <AutoplayBar onReset={restart} />
      <div className="mock-stage">
        <div className="mock-app">
          <App />
        </div>
        <ClickShield />
      </div>
      <PipelinePanel />
      <StartOverlay />
    </div>
  </React.StrictMode>,
);
