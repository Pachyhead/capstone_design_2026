import { BrowserRouter, Navigate, Routes, Route, Outlet, useLocation, useOutletContext } from 'react-router-dom';
import { useState } from 'react';
import { ChatList } from '@/screens/ChatList';
import { ChatDetail } from '@/screens/ChatDetail';
import { StatsDetail } from '@/screens/StatsDetail';
import { StatsList } from '@/screens/StatsList';
import { StatsOverview } from '@/screens/StatsOverview';
import { Me } from '@/screens/Me';
import { Onboarding } from '@/screens/Onboarding';
import { Profiles } from '@/screens/Profiles';
import { DesktopShell } from '@/components/DesktopShell';
import { EmptyState } from '@/components/EmptyState';
import { useProfiles } from '@/hooks/useProfiles';

type ViewMode = 'text' | 'voice';
export type ShellContext = { mode: ViewMode; setMode: (m: ViewMode) => void };

const VIEW_MODE_KEY = 'tone:viewMode';

function ShellWithMode() {
  const [mode, setModeState] = useState<ViewMode>(() => {
    const url = new URL(window.location.href);
    const param = url.searchParams.get('mode');
    if (param === 'voice' || param === 'text') return param;
    if (typeof window !== 'undefined') {
      const stored = window.localStorage.getItem(VIEW_MODE_KEY);
      if (stored === 'voice' || stored === 'text') return stored;
    }
    return 'text';
  });
  const setMode = (m: ViewMode) => {
    setModeState(m);
    if (typeof window !== 'undefined') window.localStorage.setItem(VIEW_MODE_KEY, m);
  };
  const location = useLocation();
  const { activeProfile } = useProfiles();

  // gate: must pick a profile before entering the app.
  if (!activeProfile) {
    return <Navigate to="/profiles" replace />;
  }

  const isVoiceChatDetail = location.pathname.startsWith('/chat/') && mode === 'voice';

  return (
    <DesktopShell variant={isVoiceChatDetail ? 'dark' : 'light'}>
      <Outlet context={{ mode, setMode } satisfies ShellContext} />
    </DesktopShell>
  );
}

function ChatLayout() {
  const ctx = useOutletContext<ShellContext>();
  return (
    <div className="flex h-full min-h-0 flex-1 w-full">
      <ChatList />
      <div className="flex-1 min-w-0 h-full">
        <Outlet context={ctx} />
      </div>
    </div>
  );
}

function StatsLayout() {
  const ctx = useOutletContext<ShellContext>();
  return (
    <div className="flex h-full min-h-0 flex-1 w-full">
      <StatsList />
      <div className="flex-1 min-w-0 h-full">
        <Outlet context={ctx} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/profiles" element={<Profiles />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route element={<ShellWithMode />}>
          <Route element={<ChatLayout />}>
            <Route index element={<EmptyState title="대화를 선택하세요" hint="왼쪽 목록에서 대화방을 선택하면 여기에 표시됩니다." />} />
            <Route path="chat/:id" element={<ChatDetail />} />
          </Route>
          <Route path="chat/:id/stats" element={<StatsDetail />} />
          <Route path="stats" element={<StatsLayout />}>
            <Route index element={<StatsOverview />} />
            <Route path=":id" element={<StatsDetail />} />
          </Route>
          <Route path="me" element={<Me />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
