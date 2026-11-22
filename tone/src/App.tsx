import { BrowserRouter, Navigate, Routes, Route, Outlet } from 'react-router-dom';
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

function Shell() {
  const { activeProfile } = useProfiles();

  // gate: must pick a profile before entering the app.
  if (!activeProfile) {
    return <Navigate to="/profiles" replace />;
  }

  return (
    <DesktopShell variant="light">
      <Outlet />
    </DesktopShell>
  );
}

function ChatLayout() {
  return (
    <div className="flex h-full min-h-0 flex-1 w-full">
      <ChatList />
      <div className="flex-1 min-w-0 h-full">
        <Outlet />
      </div>
    </div>
  );
}

function StatsLayout() {
  return (
    <div className="flex h-full min-h-0 flex-1 w-full">
      <StatsList />
      <div className="flex-1 min-w-0 h-full">
        <Outlet />
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
        <Route element={<Shell />}>
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
