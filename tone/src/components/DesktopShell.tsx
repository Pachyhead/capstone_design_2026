import { type ReactNode } from 'react';
import { SideRail } from '@/components/SideRail';

interface Props {
  children: ReactNode;
  variant?: 'light' | 'dark';
}

export function DesktopShell({ children, variant = 'light' }: Props) {
  const bg = variant === 'dark' ? '#3A2A1A' : '#FAF6EF';

  return (
    <div className="flex h-screen w-screen overflow-hidden" style={{ background: bg }}>
      <SideRail variant={variant} />
      <main className="flex-1 min-w-0 h-full flex">{children}</main>
    </div>
  );
}
