import { NavLink } from 'react-router-dom';
import { Avatar } from '@/components/Avatar';
import { useUserAvatar } from '@/hooks/useUserAvatar';

interface Props {
  variant?: 'light' | 'dark';
}

export function SideRail({ variant = 'light' }: Props) {
  const isDark = variant === 'dark';
  const railBg = isDark ? '#2D1F13' : '#EDE7DB';
  const borderColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(20,19,15,0.06)';
  const [avatar] = useUserAvatar();

  return (
    <nav
      className="flex flex-col items-center py-5 gap-1 flex-shrink-0"
      style={{
        width: 72,
        background: railBg,
        borderRight: `0.5px solid ${borderColor}`,
      }}
    >
      <div
        className="w-9 h-9 rounded-[10px] flex items-center justify-center text-[14px] font-semibold mb-3"
        style={{
          background: isDark ? '#5A4126' : '#14130F',
          color: isDark ? '#F2D89E' : '#FAF6EF',
        }}
      >
        T
      </div>

      <RailItem to="/" label="대화" end isDark={isDark}>
        {(color) => <ChatIcon color={color} />}
      </RailItem>
      <RailItem to="/me" label="나" isDark={isDark}>
        {() => <Avatar avatar={avatar} size={28} />}
      </RailItem>
    </nav>
  );
}

function RailItem({
  to,
  label,
  end,
  isDark,
  children,
}: {
  to: string;
  label: string;
  end?: boolean;
  isDark: boolean;
  children: (color: string) => React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className="w-[56px] flex flex-col items-center gap-[5px] py-[8px] rounded-[10px] transition-colors"
    >
      {({ isActive }) => {
        const fg = isActive
          ? isDark
            ? '#FAF6EF'
            : '#14130F'
          : isDark
            ? 'rgba(250,246,239,0.55)'
            : '#9B8E7B';
        const bg = isActive
          ? isDark
            ? 'rgba(250,246,239,0.08)'
            : 'rgba(20,19,15,0.06)'
          : 'transparent';
        return (
          <div
            className="w-full flex flex-col items-center gap-[5px] py-[8px] rounded-[10px]"
            style={{ background: bg }}
          >
            {children(fg)}
            <span
              className="text-[11px] tracking-wide"
              style={{ color: fg, fontWeight: isActive ? 600 : 400 }}
            >
              {label}
            </span>
          </div>
        );
      }}
    </NavLink>
  );
}

function ChatIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path
        d="M4 6.5C4 5.39543 4.89543 4.5 6 4.5H18C19.1046 4.5 20 5.39543 20 6.5V15.5C20 16.6046 19.1046 17.5 18 17.5H9L5 21V6.5Z"
        stroke={color}
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}


