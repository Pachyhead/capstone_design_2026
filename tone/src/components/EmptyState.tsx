interface Props {
  title: string;
  hint?: string;
}

export function EmptyState({ title, hint }: Props) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center text-center px-8">
      <div
        className="w-16 h-16 rounded-2xl mb-5 flex items-center justify-center"
        style={{ background: 'rgba(20,19,15,0.04)' }}
      >
        <div
          className="w-7 h-7 rounded-full"
          style={{ background: 'rgba(20,19,15,0.12)' }}
        />
      </div>
      <h2 className="text-[16px] font-medium text-ink mb-2">{title}</h2>
      {hint && <p className="text-[13px] text-muted leading-relaxed max-w-[320px]">{hint}</p>}
    </div>
  );
}
