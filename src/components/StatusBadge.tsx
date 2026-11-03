import React from 'react';

interface StatusBadgeProps {
  label: string;
  status: 'active' | 'warning' | 'error' | 'inactive';
  text: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, status, text }) => {
  const getColors = () => {
    switch (status) {
      case 'active':
        return 'text-emerald-400';
      case 'warning':
        return 'text-amber-400';
      case 'error':
        return 'text-rose-400';
      case 'inactive':
      default:
        return 'text-slate-500';
    }
  };

  const getGlyph = () => {
    switch (status) {
      case 'active':
        return '●';
      case 'warning':
        return '▲';
      case 'error':
        return '✖';
      case 'inactive':
      default:
        return '○';
    }
  };

  return (
    <div className="flex items-center gap-1.5 text-xs font-mono">
      <span className="text-slate-400 font-sans">{label}:</span>
      <span className={`inline-flex items-center gap-1 font-semibold ${getColors()}`}>
        <span aria-hidden="true" className="text-[10px]">{getGlyph()}</span>
        <span>{text}</span>
      </span>
    </div>
  );
};
