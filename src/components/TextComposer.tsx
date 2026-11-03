import React, { useState } from 'react';
import { Plus, Space, Delete, Trash2, Copy, Check, Volume2 } from 'lucide-react';
import { countWords } from '../lib/textComposer';

interface TextComposerProps {
  currentText: string;
  currentLetter: string | null;
  isStable: boolean;
  onAddLetter: () => void;
  onAddSpace: () => void;
  onBackspace: () => void;
  onClear: () => void;
  autoCommit: boolean;
  onToggleAutoCommit: () => void;
}

export const TextComposer: React.FC<TextComposerProps> = ({
  currentText,
  currentLetter,
  isStable,
  onAddLetter,
  onAddSpace,
  onBackspace,
  onClear,
  autoCommit,
  onToggleAutoCommit,
}) => {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState<string | null>(null);

  const handleCopy = async () => {
    if (!currentText) return;
    console.info('[Composer] Copy clicked', { characters: currentText.length });
    try {
      await navigator.clipboard.writeText(currentText);
      setCopyError(null);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      console.info('[Composer] Clipboard write succeeded');
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error('[Composer] Clipboard write failed:', message);
      setCopyError('Clipboard access failed. Check browser permissions.');
    }
  };

  const handleSpeak = () => {
    if (!currentText || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(currentText);
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex flex-col bg-[#111622] rounded-xl border border-slate-800/80 p-5 shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div>
          <h2 className="text-xs uppercase font-mono font-semibold tracking-wider text-slate-400">
            Composed Text
          </h2>
          <p className="text-[11px] text-slate-500">
            Assembled from recognized ASL alphabet gestures
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Auto commit toggle */}
          <button
            onClick={onToggleAutoCommit}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
              autoCommit
                ? 'bg-cyan-950/60 border-cyan-800 text-cyan-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <span className="text-[10px]">{autoCommit ? '●' : '○'}</span>
            <span>Auto-Commit: {autoCommit ? 'ON' : 'OFF'}</span>
          </button>
        </div>
      </div>

      {/* Main Text Display Box */}
      <div className="relative min-h-[90px] p-4 bg-[#0a0d14] rounded-lg border border-slate-800/90 flex flex-col justify-between mb-4">
        <div className="font-mono text-xl sm:text-2xl font-bold tracking-wide text-slate-100 break-words whitespace-pre-wrap select-all">
          {currentText ? (
            <span>{currentText}</span>
          ) : (
            <span className="text-slate-600 font-normal text-base italic select-none">
              Sign gestures in front of the camera, then click "Add Letter" to compose words...
            </span>
          )}
          {/* Blinking cursor */}
          <span className="inline-block w-2.5 h-5 ml-1 bg-cyan-400 animate-pulse align-middle" />
        </div>

        <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-900 text-[11px] text-slate-500 font-mono">
          <span>
            Characters: {currentText.length} · Words: {countWords(currentText)}
            {copyError && <span className="ml-2 text-rose-400">{copyError}</span>}
          </span>
          {currentText && (
            <button
              onClick={handleSpeak}
              className="flex items-center gap-1 text-slate-400 hover:text-cyan-400 transition-colors"
            >
              <Volume2 size={13} />
              <span>Read Aloud</span>
            </button>
          )}
        </div>
      </div>

      {/* Primary Action Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        <button
          onClick={onAddLetter}
          disabled={!currentLetter || !isStable}
          className={`flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-lg text-xs font-bold font-mono transition-all shadow-sm ${
            currentLetter && isStable
              ? 'bg-cyan-400 hover:bg-cyan-300 text-slate-950 ring-2 ring-cyan-400/40'
              : currentLetter
              ? 'bg-cyan-500/80 hover:bg-cyan-400 text-slate-950'
              : 'bg-slate-800/50 text-slate-500 cursor-not-allowed'
          }`}
        >
          <Plus size={15} />
          <span>Add Letter {currentLetter ? `(${currentLetter})` : ''}</span>
        </button>

        <button
          onClick={onAddSpace}
          className="flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium font-mono transition-colors"
        >
          <Space size={14} />
          <span>Space</span>
        </button>

        <button
          onClick={onBackspace}
          disabled={!currentText}
          className="flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:hover:bg-slate-800 text-slate-200 rounded-lg text-xs font-medium font-mono transition-colors"
        >
          <Delete size={14} />
          <span>Backspace</span>
        </button>

        <button
          onClick={onClear}
          disabled={!currentText}
          className="flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800/70 hover:bg-rose-950/60 hover:text-rose-300 disabled:opacity-40 text-slate-300 rounded-lg text-xs font-medium font-mono transition-colors"
        >
          <Trash2 size={14} />
          <span>Clear</span>
        </button>

        <button
          onClick={handleCopy}
          disabled={!currentText}
          className="flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 rounded-lg text-xs font-medium font-mono transition-colors"
        >
          {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>
    </div>
  );
};
