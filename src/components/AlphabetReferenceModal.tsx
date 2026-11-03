import React, { useState } from 'react';
import { X, Info, Hand, Image as ImageIcon } from 'lucide-react';

interface AlphabetReferenceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectLetter?: (letter: string) => void;
}

interface LetterInfo {
  letter: string;
  pose: string;
  fingers: string;
  isStatic: boolean;
}

const ALPHABET_GUIDE: LetterInfo[] = [
  { letter: 'A', pose: 'Closed fist', fingers: 'Thumb rests upright against the side of index finger.', isStatic: true },
  { letter: 'B', pose: 'Flat open palm', fingers: '4 fingers extended straight up together, thumb folded across palm.', isStatic: true },
  { letter: 'C', pose: 'Curved hand', fingers: 'Fingers and thumb form a curved "C" shape.', isStatic: true },
  { letter: 'D', pose: 'Index pointing up', fingers: 'Index finger points up, other 3 fingers touch thumb to form a circle.', isStatic: true },
  { letter: 'E', pose: 'Fingers curled tight', fingers: 'All 4 fingers curled down tight, fingertips resting on thumb curled under.', isStatic: true },
  { letter: 'F', pose: 'OK sign shape', fingers: 'Index and thumb tips touch in circle, other 3 fingers extended upward.', isStatic: true },
  { letter: 'G', pose: 'Index pointing forward', fingers: 'Index finger pointing forward horizontally, thumb parallel, others curled.', isStatic: true },
  { letter: 'H', pose: 'Two fingers forward', fingers: 'Index and middle fingers extended together horizontally forward.', isStatic: true },
  { letter: 'I', pose: 'Pinky finger up', fingers: 'Pinky finger points straight up, other fingers and thumb curled into fist.', isStatic: true },
  { letter: 'J', pose: 'Pinky tracing "J"', fingers: 'Motion required: Draw the letter J in the air with the pinky finger.', isStatic: false },
  { letter: 'K', pose: 'Index up, middle angled', fingers: 'Index up, middle finger angled forward, thumb tucked between them.', isStatic: true },
  { letter: 'L', pose: '"L" shape', fingers: 'Index finger points up, thumb extended sideways at 90 degrees.', isStatic: true },
  { letter: 'M', pose: '3 fingers over thumb', fingers: 'Fist with thumb tucked under index, middle, and ring fingers.', isStatic: true },
  { letter: 'N', pose: '2 fingers over thumb', fingers: 'Fist with thumb tucked under index and middle fingers.', isStatic: true },
  { letter: 'O', pose: 'Round "O" circle', fingers: 'All fingertips curved down to touch the thumb tip in an "O" shape.', isStatic: true },
  { letter: 'P', pose: '"K" pointing down', fingers: 'Same as K handshape, but pointed downwards towards floor.', isStatic: true },
  { letter: 'Q', pose: '"G" pointing down', fingers: 'Same as G handshape, but pointed downwards towards floor.', isStatic: true },
  { letter: 'R', pose: 'Crossed fingers', fingers: 'Index and middle fingers crossed over each other pointing straight up.', isStatic: true },
  { letter: 'S', pose: 'Fist over thumb', fingers: 'Fist with thumb folded across the front of the curled fingers.', isStatic: true },
  { letter: 'T', pose: 'Thumb under index', fingers: 'Fist with thumb tucked under the index finger only.', isStatic: true },
  { letter: 'U', pose: 'Two fingers up together', fingers: 'Index and middle fingers held together straight up.', isStatic: true },
  { letter: 'V', pose: '"V" peace sign', fingers: 'Index and middle fingers spread apart in a "V" shape.', isStatic: true },
  { letter: 'W', pose: '"W" three fingers', fingers: 'Index, middle, and ring fingers spread upward in a "W" shape.', isStatic: true },
  { letter: 'X', pose: 'Hooked index finger', fingers: 'Index finger crooked/hooked like a claw, other fingers curled.', isStatic: true },
  { letter: 'Y', pose: '"Hang loose" / shaka', fingers: 'Thumb and pinky extended out, middle 3 fingers curled in.', isStatic: true },
  { letter: 'Z', pose: 'Tracing "Z"', fingers: 'Motion required: Draw the letter Z in the air with the index finger.', isStatic: false },
];

export const AlphabetReferenceModal: React.FC<AlphabetReferenceModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [viewMode, setViewMode] = useState<'grid' | 'chart'>('grid');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#111622] border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0d121c]">
          <div className="flex items-center gap-2">
            <Hand size={18} className="text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 tracking-wide">
              American Sign Language Alphabet Reference (StartASL)
            </h3>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-mono">
              <button
                onClick={() => setViewMode('grid')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  viewMode === 'grid' ? 'bg-slate-800 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Pose Descriptions
              </button>
              <button
                onClick={() => setViewMode('chart')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  viewMode === 'chart' ? 'bg-slate-800 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Chart Image
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto">
          {viewMode === 'chart' ? (
            <div className="flex flex-col items-center justify-center">
              <div className="bg-white rounded-xl p-3 shadow-lg max-w-2xl border border-slate-700">
                <img
                  src="/assets/aistudio/image_4c5bb0df.png"
                  alt="American Sign Language Alphabet Chart"
                  className="w-full h-auto rounded"
                  referrerPolicy="no-referrer"
                />
              </div>
              <p className="text-xs text-slate-400 font-mono mt-3 text-center">
                Official StartASL American Sign Language Alphabet Reference. Notice motion arrows on letters J and Z.
              </p>
            </div>
          ) : (
            <div>
              <div className="mb-4 p-3 bg-slate-900/60 rounded-lg border border-slate-800 text-xs text-slate-400 flex items-start gap-2">
                <Info size={16} className="text-cyan-400 shrink-0 mt-0.5" />
                <span>
                  The SIGNAL model supports 24 static handshapes. Hold your hand approximately 1.5 to 2.5 feet from your webcam with good lighting. Notice that letters <strong>J</strong> and <strong>Z</strong> require motion and cannot be captured statically.
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {ALPHABET_GUIDE.map((item) => (
                  <div
                    key={item.letter}
                    className={`p-3 rounded-lg border transition-colors ${
                      item.isStatic
                        ? 'bg-[#0a0d14] border-slate-800/80 hover:border-slate-700'
                        : 'bg-rose-950/20 border-rose-900/40'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xl font-bold text-cyan-400">
                          {item.letter}
                        </span>
                        <span className="text-xs font-semibold text-slate-200">
                          {item.pose}
                        </span>
                      </div>
                      {!item.isStatic && (
                        <span className="text-[10px] font-mono text-rose-400 bg-rose-950/60 px-1.5 py-0.5 rounded border border-rose-800">
                          Motion
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      {item.fingers}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#0d121c] flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>SIGNAL v1.0.0 · 24 Static Classes</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded font-medium transition-colors"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
};
