import React from 'react';
import { X, Network, Cpu, Layers, CheckCircle2 } from 'lucide-react';

interface ArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#111622] border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0d121c]">
          <div className="flex items-center gap-2">
            <Network size={18} className="text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 tracking-wide">
              SIGNAL System & Model Architecture (PyTorch MLP)
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Master Infographic preview */}
          <div className="bg-[#0a0d14] p-3 rounded-xl border border-slate-800 flex flex-col items-center">
            <img
              src="/assets/aistudio/image_4d215270.png"
              alt="SIGNAL Master Architecture and Workflow"
              className="w-full max-h-[380px] object-contain rounded"
              referrerPolicy="no-referrer"
            />
            <span className="text-[11px] text-slate-500 font-mono mt-2">
              SIGNAL Full Specification Diagram (Workflow, Topology, MLP Architecture & Evaluation)
            </span>
          </div>

          {/* Pipeline breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-[#0a0d14] rounded-lg border border-slate-800/80">
              <div className="flex items-center gap-2 text-cyan-400 mb-2 font-mono text-xs font-bold uppercase">
                <Layers size={14} />
                <span>1. Vision Tracking</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                MediaPipe Tasks Vision running locally in browser. Extracts 21 3D coordinates (x, y, z) representing hand joints at 30+ FPS.
              </p>
            </div>

            <div className="p-4 bg-[#0a0d14] rounded-lg border border-slate-800/80">
              <div className="flex items-center gap-2 text-indigo-400 mb-2 font-mono text-xs font-bold uppercase">
                <Cpu size={14} />
                <span>2. Feature Normalization</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Wrist-relative translation (p_i - p_0) and Euclidean hand-scale normalization. Produces an invariant 63-element vector.
              </p>
            </div>

            <div className="p-4 bg-[#0a0d14] rounded-lg border border-slate-800/80">
              <div className="flex items-center gap-2 text-emerald-400 mb-2 font-mono text-xs font-bold uppercase">
                <CheckCircle2 size={14} />
                <span>3. PyTorch MLP Classifier</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                63 → Linear(128) → ReLU → Linear(64) → ReLU → Linear(24) → Softmax. Evaluated at 99.44% accuracy on held-out test data.
              </p>
            </div>
          </div>

          {/* Model Hyperparameters */}
          <div className="p-4 bg-[#0a0d14] rounded-lg border border-slate-800 text-xs font-mono">
            <h4 className="text-slate-300 font-bold mb-3 uppercase tracking-wider font-sans">
              Model Training Configuration
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-400">
              <div>
                <span className="text-slate-600 block text-[10px]">OPTIMIZER</span>
                <span className="text-slate-200">Adam (lr=0.001)</span>
              </div>
              <div>
                <span className="text-slate-600 block text-[10px]">LOSS FUNCTION</span>
                <span className="text-slate-200">CrossEntropyLoss</span>
              </div>
              <div>
                <span className="text-slate-600 block text-[10px]">REGULARIZATION</span>
                <span className="text-slate-200">Dropout (p=0.2)</span>
              </div>
              <div>
                <span className="text-slate-600 block text-[10px]">WEIGHT DECAY</span>
                <span className="text-slate-200">1e-4</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#0d121c] flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>Trained Model: backend/models/asl_mlp_model.pth</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
