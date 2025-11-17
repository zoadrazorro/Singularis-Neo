
import React from 'react';
import PageWrapper from '../components/PageWrapper';
import GeminiExplainer from '../components/GeminiExplainer';

const Block = ({ title, color, children }: { title: string, color: string, children?: React.ReactNode }) => (
  <div className={`border ${color} bg-white/60 p-3 rounded-md text-center shadow-sm`}>
    <h3 className="font-bold text-sm text-[#0D5A67]">{title}</h3>
    {children && <div className="text-xs text-slate-600 mt-1">{children}</div>}
  </div>
);

const SchematicsPage: React.FC = () => {
  return (
    <PageWrapper title="System Schematics">
      <p>
        The Singularis architecture is a direct computational analogue to biological cognitive systems. It is a synthetic nervous system complete with sensory pathways, motor pathways, global workspaces, and a coherence-driven executive function. Below is a simplified representation of the data flow in the SkyrimAGI implementation.
      </p>

      <div className="my-8 p-4 md:p-8 bg-gray-50 border border-gray-200 rounded-lg overflow-x-auto">
        <div className="relative flex flex-col items-center space-y-2 font-mono text-xs">
          {/* Layers */}
          <div className="w-full p-2 bg-green-100/50 border border-green-300 rounded-lg text-center font-bold text-green-800">INPUT/OUTPUT LAYER</div>
          <div className="flex justify-between w-full max-w-4xl">
            <Block color="border-green-400" title="Screenshot" />
            <Block color="border-green-400" title="Gamepad Out" />
            <Block color="border-green-400" title="Game State" />
          </div>
          <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-blue-100/50 border border-blue-300 rounded-lg text-center font-bold text-blue-800">PERCEPTION LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-slate-100 border border-slate-300 rounded-lg text-center font-bold text-slate-800">WORLD MODEL LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-indigo-100/50 border border-indigo-300 rounded-lg text-center font-bold text-indigo-800">COGNITIVE LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-yellow-100/50 border border-yellow-400 rounded-lg text-center font-bold text-yellow-800">COORDINATION LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-purple-100/50 border border-purple-300 rounded-lg text-center font-bold text-purple-800">LEARNING LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-red-100/50 border border-red-300 rounded-lg text-center font-bold text-red-800">CONTROL LAYER</div>
           <div className="text-2xl text-gray-400">↓</div>
          <div className="w-full p-2 bg-gray-200 border border-gray-400 rounded-lg text-center font-bold text-gray-800">SEPHIROT CLUSTER</div>
           <div className="flex justify-between w-full max-w-4xl mt-2">
            <Block color="border-gray-500" title="Node A: AMD" />
            <Block color="border-gray-500" title="Node B: Desk" />
            <Block color="border-gray-500" title="Node C: Laptop" />
            <Block color="border-gray-500" title="Node D: Tablet" />
          </div>
        </div>
      </div>

      <GeminiExplainer
        topic="the system architecture"
        prompt="Analyze the 'SkyrimAGI System Architecture' diagram. Explain the data flow from the INPUT/OUTPUT LAYER through the PERCEPTION, WORLD MODEL, COGNITIVE, COORDINATION, LEARNING, and CONTROL LAYERS. Describe the role of the Sephirot Cluster at the bottom. Simplify the explanation for someone interested in AI system design but not an expert."
      />
    </PageWrapper>
  );
};

export default SchematicsPage;
