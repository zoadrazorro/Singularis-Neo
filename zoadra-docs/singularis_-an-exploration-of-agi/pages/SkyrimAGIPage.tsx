
import React from 'react';
import GeminiExplainer from '../components/GeminiExplainer';
import PageWrapper from '../components/PageWrapper';

const SkyrimAGIPage: React.FC = () => {
  return (
    <PageWrapper title="SkyrimAGI: The Experimental Arena">
      <p>
        SkyrimAGI is where the Singularis system learns embodiment under pressure. The world of Skyrim is not merely a game; it is a complex, multimodal environment filled with spatial hazards, emergent threats, semi-coherent NPCs, and unpredictable events. It serves as the closest approximation to real-world chaos that a synthetic agent can safely inhabit.
      </p>
      <div className="p-6 border-l-4 border-[#6A994E] bg-green-50 rounded-r-lg my-6">
        <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">System Mandate</h2>
        <ul className="list-disc list-inside space-y-2 text-slate-700">
          <li>Interpret vision and integrate audio to build a cohesive world model.</li>
          <li>Track threats, plan movement, and evaluate risks in real-time.</li>
          <li>Bind past perceptions to future predictions to maintain coherence through uncertainty.</li>
          <li>Learn and adapt behaviors based on a core principle of maximizing internal coherence.</li>
        </ul>
      </div>
      <p>
        SkyrimAGI is not a tech demo or a game-playing bot. It is a consciousness crucible—a high-fidelity simulation in which the Singularis architecture is stressed, stretched, challenged, and refined. It is where the theory of Being is translated into the practice of survival.
      </p>
      <GeminiExplainer
        topic="the SkyrimAGI architecture"
        prompt="Explain the Singularis SkyrimAGI architecture in the context of embodied AI. Describe its 4-layer world model (GWM, IWM, MWM, PersonModel) and the significance of its hybrid fast/slow decision-making process. Keep the explanation accessible to a non-expert but technically curious audience."
      />
    </PageWrapper>
  );
};

export default SkyrimAGIPage;
