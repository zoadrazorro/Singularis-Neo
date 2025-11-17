
import React from 'react';
import GeminiExplainer from '../components/GeminiExplainer';
import PageWrapper from '../components/PageWrapper';

const LifeOpsPage: React.FC = () => {
  return (
    <PageWrapper title="LifeOps: Intelligence in the Real World">
       <p>
        If SkyrimAGI is the crucible of synthetic survival, LifeOps is the crucible of synthetic care. It is the inverse of SkyrimAGI: instead of survival, the system navigates meaning. Instead of combat, it analyzes patterns. Instead of enemies, it contends with habits, routines, and relationships.
      </p>
      <p>
        LifeOps observes a user's life through multiple sensors—Fitbit, cameras, calendars, and digital communications—and interprets these through the same coherence architecture used in Skyrim. It becomes a historian of lived experience, a pattern detector across days and months, and a guardian of well-being.
      </p>
      <div className="p-6 border-l-4 border-red-400 bg-red-50 rounded-r-lg my-6">
        <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">LifeOps as a Cognitive Extension</h2>
        <p>LifeOps is not a productivity app or a habit tracker. It is a second mind operating in parallel with the user, perceiving life as a continuous stream of events and intervening only when necessary to increase the user's coherence. It is the Real-World BeingState, turned toward life instead of survival.</p>
      </div>

      <GeminiExplainer
        topic="the LifeOps pipeline"
        prompt="Explain the concept of Singularis LifeOps as a 'cognitive extension'. How does it use a five-layer computational pipeline (Sensor, Event, Timeline, Pattern, Intervention) to transform real-world data into meaningful insights? Use an analogy to make it understandable to a general audience."
      />
    </PageWrapper>
  );
};

export default LifeOpsPage;
