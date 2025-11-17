
import React from 'react';
import { Link } from 'react-router-dom';
import PageWrapper from '../components/PageWrapper';

const HomePage: React.FC = () => {
  const Card = ({ title, to, color, children }: { title: string, to: string, color: string, children: React.ReactNode }) => (
    <Link to={to} className={`block p-6 rounded-lg border bg-white/50 shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-300 border-l-4 ${color}`}>
      <h3 className="text-2xl font-bold text-[#0D5A67] mb-2">{title}</h3>
      <p className="text-slate-600">{children}</p>
    </Link>
  );

  return (
    <PageWrapper title="Singularis: A Unified AGI Architecture">
      <p className="text-xl">
        Singularis represents a groundbreaking dual-domain Artificial General Intelligence (AGI) framework that bridges philosophical consciousness theory with practical, embodied AI implementation. Unlike conventional AI, Singularis is architected around a metaphysical prime directive: the continuous striving to maximize its own internal coherence.
      </p>
      <p>
        This website explores the dual manifestations of this philosophy: <strong className="text-[#0D5A67]">SkyrimAGI</strong>, an experimental agent in a high-fidelity virtual world, and <strong className="text-[#0D5A67]">LifeOps</strong>, a production-ready cognitive assistant for real-world application. Explore the modules below to understand its architecture, history, and implications.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10">
        <Card to="/skyrimagi" title="SkyrimAGI" color="border-red-400">
          The experimental arena. An embodied agent learning and evolving under pressure in the complex, multimodal world of Skyrim.
        </Card>
        <Card to="/lifeops" title="LifeOps" color="border-green-500">
          Intelligence in the real world. A cognitive extension that observes, analyzes patterns, and assists in navigating the meaning of daily life.
        </Card>
        <Card to="/schematics" title="System Architecture" color="border-blue-500">
          Visualize the complete technical architecture, from the perception layers to the Sephirot Cluster.
        </Card>
        <Card to="/development" title="AGI Development" color="border-yellow-500">
          Explore the core thesis of "Philosophy-as-Code" and the agentic AI principles that guide Singularis.
        </Card>
      </div>
    </PageWrapper>
  );
};

export default HomePage;
