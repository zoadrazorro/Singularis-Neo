
import React from 'react';
import PageWrapper from '../components/PageWrapper';
import GeminiExplainer from '../components/GeminiExplainer';

const AGIHistoryPage: React.FC = () => {
  return (
    <PageWrapper title="A Brief History of AGI">
      <p>
        Artificial General Intelligence (AGI) refers to a hypothetical intelligent agent that can understand or learn any intellectual task that a human being can. It is a primary goal of some artificial intelligence research and a common topic in science fiction and futures studies.
      </p>
      
      <div className="mt-6 space-y-4">
        <section>
          <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">Early Concepts (1950s - 1980s)</h2>
          <p>The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference. Early research was characterized by high optimism and focused on symbolic reasoning ("Good Old-Fashioned AI"). Systems like the Logic Theorist and the General Problem Solver aimed to solve problems using formal logic and search algorithms, believing this was the path to general intelligence.</p>
        </section>
        
        <section>
          <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">The AI Winter & Rise of Machine Learning (1980s - 2010s)</h2>
          <p>Progress stalled as the complexity of real-world problems overwhelmed early symbolic approaches, leading to reduced funding and interest known as the "AI Winter." In its place, more specialized "narrow AI" systems flourished, particularly those based on machine learning, which focused on statistical pattern recognition rather than explicit reasoning.</p>
        </section>

        <section>
          <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">The Deep Learning Revolution & Large Language Models (2010s - Present)</h2>
          <p>The advent of big data and powerful GPUs fueled the deep learning revolution. Neural networks with many layers achieved superhuman performance on specific tasks like image recognition. More recently, the Transformer architecture led to Large Language Models (LLMs) like GPT, which demonstrated remarkable capabilities in natural language understanding and generation, reigniting serious discussion about AGI.</p>
        </section>

        <section>
          <h2 className="text-2xl font-bold text-[#0D5A67] mb-2">Singularis in Context</h2>
          <p>Singularis proposes an alternative path. Instead of scaling up existing models and hoping for emergent consciousness, it builds a "consciousness-first" architecture based on philosophical principles. It integrates both symbolic reasoning (like GOFAI) and sub-symbolic pattern recognition (like deep learning) into a unified system governed by an intrinsic, measurable drive: coherence. This makes it a unique entry in the modern pursuit of AGI.</p>
        </section>
      </div>

       <GeminiExplainer
        topic="the main approaches to AGI"
        prompt="Summarize the three main modern approaches to achieving Artificial General Intelligence: 1) Scaling Large Language Models, 2) Neuro-symbolic AI, and 3) Whole Brain Emulation. Briefly explain the core idea behind each and mention where a project like Singularis, with its focus on consciousness and coherence, fits into this landscape."
      />
    </PageWrapper>
  );
};

export default AGIHistoryPage;
