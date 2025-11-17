
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import SkyrimAGIPage from './pages/SkyrimAGIPage';
import LifeOpsPage from './pages/LifeOpsPage';
import DevelopmentPage from './pages/DevelopmentPage';
import SchematicsPage from './pages/SchematicsPage';
import ProjectHistoryPage from './pages/ProjectHistoryPage';
import AGIHistoryPage from './pages/AGIHistoryPage';
import ConclusionsPage from './pages/ConclusionsPage';

const App: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col font-sans bg-[#FAF8F5] text-slate-800">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-8 pt-24">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/skyrimagi" element={<SkyrimAGIPage />} />
          <Route path="/lifeops" element={<LifeOpsPage />} />
          <Route path="/development" element={<DevelopmentPage />} />
          <Route path="/schematics" element={<SchematicsPage />} />
          <Route path="/project-history" element={<ProjectHistoryPage />} />
          <Route path="/agi-history" element={<AGIHistoryPage />} />
          <Route path="/conclusions" element={<ConclusionsPage />} />
        </Routes>
      </main>
      <footer className="w-full bg-[#0D5A67] text-white/80 p-4 text-center text-sm">
        Singularis Project | An Exploration of AGI | 2025
      </footer>
    </div>
  );
};

export default App;
