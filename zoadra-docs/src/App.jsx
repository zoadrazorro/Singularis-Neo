import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Architecture from './pages/Architecture'
import InfinityEngine from './pages/InfinityEngine'
import Philosophy from './pages/Philosophy'
import GettingStarted from './pages/GettingStarted'
import API from './pages/API'
import AGIHistory from './pages/AGIHistory'
import ProjectHistory from './pages/ProjectHistory'
import SkyrimAGI from './pages/SkyrimAGI'
import DevelopmentPhilosophy from './pages/DevelopmentPhilosophy'
import BlueprintSingularis from './pages/BlueprintSingularis'
import BlueprintSkyrimAGI from './pages/BlueprintSkyrimAGI'
import BlueprintLifeOps from './pages/BlueprintLifeOps'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/infinity-engine" element={<InfinityEngine />} />
        <Route path="/philosophy" element={<Philosophy />} />
        <Route path="/getting-started" element={<GettingStarted />} />
        <Route path="/api" element={<API />} />
        <Route path="/agi-history" element={<AGIHistory />} />
        <Route path="/project-history" element={<ProjectHistory />} />
        <Route path="/skyrim-agi" element={<SkyrimAGI />} />
        <Route path="/development-philosophy" element={<DevelopmentPhilosophy />} />
        <Route path="/blueprint-singularis" element={<BlueprintSingularis />} />
        <Route path="/blueprint-skyrimagi" element={<BlueprintSkyrimAGI />} />
        <Route path="/blueprint-lifeops" element={<BlueprintLifeOps />} />
      </Routes>
    </Layout>
  )
}

export default App
