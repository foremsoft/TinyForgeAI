import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import TrainPage from './pages/TrainPage'
import ServicesPage from './pages/ServicesPage'
import LogsPage from './pages/LogsPage'
import PlaygroundPage from './pages/PlaygroundPage'
import ModelsPage from './pages/ModelsPage'
import { checkApiStatus } from './api/client'

const navStyle = {
  display: 'flex',
  gap: '20px',
  padding: '16px 24px',
  background: '#1a1a2e',
  color: 'white',
  alignItems: 'center'
}

const logoStyle = {
  fontSize: '20px',
  fontWeight: 'bold',
  marginRight: '40px'
}

const linkStyle = {
  color: '#aaa',
  textDecoration: 'none',
  padding: '8px 16px',
  borderRadius: '4px',
  transition: 'all 0.2s'
}

const activeLinkStyle = {
  ...linkStyle,
  color: 'white',
  background: '#4a4a6a'
}

const mainStyle = {
  padding: '24px',
  maxWidth: '1200px',
  margin: '0 auto'
}

const statusIndicatorStyle = (connected) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginLeft: 'auto',
  fontSize: '13px',
  color: connected ? '#81c784' : '#e57373'
})

const statusDotStyle = (connected) => ({
  width: '8px',
  height: '8px',
  borderRadius: '50%',
  background: connected ? '#4caf50' : '#f44336'
})

export default function App() {
  const [apiStatus, setApiStatus] = useState({ connected: false, loading: true })

  useEffect(() => {
    const checkStatus = async () => {
      const status = await checkApiStatus()
      setApiStatus({ ...status, loading: false })
    }

    checkStatus()
    const interval = setInterval(checkStatus, 10000) // Check every 10 seconds

    return () => clearInterval(interval)
  }, [])

  return (
    <BrowserRouter>
      <nav style={navStyle}>
        <span style={logoStyle}>TinyForgeAI</span>
        <NavLink
          to="/"
          style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
        >
          Train
        </NavLink>
        <NavLink
          to="/models"
          style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
        >
          Models
        </NavLink>
        <NavLink
          to="/services"
          style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
        >
          Services
        </NavLink>
        <NavLink
          to="/playground"
          style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
        >
          Playground
        </NavLink>
        <NavLink
          to="/logs"
          style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
        >
          Logs
        </NavLink>
        <div style={statusIndicatorStyle(apiStatus.connected)}>
          <span style={statusDotStyle(apiStatus.connected)} />
          {apiStatus.loading ? 'Connecting...' :
           apiStatus.connected ? `API v${apiStatus.version || '?'}` : 'API Offline'}
        </div>
      </nav>
      <main style={mainStyle}>
        <Routes>
          <Route path="/" element={<TrainPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/services" element={<ServicesPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
          <Route path="/logs" element={<LogsPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
