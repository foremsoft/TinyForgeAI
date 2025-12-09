import React from 'react'
import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom'
import TrainPage from './pages/TrainPage'
import ServicesPage from './pages/ServicesPage'
import LogsPage from './pages/LogsPage'
import PlaygroundPage from './pages/PlaygroundPage'

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

export default function App() {
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
      </nav>
      <main style={mainStyle}>
        <Routes>
          <Route path="/" element={<TrainPage />} />
          <Route path="/services" element={<ServicesPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
          <Route path="/logs" element={<LogsPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
