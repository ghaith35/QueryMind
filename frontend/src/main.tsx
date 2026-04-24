import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { StyleGuide } from './StyleGuide.tsx'

const root = document.getElementById('root')!
const isStyleGuide = window.location.pathname === '/style-guide'

createRoot(root).render(isStyleGuide ? <StyleGuide /> : <App />)
