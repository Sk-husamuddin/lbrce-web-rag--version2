'use client'

import { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'light' | 'dark'

type ThemeContextValue = {
  theme: Theme
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light')

  useEffect(() => {
    const savedTheme = window.localStorage.getItem('lbrce-theme') as Theme | null
    const preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
    const nextTheme = savedTheme === 'light' || savedTheme === 'dark' ? savedTheme : preferredTheme

    // The persisted preference is read once on mount to synchronize React with the browser.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(nextTheme)
    document.documentElement.classList.remove('light', 'dark')
    document.documentElement.classList.add(nextTheme)
  }, [])

  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark')
    document.documentElement.classList.add(theme)
    window.localStorage.setItem('lbrce-theme', theme)
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme: () => setTheme((current) => current === 'light' ? 'dark' : 'light') }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used within ThemeProvider')
  return context
}
