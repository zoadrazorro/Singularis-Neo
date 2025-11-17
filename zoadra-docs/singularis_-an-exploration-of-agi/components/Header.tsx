
import React from 'react';
import { NavLink as RouterNavLink, useLocation } from 'react-router-dom';
import { NAV_LINKS } from '../constants';
import { NavLink } from '../types';

const Header: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    // Handle home page vs other pages
    if (path === '#/') {
      return location.hash === '#/';
    }
    return location.hash.startsWith(path);
  };
  
  return (
    <header className="fixed top-0 left-0 right-0 bg-white/80 backdrop-blur-lg shadow-md z-50">
      <nav className="container mx-auto px-4 py-3 flex justify-between items-center">
        <RouterNavLink to="/" className="text-2xl font-bold text-[#0D5A67] hover:text-[#E76F51] transition-colors">
          Singularis
        </RouterNavLink>
        <div className="hidden md:flex items-center space-x-2 lg:space-x-4">
          {NAV_LINKS.map((link: NavLink) => (
            <RouterNavLink
              key={link.path}
              to={link.path.replace('#', '')}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive(link.path)
                  ? 'bg-[#0D5A67] text-white'
                  : 'text-slate-600 hover:bg-[#0D5A67]/10 hover:text-[#0D5A67]'
              }`}
            >
              {link.label}
            </RouterNavLink>
          ))}
        </div>
      </nav>
    </header>
  );
};

export default Header;
