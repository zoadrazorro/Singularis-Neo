
import React from 'react';

interface PageWrapperProps {
  title: string;
  children: React.ReactNode;
}

const PageWrapper: React.FC<PageWrapperProps> = ({ title, children }) => {
  return (
    <div className="animate-fade-in">
      <h1 className="text-4xl md:text-5xl font-bold text-[#0D5A67] mb-4 pb-2 border-b-2 border-[#E76F51]/50">
        {title}
      </h1>
      <div className="space-y-6 text-lg text-slate-700 leading-relaxed">
        {children}
      </div>
    </div>
  );
};

export default PageWrapper;
