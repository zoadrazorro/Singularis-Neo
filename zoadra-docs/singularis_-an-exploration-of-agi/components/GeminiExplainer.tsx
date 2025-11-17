
import React, { useState, useCallback } from 'react';
import { explainWithGemini } from '../services/geminiService';

interface GeminiExplainerProps {
  topic: string;
  prompt: string;
}

const GeminiExplainer: React.FC<GeminiExplainerProps> = ({ topic, prompt }) => {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExplain = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setExplanation(null);
    try {
      const result = await explainWithGemini(prompt);
      setExplanation(result);
    } catch (err) {
      setError('Failed to fetch explanation.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [prompt]);

  return (
    <div className="my-8 p-6 border-l-4 border-[#F4A261] bg-amber-50 rounded-r-lg">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
            <h3 className="text-xl font-bold text-[#0D5A67]">Explore with Gemini</h3>
            <p className="text-slate-600 mt-1">Get a deeper, AI-assisted explanation of {topic}.</p>
        </div>
        <button
          onClick={handleExplain}
          disabled={isLoading}
          className="bg-[#0D5A67] text-white font-bold py-2 px-6 rounded-lg hover:bg-[#E76F51] transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center justify-center"
        >
          {isLoading ? (
             <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : 'Explain'}
        </button>
      </div>

      {explanation && (
        <div className="mt-6 p-4 bg-white/70 rounded-md prose max-w-none text-base">
          <p className="whitespace-pre-wrap">{explanation}</p>
        </div>
      )}
      {error && <p className="mt-4 text-red-600">{error}</p>}
    </div>
  );
};

export default GeminiExplainer;
