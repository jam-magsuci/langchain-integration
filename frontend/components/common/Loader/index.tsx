
"use client";
import { RingLoader } from 'react-spinners';

import { useLoading } from "@/components/providers/LoadingProvider";
import { useEffect } from 'react';

const LoadingScreen: React.FC = () => {
  const { loading, context } = useLoading();

  // enhancement -> Prevent scrolling while loading.
  useEffect(() => {
    if (loading) {
      
      document.body.classList.add('overflow-hidden');
    } else {

      document.body.classList.remove('overflow-hidden');
    }
  }, [loading]);

  if (!loading) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-800 bg-opacity-80 dark:bg-gray-900 dark:bg-opacity-80 select-none">
        <div className="flex flex-col items-center">
            <RingLoader color="#3498db" size={100} />
            <p className="mt-4 text-lg font-semibold text-white dark:text-gray-200 animate-pulse">{context !== "" ? context : "AI-Prompt"}</p>
        </div>
    </div>

  );
};

export default LoadingScreen;
