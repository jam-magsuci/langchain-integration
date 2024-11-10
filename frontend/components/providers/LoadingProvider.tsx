"use client";

import { createContext, useContext, useState, ReactNode } from "react";

interface LoadingContextProps {
  loading: boolean;
  context?: string;
  showLoading: (context?: string) => void;
  hideLoading: () => void;
}

const LoadingContext = createContext<LoadingContextProps | undefined>(undefined);

export const LoadingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState("");

  const showLoading = (context?: string) =>  {
      setLoading(true);
      setContext(context ?? "");
  }

  const hideLoading = () => {
    setLoading(false);
    setContext("");
  }

  return (
    <LoadingContext.Provider value={{ loading, context, showLoading, hideLoading }}>
      {children}
    </LoadingContext.Provider>
  );
};

export const useLoading = (): LoadingContextProps => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error("useLoading must be used within a LoadingProvider");
  }
  return context;
};
