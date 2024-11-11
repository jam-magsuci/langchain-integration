"use client";
import { useEffect, useState } from "react";

const useChatMessages = (chatRoomId: string) => {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(() => {
    if (typeof window !== "undefined") {
      const storedMessages = window.localStorage.getItem(`messages_${chatRoomId}`);
      return storedMessages ? JSON.parse(storedMessages) : [];
    }
    return [];
  });

  useEffect(() => {
    // Sync messages with localStorage whenever they change
    if (typeof window !== "undefined") {
    //   window.localStorage.setItem(`messages_${chatRoomId}`, JSON.stringify(messages));
    }
  }, [messages, chatRoomId]);

  const addMessage = (message: { role: string; content: string }) => {
    setMessages((prevMessages) => [...prevMessages, message]);
  };

  return { messages, addMessage };
};

export default useChatMessages;
