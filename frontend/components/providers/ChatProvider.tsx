"use client";
import { createContext, useContext, useEffect, useState } from "react";

interface ChatRoom {
  id: string;
  title: string;
}

interface Message {
  role: string;
  content: string;
}

interface ChatContextType {
  chatRooms: ChatRoom[];
  addChatRoom: (room: ChatRoom) => void;
  deleteChatRoom: (id: string) => void; // Add delete functionality
  addMessage: (chatRoomId: string, message: Message) => void;
  loading: boolean;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [chatRooms, setChatRooms] = useState<ChatRoom[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const loadChatRooms = () => {
      if (typeof window !== 'undefined') {
        const storedRooms = window.localStorage.getItem("chatRooms");
        if (storedRooms) {
          setChatRooms(JSON.parse(storedRooms));
        }
        setLoading(false);
      }
    };
    loadChatRooms();
  }, []);

  const addChatRoom = (room: ChatRoom) => {
    setChatRooms((prev) => {
      const updatedChatRooms = [...prev, room];

      // Store the updated chat rooms in local storage
      if (typeof window !== 'undefined') {
        window.localStorage.setItem("chatRooms", JSON.stringify(updatedChatRooms));
      }

      return updatedChatRooms;
    });
  };

  const deleteChatRoom = (id: string) => {
    setChatRooms((prev) => {
      const updatedChatRooms = prev.filter((room) => room.id !== id);

      // Update local storage with the new chat rooms list
      if (typeof window !== 'undefined') {
        window.localStorage.setItem("chatRooms", JSON.stringify(updatedChatRooms));
        window.localStorage.removeItem(`messages_${id}`); // Optionally remove chat messages
      }

      return updatedChatRooms;
    });
  };

  const addMessage = (chatRoomId: string, message: Message) => {
    const existingMessages = JSON.parse(window.localStorage.getItem(`messages_${chatRoomId}`) || "[]");
    const updatedMessages = [...existingMessages, message];
    window.localStorage.setItem(`messages_${chatRoomId}`, JSON.stringify(updatedMessages));
  };

  return (
    <ChatContext.Provider value={{ chatRooms, addChatRoom, deleteChatRoom, addMessage, loading }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};
