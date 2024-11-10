"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import ChatWindow from "@/components/conversation/ChatWindow";
import ChatInput from "@/components/conversation/ChatInput";
import useLocalStorage from "@/hooks/useLocalStorage"; // Import the localStorage hook

const ChatPage = () => {
  const { id } = useParams();
  const router = useRouter();

  const [chatRooms] = useLocalStorage("chatRooms", []);
  const [messages, setMessages] = useLocalStorage(`messages_${id}`, []);
  const [isLoading, setIsLoading] = useState(true);
  const [chatRoomId, setChatRoomId] = useState<string>("");

  useEffect(() => {
    // Check if the current chat room exists based on its ID
    const roomExists = chatRooms.some((room:any) => room.id === id);
    if (!roomExists) router.push("/");
    else 
    {
      setIsLoading(false);
      // Check if id is a string and assign it to chatRoomId

      if (typeof id === "string") {
        setChatRoomId(id);
      } else if (Array.isArray(id)) {
        setChatRoomId(id[0]); 
      }
    }
  }, [id, chatRooms, router]);

 

  const sendMessage = async (input: string) => {
    if (input.trim()) {
      const newMessage = { role: "user", content: input };
      const updatedMessages = [...messages, newMessage];

      setMessages(updatedMessages); 
      const response = await fetch('/api/ai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          conversation_id: chatRoomId,
        }),
      });

      const data = await response.json();
      const aiMessage = { role: "ai", content: data.response };
      const newMessages = [...updatedMessages, aiMessage];
      setMessages(newMessages); 
      
      // setTimeout(() => {
        
      //   
      // }, 1000);
    }
  };

  // Prevent rendering while loading
  if (isLoading) return null;

  return (
    <div className="flex flex-col h-[80vh]">
      <ChatWindow messages={messages} />
      <ChatInput onSendMessage={sendMessage} />
    </div>
  );
};

export default ChatPage;
