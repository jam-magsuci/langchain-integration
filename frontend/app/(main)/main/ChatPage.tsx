"use client"; 

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useChat } from "@/components/providers/ChatProvider"; // Access chat context
import ChatWindow from "@/components/conversation/ChatWindow";
import ChatInput from "@/components/conversation/ChatInput";
import useChatMessages from "@/hooks/useChatMessage"; // Custom hook for chat messages
import useLocalStorage from "@/hooks/useLocalStorage"; // Import the localStorage hook
import { v4 as uuidv4 } from 'uuid';


const ChatPage = () => {
  const { addChatRoom } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  // Generate a unique chat room ID when the component mounts
  const chatRoomId = uuidv4();
  const { messages, addMessage } = useChatMessages(chatRoomId); // Pass chatRoomId to the hook

  // Use the localStorage hook to manage messages
  const [storedMessages, setStoredMessages] = useLocalStorage(`messages_${chatRoomId}`, []); 

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [storedMessages]); // Scroll to the end of the messages when storedMessages change

  const handleSendMessage = async (input: string) => {
    if (input.trim()) {
      const newMessage = { role: "user", content: input };
      addMessage(newMessage); // Call the modified addMessage that expects only the new message
      
      // Update stored messages with the new user message
      const updatedMessages = [...storedMessages, newMessage];
      setStoredMessages(updatedMessages);

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
      // console.log("data: ", data);

      const aiMessage = { role: "ai", content: data.response };
      const finalMessages = [...updatedMessages, aiMessage];
      setStoredMessages(finalMessages); // Update stored messages with AI response

      const maxTitleLength = 20; 
      const chatRoomTitle = input.length > maxTitleLength 
      ? `${input.slice(0, maxTitleLength)}...` 
      : input; 

      const newRoom = { id: chatRoomId, title: chatRoomTitle };
      addChatRoom(newRoom);
      router.push(`/conversation/${chatRoomId}`);
    }
  };

  return (
    <div className="flex flex-col h-[80vh]">
      <ChatWindow messages={storedMessages} /> {/* Use storedMessages instead of messages */}
      <ChatInput onSendMessage={handleSendMessage} />
      <div ref={messagesEndRef} /> {/* Reference for scrolling */}
    </div>
  );
};

export default ChatPage;
