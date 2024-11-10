import React, { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage }) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const sendMessage = () => {
    if (input.trim()) {
      onSendMessage(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"; // Reset height
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`; // Set height based on scrollHeight
    }
  }, [input]);

  return (
    <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-full max-w-3xl px-4 py-5 bg-gray-100 dark:bg-gray-800 shadow-lg rounded-lg flex md:mx-[150px] transition-all ease-in-out">
      <div className="flex items-center w-full">
        <textarea
          ref={textareaRef}
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          className="flex-1 px-4 py-2 rounded-lg bg-white dark:bg-gray-700 dark:text-gray-100 border focus:outline-none transition-all"
          style={{ maxHeight: "150px" }}
        />
        <button
          onClick={sendMessage}
          className="ml-4 p-2 rounded-full bg-blue-500 text-white flex-shrink-0 transition-all transform hover:bg-blue-600 active:scale-95 dark:bg-blue-600 dark:hover:bg-blue-700"
        >
          <Send className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
