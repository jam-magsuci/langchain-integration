import React, { useEffect, useRef, useState } from "react";
import { Clipboard, User, Bot } from "lucide-react";
import { toast, useToast } from "../ui/use-toast";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { solarizedlight, dark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Message {
  role: "user" | "ai";
  content: string;
}

interface ChatWindowProps {
  messages: Message[];
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [copiedIndexes, setCopiedIndexes] = useState<{ [key: string]: boolean }>({});
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const { toast } = useToast();

  useEffect(() => {
    // Check for dark mode on mount and whenever the class changes
    const checkDarkMode = () => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    };

    // Set initial dark mode
    checkDarkMode();

    // Listen for changes in dark mode
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, { attributes: true });

    return () => observer.disconnect(); // Clean up observer on unmount
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const copyToClipboard = (code: string, index: string) => {
    navigator.clipboard.writeText(code).then(() => {
      setCopiedIndexes(prev => ({ ...prev, [index]: true })); // Set the copied index
      toast({
        title: "Success!",
        description: "Code copied to clipboard!",
        className: "fixed top-4 right-3 md:max-w-96"
      });

      // Reset the copied index after a delay
      setTimeout(() => {
        setCopiedIndexes(prev => ({ ...prev, [index]: false })); // Reset specific index
      }, 2000);
    }).catch(() => {
      alert("Failed to copy code.");
    });
  };

  const formatCode = (code: string) => {
    return code.split('\n').map(line => line.trim()).join('\n');
  };



const renderMessageContent = (message: Message, messageIndex: number) => {
  if (message.role === "ai") {
    // Clean the content to remove unnecessary lines
    const cleanedContent = message.content.replace(/^[\s*-]+(?:\n|$)/gm, '');

    // Regex to capture code blocks in different languages
    const codeRegex = /```(\w+)?\n([\s\S]*?)```/g; // Matches code blocks
    const parts = [];
    let lastIndex = 0;
    let match;

    // Parse the response for text and code blocks
    while ((match = codeRegex.exec(cleanedContent)) !== null) {
      if (match.index > lastIndex) {
        // Push preceding text if exists
        parts.push({ type: 'text', content: cleanedContent.slice(lastIndex, match.index).trim() });
      }
      // Push code block with language if available
      parts.push({ type: 'code', language: match[1] || 'plaintext', content: match[2].trim() });
      lastIndex = match.index + match[0].length; // Update last index
    }

    // Push remaining text after the last code block
    if (lastIndex < cleanedContent.length) {
      parts.push({ type: 'text', content: cleanedContent.slice(lastIndex).trim() });
    }

    const codeParts = parts.filter(part => part.type === 'code');

    return (
      <div className="space-y-4">
        {parts.map((part, index) => {
          if (part.type === 'text' && part.content) {
            return (
              <div key={index} className="text-left break-words">
                {part.content}
              </div>
            );
          } else if (part.type === 'code') {
            const codeIndex = `${messageIndex}-${index}`; // Create a unique key for each code block
            return (
              <div key={codeIndex} className="relative bg-gray-800 text-white rounded-md p-4 mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-semibold text-gray-400">{part.language}</span>
                  <button
                    onClick={() => copyToClipboard(part.content, codeIndex)} // Function to copy code
                    className="text-xs bg-blue-600 text-white rounded-md px-2 py-1 hover:bg-blue-700 transition duration-200 flex items-center"
                  >
                    <Clipboard className="w-4 h-4 mr-1" />
                    {copiedIndexes[codeIndex] ? "Copied!" : "Copy code"}
                  </button>
                </div>
                {/* Render SyntaxHighlighter only if there are code parts */}
                <SyntaxHighlighter
                  language={part.language} // Use the language from the parsed code block
                  style={typeof document !== "undefined" && document.documentElement.classList.contains('dark') ? dark : solarizedlight}
                  customStyle={{ padding: '1rem', borderRadius: '0.5rem' }}
                >
                  {part.content}
                </SyntaxHighlighter>
              </div>
            );
          }
          return null; // Safeguard for other cases
        })}

        {/* If there are multiple code parts, you can render them together */}
        {codeParts.length > 1 && (
          <div className="bg-gray-800 text-white rounded-md p-4 mb-4">
            <h4 className="text-sm font-semibold text-gray-400">Multiple Code Blocks</h4>
            {codeParts.map((codePart, index) => (
              <div key={index} className="mb-2">
                <SyntaxHighlighter
                  language={codePart.language}
                  style={typeof document !== "undefined" && document.documentElement.classList.contains('dark') ? dark : solarizedlight}
                  customStyle={{ padding: '1rem', borderRadius: '0.5rem' }}
                >
                  {codePart.content}
                </SyntaxHighlighter>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Render non-AI messages as plain text
  return <div className="text-left break-words">{message.content}</div>;
};

  
    

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 transition-all ease-in-out dark:text-gray-100">
      {messages.length === 0 ? (
        <div className="text-gray-500 dark:text-gray-400 flex items-center justify-center h-full">
          <div className="text-center">No messages yet. Start the conversation!</div>
        </div>
      ) : (
        messages.map((message, index) => (
          <div
            key={index}
            className={`flex transition-all ease-in-out ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-3xl p-4 rounded-lg shadow-lg transition-all ${
                message.role === "user"
                  ? "bg-blue-500 text-white dark:bg-blue-600"
                  : "bg-gray-200 dark:bg-gray-700 dark:text-white"
              }`}
            >
              <div className="flex items-start mb-2">
                {message.role === "user" ? (
                  <User className="w-6 h-6 mr-2" />
                ) : (
                  <Bot className="w-6 h-6 mr-2" />
                )}
                <span className="font-semibold">{message.role === "user" ? "You" : "AI"}</span>
              </div>
              {renderMessageContent(message, index)}
            </div>
          </div>
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow;
