"use client";
import { Box, SquarePen, Trash2 } from "lucide-react"; // Import the Trash2 icon
import Link from "next/link";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat } from "@/components/providers/ChatProvider";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { v4 as uuidv4 } from 'uuid';

const Sidebar: React.FC = () => {
  const { chatRooms, addChatRoom, deleteChatRoom, loading } = useChat(); // Include deleteChatRoom
  const [isClient, setIsClient] = useState(false); // Track if running on the client
  const pathname = usePathname();
  const router = useRouter();

  // Ensure this only runs on the client
  useEffect(() => {
    setIsClient(true); // Once mounted, confirm we are on the client
  }, []);

  const createNewChat = () => {
    if (!isClient) return; // Ensure client-side only operation
    // const chatRoomId = uuid();
    // const newChatRoom = { id: chatRoomId, title: `Chat ${chatRoomId}` };
    // addChatRoom(newChatRoom);
    router.push(`/?conversation=new`);
  };

  if (loading) {
    return (
      <aside className="md:block border-r bg-muted/40 h-screen flex items-center justify-center">
        <div className="text-lg font-semibold">Loading chat rooms...</div>
      </aside>
    );
  }

  return (
    <aside className="hidden md:block border-r bg-muted/40 h-screen">
      <div className="flex flex-col h-full">
        <div className="flex justify-between h-14 items-center border-b px-4 lg:h-[60px] lg:px-6">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Box className="h-6 w-6" />
            <span>AI Prompting</span>
          </Link>
          <button onClick={createNewChat} className="flex items-center gap-2 text-sm">
            <SquarePen />
          </button>
        </div>

        <ScrollArea className="flex-1 px-2 lg:px-4 py-5">
          <nav className="grid items-start text-sm font-medium">
            {chatRooms.map((room) => {
              const isActive = pathname === `/conversation/${room.id}`;
              return (
                <div key={room.id} className={`flex justify-between items-center  hover:bg-gray-200  ${
                  isActive ? "bg-blue-500 text-white rounded-md" : ""
                }`}>
                  <Link
                    href={`/conversation/${room.id}`}
                    className={`block p-2 flex-1`}
                  >
                    {room.title}
                  </Link>
                  <button
                    onClick={() => {
                      deleteChatRoom(room.id);
                      if(pathname === `/conversation/${room.id}`) router.push(`/`);
                    }}
                    className="text-red-600 p-1 hover:bg-red-500 rounded"
                  >
                    <Trash2  color="#c7c7c7" className="h-5 w-5" />
                  </button>
                </div>
              );
            })}
          </nav>
        </ScrollArea>

        <footer className="border-t p-4 text-center">
          <span>© AI Prompting</span>
        </footer>
      </div>
    </aside>
  );
};

export default Sidebar;
