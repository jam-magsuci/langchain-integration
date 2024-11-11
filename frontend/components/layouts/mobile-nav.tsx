"use client";
import Link from "next/link";
import { Menu, Box, SquarePen, Trash2 } from "lucide-react"; // Import Trash2 icon
import { useChat } from "@/components/providers/ChatProvider";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { uuid } from 'uuidv4';

const MobileNav = () => {
  const { chatRooms, addChatRoom, deleteChatRoom, loading } = useChat(); // Include deleteChatRoom
  const [isClient, setIsClient] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  // Ensure this only runs on the client
  useEffect(() => {
    setIsClient(true);
  }, []);

  const createNewChat = () => {
    if (!isClient) return;
    
    // const chatRoomId = uuid();
    // const newChatRoom = { id: chatRoomId, title: `Chat ${chatRoomId}` };
    // addChatRoom(newChatRoom);
    router.push(`/?conversation=new`);
  };

  return (
    <Sheet>
      {/* Toggle Button for Mobile Nav */}
      <SheetTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="shrink-0 md:hidden dark:text-gray-100 dark:bg-gray-800 dark:border-gray-700"
        >
          <Menu className="h-6 w-6" />
          <span className="sr-only">Toggle navigation menu</span>
        </Button>
      </SheetTrigger>

      {/* Mobile Menu Content */}
      <SheetContent
        side="left"
        className="flex flex-col h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-transform ease-in-out duration-300"
      >
        {/* Header with Logo and New Chat Button */}
        <div className="flex justify-between items-center border-b border-gray-200 dark:border-gray-700 p-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Box className="h-6 w-6" />
            <span>AI Prompting</span>
          </Link>
          <button
            onClick={createNewChat}
            className="flex items-center gap-2 text-sm"
          >
            <SquarePen className="h-5 w-5" />
          </button>
        </div>

        {/* Scrollable Chat Room List */}
        <ScrollArea className="flex-1 px-2 lg:px-4 py-4">
          {loading ? (
            <div className="text-center">Loading chat rooms...</div>
          ) : (
            <nav className="grid items-start text-sm font-medium">
              {chatRooms.map((room) => {
                const isActive = pathname === `/conversation/${room.id}`;
                return (
                  <div key={room.id} className={`flex justify-between items-center hover:bg-gray-200 ${
                    isActive ? "bg-blue-500 text-white rounded-md" : ""
                  }`}>
                    <Link
                      href={`/conversation/${room.id}`}
                      className={`block p-2 flex-1 `}
                    >
                      {room.title}
                    </Link>
                    <button
                      onClick={() => {
                        deleteChatRoom(room.id);
                        if (pathname === `/conversation/${room.id}`) router.push(`/`);
                      }}
                      className="text-red-600 p-1 hover:bg-red-500 rounded"
                    >
                      <Trash2  color="#c7c7c7" className="h-5 w-5" />
                    </button>
                  </div>
                );
              })}
            </nav>
          )}
        </ScrollArea>

        {/* Footer */}
        <footer className="border-t border-gray-200 dark:border-gray-700 p-4 text-center">
          <span>© AI Prompting</span>
        </footer>
      </SheetContent>
    </Sheet>
  );
};

export default MobileNav;
