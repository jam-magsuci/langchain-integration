'use client'
// import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "../ui/dropdown-menu";
// import { Search } from "lucide-react";
import MobileNav from "./mobile-nav";
// import { Input } from "../ui/input";
import ThemeToggler from "../theme/ThemeToggler";
// import useUser from "@/hooks/useUser";
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
// import { Button } from "../ui/button";
// import { signOut } from "@/app/(auth)/auth/actions/Auth";

// const getInitials = (name: string) => {
//     return name
//       .split(' ')
//       .map(part => part[0])
//       .join('');
//   };
  
function Navbar(){
    // const {data} = useUser();
    return (
    <header className="flex h-14 items-center gap-4 border-b bg-muted/40 px-4 lg:h-[60px] lg:px-6">
        <MobileNav/>
        <div className="w-full flex-1">
            {/* <form>
            <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                type="search"
                placeholder="Search heres..."
                className="w-full appearance-none bg-background pl-8 shadow-none md:w-2/3 lg:w-1/3"
                />
            </div>
            </form> */}
        </div>
        
        {/* <Button variant="secondary" size="icon" className="rounded-full">
                    <CircleUser className="h-5 w-5" />
                    <span className="sr-only">Toggle user menu</span>
                </Button> */}
        <div className="flex items-center">
            <ThemeToggler/>
            {/* <DropdownMenu>
                <DropdownMenuTrigger asChild>
               
                <Avatar>
                    <AvatarImage src={data?.image_url||"https://github.com/shadcn.png"} alt={data?.display_name||""} title={data?.display_name||""}/>
                    <AvatarFallback>{getInitials(data?.display_name||"")}</AvatarFallback>
                </Avatar>

                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                <DropdownMenuLabel className="mt-0 pb-0">{data?.display_name||""}</DropdownMenuLabel>
                <DropdownMenuLabel className="mt-0"><small>{data?.email||""}</small></DropdownMenuLabel>
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Settings</DropdownMenuItem>
                <DropdownMenuItem>Support</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                    <form action={signOut}>
                        <Button variant='ghost' className="text-left p-0 m-0">Logout</Button>
                    </form>
                </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu> */}
        </div>
    </header>
    );
}
 
export default Navbar;