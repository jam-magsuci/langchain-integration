import {
    LayoutDashboard,
    RocketIcon,

} from 'lucide-react';
import { ReactNode } from "react";

export interface MenuItem {
    link: string;
    text: string;
    icon: ReactNode;
    shortcut?:string;
    soon?:boolean
  }

export interface MenuGroup {
    group: string;
    items: MenuItem[];
  }

  export interface MenuListProps extends Array<MenuGroup> {}

  export const MenuList:MenuListProps = [
    {
        group:"General",
        items:[
            {
                link:"/dashboard",
                text:"Dashboard",
                icon: <LayoutDashboard className="mr-2 h-4 w-4"/>,
                soon:false,
            },
            {
                link:"/chats",
                text:"Email warmup",
                icon: <RocketIcon className="mr-2 h-4 w-4"/>
            }
        ]
    },
];
