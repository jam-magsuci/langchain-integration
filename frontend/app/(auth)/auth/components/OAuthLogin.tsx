'use client'
import { Card,CardContent,CardDescription,CardHeader,CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FcGoogle } from 'react-icons/fc';
import { supabaseBrowserClient } from '@/lib/supabase/client';

type ProviderProps = 'google' | 'github';

const OAuthLogin = () => {
    const handleLoginWithOAuth= async (provider:ProviderProps)=>{
        const supabase = supabaseBrowserClient();
        return await supabase.auth.signInWithOAuth({
        provider,
            options: {
                redirectTo: location.origin+"/auth/callback",
            },
        })
    }

    return(
        <Card>
            <CardHeader>
                <CardTitle className="text-center">Welcome to AI-Prompt</CardTitle>
                <CardDescription>
                    Login/SignUp with your Google Account
                </CardDescription>  
                <CardContent className='space-y-2 p-0 pt-4'>
                <Button className='w-full flex items-center gap-2 dark:bg-slate-800 dark:text-white'
                variant='outline'
                onClick={()=>handleLoginWithOAuth('google')}>
                <FcGoogle/> Sign In With Google
                </Button>
                </CardContent>
            </CardHeader>
        </Card>
    );
}
 
export default OAuthLogin;