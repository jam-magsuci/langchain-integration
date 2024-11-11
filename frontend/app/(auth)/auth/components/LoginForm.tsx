'use client'
import * as z from 'zod';
import {useForm} from 'react-hook-form'
import {zodResolver} from '@hookform/resolvers/zod'
import { Form,FormControl,FormField,FormItem,FormLabel,FormMessage } from "@/components/ui/form";
import { Card,CardContent,CardDescription,CardHeader,CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { FcGoogle } from 'react-icons/fc';
import { supabaseBrowserClient } from '@/lib/supabase/client';

const formSchema = z.object({
    email: z.string().min(1,{
        message:'Email is required'
    }).email({
        message:'Please enter a valid email'
    }),
    password: z.string().min(1,{
        message:'Password is required'
    })
})

type ProviderProps = 'google' | 'github';

const LoginForm = () => {
    const router = useRouter();
    const form = useForm<z.infer<typeof formSchema>>({
        resolver:zodResolver(formSchema),
        defaultValues:{
            email: '',
            password:''
        }
    });

    const handleLogin = (data:z.infer<typeof formSchema>) =>{
        router.push('/dashboard');
    }
    
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
                <CardTitle>Login</CardTitle>
                <CardDescription>
                    Login to your account with your credentials
                </CardDescription>
                <CardContent className='space-y-2 p-0'>
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(handleLogin)} className="space-y-6">
                        <FormField
                        control={form.control}
                        name="email"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="uppercase text-xs font-bold text-zinc-500 dark:text-white">Email</FormLabel>
                                <FormControl>
                                    <Input type="email" className="bg-slate-100 dark:bg-slate-500 border-0 focus-visible:ring-0 text-black dark:text-white focus-visible:ring-offset-0" placeholder="Enter Email" {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                        />
                        
                        <FormField
                        control={form.control}
                        name="password"
                        render={({ field }) => (
                            <FormItem>
                            <FormLabel className="uppercase text-xs font-bold text-zinc-500 dark:text-white">Password</FormLabel>
                            <FormControl>
                                <Input type="password" className="bg-slate-100 dark:bg-slate-500 border-0 focus-visible:ring-0 text-black dark:text-white focus-visible:ring-offset-0" placeholder="Enter Password" {...field} />
                            </FormControl>
                            <FormMessage />
                            </FormItem>
                        )}
                        />
                        <Button className='w-full dark:bg-slate-800 dark:text-white'
                        variant='outline'
                        >Sign In</Button>
                    </form>
                </Form>
                <Button className='w-full flex items-center gap-2 dark:bg-slate-800 dark:text-white'
                variant='outline'
                onClick={()=>handleLoginWithOAuth('google')}>
                <FcGoogle/> Google
                </Button>
                </CardContent>
            </CardHeader>
        </Card>
    );
}
 
export default LoginForm;