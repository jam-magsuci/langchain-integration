'use client'
import * as z from 'zod';
import {useForm} from 'react-hook-form'
import {zodResolver} from '@hookform/resolvers/zod'
import { Form,FormControl,FormField,FormItem,FormLabel,FormMessage } from "@/components/ui/form";
import { Card,CardContent,CardDescription,CardHeader,CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

const formSchema = z.object({
    first_name: z.string().min(1,{
        message:'Name is required'
    }),
    last_name: z.string().min(1,{
        message:'Name is required'
    }),
    email: z.string().min(1,{
        message:'Email is required'
    }).email({
        message:'Please enter a valid email'
    }),
    password: z.string().min(1,{
        message:'Password is required'
    }),
    confirmPassword: z.string().min(1,{
        message:'Confirm Password is required'
    }),
})

const RegisterForm = () => {
    const router = useRouter();
    const form = useForm<z.infer<typeof formSchema>>({
        resolver:zodResolver(formSchema),
        defaultValues:{
            first_name: '',
            last_name: '',
            email: '',
            password:'',
            confirmPassword:''
        }
    });

    const handleSubmit = (data:z.infer<typeof formSchema>) =>{
        router.push('/');
    }

    return(
        <Card>
            <CardHeader>
                <CardTitle>Register</CardTitle>
                <CardDescription>
                    Sign up by adding the information below
                </CardDescription>
                <CardContent className='space-y-2 p-0'>
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
                        <FormField
                        control={form.control}
                        name="first_name"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="uppercase text-xs font-bold text-zinc-500 dark:text-white">First Name</FormLabel>
                                <FormControl>
                                    <Input type="text" className="bg-slate-100 dark:bg-slate-500 border-0 focus-visible:ring-0 text-black dark:text-white focus-visible:ring-offset-0" placeholder="Enter First Name" {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                        /> <FormField
                        control={form.control}
                        name="last_name"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="uppercase text-xs font-bold text-zinc-500 dark:text-white">Last Name</FormLabel>
                                <FormControl>
                                    <Input type="text" className="bg-slate-100 dark:bg-slate-500 border-0 focus-visible:ring-0 text-black dark:text-white focus-visible:ring-offset-0" placeholder="Enter Last Name" {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                        />
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
                        <FormField
                        control={form.control}
                        name="confirmPassword"
                        render={({ field }) => (
                            <FormItem>
                            <FormLabel className="uppercase text-xs font-bold text-zinc-500 dark:text-white">Confirm Password</FormLabel>
                            <FormControl>
                                <Input type="password" className="bg-slate-100 dark:bg-slate-500 border-0 focus-visible:ring-0 text-black dark:text-white focus-visible:ring-offset-0" placeholder="Enter Confirm Password" {...field} />
                            </FormControl>
                            <FormMessage />
                            </FormItem>
                        )}
                        />
                        {/* <Button className='w-full dark:bg-slate-800 dark:text-white'>Sign In</Button> */}
                        <Button className='w-full dark:bg-slate-800 dark:text-white'
                        variant='outline'
                        >Sign Up</Button>
                    </form>
                </Form>
                </CardContent>
            </CardHeader>
        </Card>
    );
}
 
export default RegisterForm;