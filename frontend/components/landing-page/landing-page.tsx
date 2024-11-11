import ThemeToggler from "@/components/theme/ThemeToggler";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const LandingPage = () => {
    return (
        <>
            <main className="grid min-h-screen w-full bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 dark:from-gray-800 dark:via-gray-900 dark:to-black text-white">
                <div className="h-[100vh] flex flex-col items-center justify-center relative space-y-8">
                    <h1 className="text-4xl md:text-6xl font-bold text-center dark:text-gray-100">
                        Welcome to AI-Prompt
                    </h1>
                    <p className="text-lg md:text-2xl text-center max-w-2xl dark:text-gray-300">
                        Simplify your lead management with our advanced AI tool, designed to enhance your workflow and boost productivity.
                    </p>
                    <Button className="bg-primary px-6 py-3 text-lg font-semibold rounded-md transition transform hover:scale-105 dark:bg-gray-700 dark:text-white">
                        <Link href="/auth">Get Started</Link>
                    </Button>
                    <div className="absolute bottom-5 right-5">
                        <ThemeToggler />
                    </div>
                </div>
            </main>
        </>
    );
}

export default LandingPage;
