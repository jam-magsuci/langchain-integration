import ThemeToggler from "@/components/theme/ThemeToggler";
import { redirect } from 'next/navigation'
import { supabaseServer } from '@/lib/supabase/server'

const AuthLayout =async ({children}:{children:React.ReactNode}) => {
    
    const supabase = supabaseServer()

    const { data, error } = await supabase.auth.getUser()
    if (!error || data?.user) {
      redirect('/dashboard')
    }

    return ( 
        <div className="h-[100vh] flex items-center justify-center relative">
            <div className="absolute bottom-5 right-0 text-white">
                <ThemeToggler/>
            </div>
            {children}
        </div>
     );
}
 
export default AuthLayout;